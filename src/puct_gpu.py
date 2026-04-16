"""
puct_gpu.py
-----------
GPU-accelerated PUCT (Predictor-Upper Confidence bounds applied to Trees)
for multi-agent, continuous-state, neural-network-guided search.

Design
------
``PUCTGpu`` is the top-level class, analogous to ``MCTSNC`` but adapted for:

* **Continuous states** — float32 vectors instead of int8 boards.
* **Neural network guidance** — policy and value networks replace random
  playouts.  NN calls happen on the host (PyTorch) between two GPU kernel
  dispatches.
* **Multi-agent turn model** — ``robot_turn ∈ {0..num_robots-1}`` replaces
  the binary ``{-1, +1}`` flip.
* **Progressive widening** — children are added lazily according to
  ``ceil(C_pw * N^alpha_pw)``, capped at max_actions.
* **Root parallelism** — N independent trees, each managed by one CUDA block.

3-phase iteration sandwich per step
------------------------------------
  Kernel A  :  _select_puct + _extract_leaf_states
  Host      :  PyTorch batched inference (policy + value)
  Kernel B  :  _expand_and_backup_puct

References
----------
- Migration plan Phases 0-11 (gpu_puct_migration_plan.md).
- mctsnc.py (MCTSNC class) as structural reference.
- puct_v1.hpp as algorithmic reference.
"""

import sys
import warnings
import time
import math

import numpy as np
from numpy import inf
import torch

# IMPORTANT: PyTorch must own the CUDA primary context before Numba does.
if torch.cuda.is_available():
    _torch_init_sentinel = torch.zeros(1, device="cuda")

# Patch Numba's CTK_SUPPORTED for CUDA toolkit versions >= 12.5 (sm_89 / RTX 4090)
try:
    from numba.cuda.cudadrv import nvvm as _nvvm_mod
    if hasattr(_nvvm_mod, "CTK_SUPPORTED"):
        _ctk = _nvvm_mod.CTK_SUPPORTED
        for _v in [(12, 5), (12, 6), (12, 7), (12, 8), (12, 9)]:
            if _v not in _ctk:
                _ctk[_v] = ((5, 0), (9, 0))
except Exception:
    pass

from numba import cuda
from numba import int8, int16, int32, int64, float32

from puct_gpu_kernels import (
    _reset_puct,
    _select_puct,
    _extract_leaf_states,
    _expand_and_backup_puct,
    _reduce_over_trees_puct,
    _reduce_over_actions_puct,
)
from puct_gpu_nn_bridge import NumbaPytorchBridge

__version__ = "0.1.0"
__author__ = "Antigravity / GPU PUCT Migration"


class PUCTGpu:
    """
    GPU-accelerated PUCT search.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the continuous state vector.
    action_dim : int
        Dimensionality of the continuous action vector stored per node.
    max_actions : int
        Maximum branching factor (number of discrete action slots per node).
        Must be ≤ 512 (shared-memory limit).
    num_robots : int
        Number of agents (robots / players).
    n_trees : int
        Number of independent trees for root parallelism.  Defaults to 8.
    max_simulations : int
        Budget: maximum number of Select→NN→Expand+Backup iterations.
    search_time_limit : float
        Wall-clock time budget [s].  ``np.inf`` means use only
        ``max_simulations``.
    C_exp : float
        Exploration constant in the PUCT UCB formula.
    alpha_exp : float
        Parent-count exponent (alpha_exp=0.5 ≈ standard UCT log; 1.0 = linear).
    C_pw : float
        Progressive widening coefficient.
    alpha_pw : float
        Progressive widening exponent.
    gamma : float
        Discount factor for the value backup.
    device_memory : float
        GPU memory budget [GiB] for tree arrays.
    verbose_info : bool
        Print info after ``run()``.
    verbose_debug : bool
        Print per-step debug info during ``run()``.
    """

    # ── Class-level constants ────────────────────────────────────────────────
    MAX_TREE_SIZE = 2 ** 22          # hard cap; ~4M nodes
    MAX_TREE_DEPTH = 2048            # for path storage
    MAX_MAX_ACTIONS = 512            # shared-memory limit

    DEFAULT_N_TREES = 8
    DEFAULT_MAX_SIMULATIONS = 800
    DEFAULT_SEARCH_TIME_LIMIT = np.inf
    DEFAULT_C_EXP = 1.0
    DEFAULT_ALPHA_EXP = 0.5
    DEFAULT_C_PW = 1.0
    DEFAULT_ALPHA_PW = 0.5
    DEFAULT_GAMMA = 0.99
    DEFAULT_DEVICE_MEMORY = 2.0     # GiB

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_actions: int,
        num_robots: int = 1,
        n_trees: int = DEFAULT_N_TREES,
        max_simulations: int = DEFAULT_MAX_SIMULATIONS,
        search_time_limit: float = DEFAULT_SEARCH_TIME_LIMIT,
        C_exp: float = DEFAULT_C_EXP,
        alpha_exp: float = DEFAULT_ALPHA_EXP,
        C_pw: float = DEFAULT_C_PW,
        alpha_pw: float = DEFAULT_ALPHA_PW,
        gamma: float = DEFAULT_GAMMA,
        device_memory: float = DEFAULT_DEVICE_MEMORY,
        verbose_info: bool = True,
        verbose_debug: bool = False,
    ):
        self._check_cuda()

        # ── Validate and store parameters ────────────────────────────────────
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_actions = int(max_actions)
        self.num_robots = int(num_robots)
        self.n_trees = int(n_trees)
        self.max_simulations = int(max_simulations)
        self.search_time_limit = float(search_time_limit)
        self.C_exp = float(C_exp)
        self.alpha_exp = float(alpha_exp)
        self.C_pw = float(C_pw)
        self.alpha_pw = float(alpha_pw)
        self.gamma = float(gamma)
        self.device_memory_bytes = int(device_memory * 1024 ** 3)
        self.verbose_info = bool(verbose_info)
        self.verbose_debug = bool(verbose_debug)

        if self.max_actions > self.MAX_MAX_ACTIONS:
            sys.exit(
                f"[PUCTGpu: max_actions={self.max_actions} exceeds "
                f"MAX_MAX_ACTIONS={self.MAX_MAX_ACTIONS}]"
            )
        if self.num_robots > 8:
            sys.exit("[PUCTGpu: num_robots > 8 not supported (shared memory limit)]")

    # ── CUDA availability ────────────────────────────────────────────────────

    def _check_cuda(self):
        # cuda.is_available() has a known false-negative on some Numba versions;
        # fall back to checking the GPU list directly.
        cuda_ok = cuda.is_available()
        if not cuda_ok:
            try:
                cuda_ok = len(list(cuda.gpus)) > 0
            except Exception:
                cuda_ok = False
        if not cuda_ok:
            sys.exit("[PUCTGpu: CUDA not available]")
        device = cuda.get_current_device()
        self.cuda_tpb_default = device.MAX_THREADS_PER_BLOCK // 2

    # ── Memory calculation & device array allocation ──────────────────────────

    def init_device_side_arrays(self):
        """
        Allocate all GPU device arrays based on the memory budget.

        Per-state memory breakdown (Appendix A of migration plan)
        ----------------------------------------------------------
        float32 * state_dim            state vector
        float32 * action_dim           action that created this node
        float32 * num_robots           total_value vector
        int32                          ns (visit count)
        int32  * (1 + max_actions)     tree array row (parent + children)
        int16                          depth
        int8                           robot_turn
        bool * 2                       leaf + terminal flags
        float32 * max_actions          action_priors
        int16  * max_actions           prior_rank
        int16                          pw_boundary
        """
        if self.verbose_info:
            print(f"[PUCTGpu.init_device_side_arrays()... for {self}]")
        t0 = time.time()

        f32 = np.float32().itemsize   # 4
        i32 = np.int32().itemsize     # 4
        i16 = np.int16().itemsize     # 2
        i8  = np.int8().itemsize      # 1
        i64 = np.int64().itemsize     # 8

        per_state_memory = (
            f32 * self.state_dim
            + f32 * self.action_dim
            + f32 * self.num_robots
            + i32                                        # ns
            + i32 * (1 + self.max_actions)              # tree row
            + i16                                        # depth
            + i8                                         # robot_turn
            + 2                                          # leaf + terminal (bool)
            + f32 * self.max_actions                    # action_priors
            + i16 * self.max_actions                    # prior_rank
            + i16                                        # pw_boundary
        )

        # Per-tree overhead (path, selected node, sizes)
        per_tree_overhead = (
            i32                                           # trees_sizes
            + i32                                         # trees_nodes_selected
            + i32 * (self.MAX_TREE_DEPTH + 2)            # selected_paths
        )

        available = self.device_memory_bytes - self.n_trees * per_tree_overhead
        self.max_tree_size = available // (per_state_memory * self.n_trees)
        self.max_tree_size = min(int(self.max_tree_size), self.MAX_TREE_SIZE)
        self.max_tree_size = max(self.max_tree_size, 2)  # at least root + 1 child

        if self.verbose_info:
            print(
                f"[PUCTGpu: per_state_memory={per_state_memory} B, "
                f"max_tree_size={self.max_tree_size}]"
            )

        T = self.n_trees
        S = self.max_tree_size
        A = self.max_actions
        R = self.num_robots
        D = self.state_dim
        Da = self.action_dim

        # ── Topology (kept from MCTSNC) ──────────────────────────────────────
        self.dev_trees = cuda.device_array((T, S, 1 + A), dtype=np.int32)
        self.dev_trees_sizes = cuda.device_array(T, dtype=np.int32)
        self.dev_trees_depths = cuda.device_array((T, S), dtype=np.int16)
        self.dev_trees_nodes_selected = cuda.device_array(T, dtype=np.int32)
        self.dev_trees_selected_paths = cuda.device_array(
            (T, self.MAX_TREE_DEPTH + 2), dtype=np.int32
        )

        # ── Node flags / scalars (kept + renamed) ────────────────────────────
        self.dev_trees_leaves = cuda.device_array((T, S), dtype=bool)
        self.dev_trees_terminals = cuda.device_array((T, S), dtype=bool)
        self.dev_trees_ns = cuda.device_array((T, S), dtype=np.int32)

        # ── MODIFIED from MCTSNC ──────────────────────────────────────────────
        # turns: 0..num_robots-1 (was binary {-1,+1})
        self.dev_trees_robot_turns = cuda.device_array((T, S), dtype=np.int8)
        # total_value: float32 per-robot (was int32 ns_wins)
        self.dev_trees_total_value = cuda.device_array((T, S, R), dtype=np.float32)

        # ── NEW arrays ────────────────────────────────────────────────────────
        self.dev_trees_states = cuda.device_array((T, S, D), dtype=np.float32)
        self.dev_trees_action_params = cuda.device_array((T, S, Da), dtype=np.float32)
        self.dev_trees_action_priors = cuda.device_array((T, S, A), dtype=np.float32)
        self.dev_trees_prior_rank = cuda.device_array((T, S, A), dtype=np.int16)
        self.dev_trees_pw_boundary = cuda.device_array((T, S), dtype=np.int16)

        # ── Reduction output arrays ───────────────────────────────────────────
        self.dev_actions_ns = cuda.device_array(A, dtype=np.int64)
        self.dev_actions_total_value = cuda.device_array((A, R), dtype=np.float32)
        self.dev_best_action = cuda.device_array(1, dtype=np.int32)
        self.dev_best_n = cuda.device_array(1, dtype=np.int64)

        # ── NN bridge (shared GPU memory Numba ↔ PyTorch) ─────────────────────
        self.bridge = NumbaPytorchBridge(T, D, Da, A, R)

        # ── Thread-per-block settings ─────────────────────────────────────────
        self.tpb_r = min(
            int(2 ** math.ceil(math.log2(max(D, 1)))),
            self.cuda_tpb_default
        )
        self.tpb_s = self.cuda_tpb_default
        self.tpb_rot = min(
            int(2 ** math.ceil(math.log2(max(T, 1)))),
            self.cuda_tpb_default
        )
        self.tpb_roa = min(
            int(2 ** math.ceil(math.log2(max(A, 1)))),
            self.cuda_tpb_default
        )

        elapsed = time.time() - t0
        if self.verbose_info:
            print(f"[PUCTGpu.init_device_side_arrays() done; time: {elapsed:.3f} s]")

    # ── Main run loop ────────────────────────────────────────────────────────

    def run(self, root_state: np.ndarray, root_turn: int,
            policy_model, value_model):
        """
        Run PUCT search from *root_state* and return the best action index.

        Parameters
        ----------
        root_state : np.ndarray, shape (state_dim,), float32
            Initial continuous state vector.
        root_turn : int
            Index of the robot that acts at the root (0..num_robots-1).
        policy_model : torch.nn.Module
            Maps (B, state_dim) → (B, max_actions).  Called with no_grad.
        value_model : torch.nn.Module
            Maps (B, state_dim) → (B, num_robots).  Called with no_grad.

        Returns
        -------
        best_action : int
            Index of the most-visited root action.
        best_n : int
            Total visit count for the best action.
        actions_info : dict
            Per-action statistics: ns, q (= total_value / ns), for robot 0.
        """
        if self.verbose_info:
            print(f"PUCTGpu RUN... [{self}]")
        t_start = time.time()

        root_state = np.asarray(root_state, dtype=np.float32)
        dev_root_state = cuda.to_device(root_state)

        # ── Reset ────────────────────────────────────────────────────────────
        bpg_r = self.n_trees
        tpb_r = self.tpb_r

        _reset_puct[bpg_r, tpb_r](
            dev_root_state, np.int8(root_turn),
            self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths,
            self.dev_trees_robot_turns,
            self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns,
            self.dev_trees_total_value, self.dev_trees_states,
            self.dev_trees_action_priors, self.dev_trees_pw_boundary,
        )
        cuda.synchronize()

        # ── Timing accumulators ───────────────────────────────────────────────
        self.time_select = 0.0
        self.time_extract = 0.0
        self.time_nn = 0.0
        self.time_expand_backup = 0.0
        self.steps = 0

        max_actions_i32 = np.int32(self.max_actions)
        max_tree_size_i32 = np.int32(self.max_tree_size)
        state_dim_i32 = np.int32(self.state_dim)
        num_robots_i32 = np.int32(self.num_robots)
        gamma_f32 = np.float32(self.gamma)
        C_pw_f32 = np.float32(self.C_pw)
        alpha_pw_f32 = np.float32(self.alpha_pw)
        C_exp_f32 = np.float32(self.C_exp)
        alpha_exp_f32 = np.float32(self.alpha_exp)

        t_loop_start = time.time()

        for step in range(self.max_simulations):
            # ── Wall-clock check ──────────────────────────────────────────────
            if time.time() - t_loop_start >= self.search_time_limit:
                break

            if self.verbose_debug:
                print(f"  [step {step + 1}]")

            # ── KERNEL A: Selection ───────────────────────────────────────────
            t1 = time.time()
            _select_puct[self.n_trees, self.tpb_s](
                C_exp_f32, alpha_exp_f32,
                self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns,
                self.dev_trees_total_value,
                self.dev_trees_robot_turns, self.dev_trees_action_priors,
                self.dev_trees_pw_boundary,
                self.dev_trees_prior_rank,
                self.dev_trees_nodes_selected, self.dev_trees_selected_paths,
            )
            cuda.synchronize()
            self.time_select += time.time() - t1

            # ── KERNEL A2: Extract leaf states ────────────────────────────────
            t1 = time.time()
            _extract_leaf_states[self.n_trees, self.tpb_r](
                self.dev_trees_nodes_selected, self.dev_trees_states,
                self.dev_trees_terminals,
                self.bridge.dev_leaf_states, self.bridge.dev_leaf_valid,
            )
            cuda.synchronize()
            self.time_extract += time.time() - t1

            # ── HOST: PyTorch batch inference (zero-copy) ─────────────────────
            t1 = time.time()
            with torch.no_grad():
                valid_mask = self.bridge.leaf_valid.bool()   # (n_trees,)
                if valid_mask.any():
                    batch_states = self.bridge.leaf_states   # zero-copy view
                    priors = policy_model(batch_states)      # (n_trees, max_actions)
                    values = value_model(batch_states)        # (n_trees, num_robots)
                    # Normalise priors
                    priors = torch.softmax(priors, dim=-1)
                    # NaN guard
                    if torch.isnan(priors).any():
                        priors = torch.nan_to_num(priors, nan=1.0 / self.max_actions)
                    if torch.isnan(values).any():
                        values = torch.nan_to_num(values, nan=0.0)
                    # Write back into shared GPU tensors (still zero-copy for same device)
                    self.bridge.nn_priors.copy_(priors)
                    self.bridge.nn_values.copy_(values)
            self.time_nn += time.time() - t1

            # ── KERNEL B: Expand + Backup ─────────────────────────────────────
            t1 = time.time()
            _expand_and_backup_puct[self.n_trees, self.tpb_s](
                gamma_f32, C_pw_f32, alpha_pw_f32, num_robots_i32,
                self.bridge.dev_nn_priors, self.bridge.dev_nn_values,
                self.bridge.dev_leaf_valid,
                self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths,
                self.dev_trees_robot_turns,
                self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns,
                self.dev_trees_total_value, self.dev_trees_states,
                self.dev_trees_action_priors,
                self.dev_trees_prior_rank, self.dev_trees_pw_boundary,
                self.dev_trees_nodes_selected, self.dev_trees_selected_paths,
                max_tree_size_i32, state_dim_i32, max_actions_i32,
            )
            cuda.synchronize()
            self.time_expand_backup += time.time() - t1

            self.steps += 1

        self.time_loop = time.time() - t_loop_start

        # ── Reduction: sum over trees ─────────────────────────────────────────
        t1 = time.time()
        _reduce_over_trees_puct[self.max_actions, self.tpb_rot](
            self.dev_trees, self.dev_trees_ns, self.dev_trees_total_value,
            num_robots_i32,
            self.dev_actions_ns, self.dev_actions_total_value,
        )
        cuda.synchronize()
        self.time_reduce_trees = time.time() - t1

        # ── Reduction: argmax over actions ────────────────────────────────────
        t1 = time.time()
        _reduce_over_actions_puct[1, self.tpb_roa](
            self.dev_actions_ns,
            self.dev_best_action, self.dev_best_n,
        )
        cuda.synchronize()
        self.time_reduce_actions = time.time() - t1

        # ── Copy results to host ──────────────────────────────────────────────
        best_action = int(self.dev_best_action.copy_to_host()[0])
        best_n = int(self.dev_best_n.copy_to_host()[0])
        actions_ns_host = self.dev_actions_ns.copy_to_host()
        actions_tv_host = self.dev_actions_total_value.copy_to_host()  # (A, R)

        # Build per-action info dict (robot 0's perspective)
        actions_info = {}
        for a in range(self.max_actions):
            n = int(actions_ns_host[a])
            if n == 0:
                continue
            q0 = float(actions_tv_host[a, 0]) / n
            actions_info[a] = {"n": n, "q": q0}
        actions_info["best"] = {"index": best_action, "n": best_n}

        self.time_total = time.time() - t_start
        self.best_action = best_action
        self.best_n = best_n
        self.actions_info = actions_info

        if self.verbose_info:
            self._print_info()

        return best_action, best_n, actions_info

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def _print_info(self):
        ms = 1e3
        print(
            f"PUCTGpu RUN DONE. ["
            f"time: {self.time_total:.3f} s | "
            f"steps: {self.steps} | "
            f"best_action: {self.best_action}, best_n: {self.best_n}]\n"
            f"  times [ms]: "
            f"select={self.time_select*ms:.1f}, "
            f"extract={self.time_extract*ms:.1f}, "
            f"nn={self.time_nn*ms:.1f}, "
            f"expand_backup={self.time_expand_backup*ms:.1f}, "
            f"reduce_trees={self.time_reduce_trees*ms:.1f}, "
            f"reduce_actions={self.time_reduce_actions*ms:.1f}"
        )
        print(f"  actions_info: {self.actions_info}")

    def get_tree_stats(self):
        """Return dict with mean/max tree size and depth after last run."""
        sizes = self.dev_trees_sizes.copy_to_host()
        depths = self.dev_trees_depths.copy_to_host()
        stats = {
            "mean_size": float(np.mean(sizes)),
            "max_size": int(np.max(sizes)),
        }
        all_depths = []
        for ti in range(self.n_trees):
            all_depths.extend(depths[ti, : sizes[ti]].tolist())
        if all_depths:
            stats["mean_depth"] = float(np.mean(all_depths))
            stats["max_depth"] = int(np.max(all_depths))
        return stats

    # ── String representation ─────────────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"PUCTGpu("
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"max_actions={self.max_actions}, num_robots={self.num_robots}, "
            f"n_trees={self.n_trees}, max_simulations={self.max_simulations}, "
            f"C_exp={self.C_exp}, alpha_exp={self.alpha_exp}, "
            f"C_pw={self.C_pw}, alpha_pw={self.alpha_pw}, "
            f"gamma={self.gamma})"
        )

    def __repr__(self) -> str:
        return str(self)
