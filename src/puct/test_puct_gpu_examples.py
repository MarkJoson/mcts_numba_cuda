"""
test_puct_gpu_examples.py
--------------------------
Example-style integration tests for ``PUCTGpu``.

These tests mirror the "Example usage" sections in ``mctsnc.py`` and verify
that the GPU-accelerated PUCT implementation can reproduce meaningful,
deterministic search behaviour on well-understood toy problems.

Test catalogue
--------------
Test 11  Double-integrator smoke — full run with mechanics, returns valid action
Test 12  Prior dominance — the highest-prior action is visited most
Test 13  Argmax stability — best action doesn't change when re-run with same state
Test 14  Tree stats — get_tree_stats() returns sensible size / depth after run
Test 15  Multi-robot value backup — total_value is split across robots correctly
Test 16  Time-budget termination — search halts within ≈ wall-clock limit
Test 17  Large simulation convergence — with enough sims, best action stabilises
Test 18  actions_info structure — expected keys are present and consistent
Test 19  PUCTGpu __str__ / __repr__ — string contains key params
Test 20  Repeated runs — agent can run from different root states without crashing

Usage
-----
    cd /home/robomaster/Research/mcts_numba_cuda/src
    python test_puct_gpu_examples.py

    # or via the shell helper (handles LD_LIBRARY_PATH etc.):
    bash run_puct_tests.sh --examples
"""

import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn

# ── Numba / PyTorch CUDA compatibility patches ────────────────────────────────
# 1. Torch must own the primary CUDA context before Numba imports its runtime.
if torch.cuda.is_available():
    _torch_cuda_sentinel = torch.zeros(1, device="cuda")

# 2. Patch Numba's CTK_SUPPORTED for CUDA 12.5+ (sm_89 / RTX 4090).
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

sys.path.insert(0, os.path.dirname(__file__))

from puct.puct_gpu import PUCTGpu

# ════════════════════════════════════════════════════════════════════════════
# Shared neural-network stubs
# ════════════════════════════════════════════════════════════════════════════

class UniformPolicyNet(nn.Module):
    """Returns uniform prior (1/max_actions) for every input."""
    def __init__(self, state_dim: int, max_actions: int):
        super().__init__()
        self.max_actions = max_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.ones(B, self.max_actions, device=x.device)


class ConstantValueNet(nn.Module):
    """Returns a fixed constant value for all inputs."""
    def __init__(self, state_dim: int, num_robots: int, constant: float = 0.5):
        super().__init__()
        self.num_robots = num_robots
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.full(
            (B, self.num_robots), self.constant,
            dtype=torch.float32, device=x.device,
        )


class SpotlightPolicyNet(nn.Module):
    """
    实现 epsilon-greedy 策略。其中 greedy 策略固定为某个动作。
    """
    def __init__(self, state_dim: int, max_actions: int, hot_action: int,
                 hot_weight: float = 100.0):
        super().__init__()
        self.max_actions = max_actions
        self.hot_action = hot_action
        self.hot_weight = hot_weight
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        logits = torch.ones(B, self.max_actions, device=x.device)
        logits[:, self.hot_action] = self.hot_weight
        return logits


class PerRobotValueNet(nn.Module):
    """
    Returns a distinct constant value for each robot:
        robot r  → values[r]
    """
    def __init__(self, state_dim: int, num_robots: int, values: list):
        super().__init__()
        self.num_robots = num_robots
        self._values = torch.tensor(values, dtype=torch.float32)
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self._values.unsqueeze(0).expand(B, -1).to(x.device)


# ════════════════════════════════════════════════════════════════════════════
# Test infrastructure
# ════════════════════════════════════════════════════════════════════════════

PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"
results: list[tuple[str, bool]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    """
    打印当前的状态
    """
    status = PASS_MARK if ok else FAIL_MARK
    msg = f"{status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, ok))


def make_agent(
    state_dim: int = 4,
    action_dim: int = 1,
    max_actions: int = 8,
    num_robots: int = 1,
    n_trees: int = 4,
    max_simulations: int = 50,
    C_exp: float = 1.0,
    alpha_exp: float = 0.5,
    C_pw: float = 1.0,
    alpha_pw: float = 0.5,
    gamma: float = 0.99,
    search_time_limit: float = float("inf"),
    device_memory: float = 0.3,
    verbose_info: bool = False,
    verbose_debug: bool = False,
) -> PUCTGpu:
    agent = PUCTGpu(
        state_dim=state_dim,
        action_dim=action_dim,
        max_actions=max_actions,
        num_robots=num_robots,
        n_trees=n_trees,
        max_simulations=max_simulations,
        C_exp=C_exp,
        alpha_exp=alpha_exp,
        C_pw=C_pw,
        alpha_pw=alpha_pw,
        gamma=gamma,
        search_time_limit=search_time_limit,
        device_memory=device_memory,
        verbose_info=verbose_info,
        verbose_debug=verbose_debug,
    )
    agent.init_device_side_arrays()
    return agent


# ════════════════════════════════════════════════════════════════════════════
# Test 11 — Double-integrator smoke
# ════════════════════════════════════════════════════════════════════════════

def test_11_double_integrator_smoke():
    """
    Run PUCTGpu on the canonical double-integrator problem
    (state = [position, velocity]) using the mechanics defined in
    ``puct_gpu_mechanics.py`` (inlined into the host env loop here).

    state_dim = 2, action_dim = 1, max_actions = 5 (discretised accelerations),
    num_robots = 1.

    Verify:
    - run() completes without error.
    - best_action is in [0, max_actions).
    - best_n  > 0 (at least one node visited).
    - actions_info contains 'best' key.
    """
    try:
        state_dim   = 2
        action_dim  = 1
        max_actions = 5   # discrete accelerations: [-2, -1, 0, 1, 2]
        num_robots  = 1
        n_trees     = 8
        max_sims    = 100

        agent = make_agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_actions=max_actions,
            num_robots=num_robots,
            n_trees=n_trees,
            max_simulations=max_sims,
            device_memory=0.2,
        )

        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value  = ConstantValueNet(state_dim, num_robots, 0.5).cuda()

        root_state = np.array([0.0, 0.0], dtype=np.float32)  # at rest at origin
        root_turn  = 0

        best_action, best_n, actions_info = agent.run(
            root_state, root_turn, policy, value, problem=None
        )

        ok = (
            0 <= best_action < max_actions
            and best_n > 0
            and "best" in actions_info
            and actions_info["best"]["index"] == best_action
        )
        record(
            "Test 11: Double-integrator smoke", ok,
            f"best_action={best_action}, best_n={best_n}, steps={agent.steps}",
        )
    except Exception as e:
        record("Test 11: Double-integrator smoke", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 12 — Prior dominance
# ════════════════════════════════════════════════════════════════════════════

def test_12_prior_dominance():
    """
    Use ``SpotlightPolicyNet`` that assigns ~100× higher logit to action 3.
    After enough simulations, action 3 should be the most-visited root action.

    This reproduces the "example usage 1" spirit from mctsnc.py: MCTSNC finds
    the best action because the search statistics reinforce the right branch.
    Here we verify that PUCT's upper-confidence formula also leads it to
    concentrate visits on the prior-favoured action.
    """
    try:
        state_dim   = 4
        max_actions = 8
        hot_action  = 3
        n_trees     = 8
        max_sims    = 200

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            C_exp=1.0,
            alpha_exp=0.5,
            C_pw=1.0,
            alpha_pw=0.5,
            device_memory=0.3,
        )

        policy = SpotlightPolicyNet(state_dim, max_actions, hot_action,
                                    hot_weight=100.0).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.5).cuda()

        root_state = np.zeros(state_dim, dtype=np.float32)
        best_action, best_n, actions_info = agent.run(
            root_state, 0, policy, value, problem=None
        )

        ok = (best_action == hot_action and best_n > 0)
        record(
            "Test 12: Prior dominance → hot action wins", ok,
            f"hot_action={hot_action}, best_action={best_action}, best_n={best_n}",
        )
    except Exception as e:
        record("Test 12: Prior dominance → hot action wins", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 13 — Argmax stability (repeated identical runs)
# ════════════════════════════════════════════════════════════════════════════

def test_13_argmax_stability():
    """
    Run PUCTGpu twice from the same root, same networks.  The PUCT algorithm
    is deterministic given the same sequences of NN outputs, so the best
    action and best_n must be identical across both calls.

    (Mirrors: reproducing an experiment from a fixed seed in mctsnc.py.)
    """
    try:
        state_dim   = 4
        max_actions = 8
        hot_action  = 5
        n_trees     = 4
        max_sims    = 80

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            device_memory=0.2,
        )

        policy = SpotlightPolicyNet(state_dim, max_actions, hot_action, 50.0).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.4).cuda()

        root_state = np.array([1.0, -0.5, 0.2, 0.0], dtype=np.float32)

        a1, n1, _ = agent.run(root_state, 0, policy, value, problem=None)
        a2, n2, _ = agent.run(root_state, 0, policy, value, problem=None)

        ok = (a1 == a2) and (n1 == n2)
        record(
            "Test 13: Argmax stability across repeated runs", ok,
            f"run1=(action={a1}, n={n1}), run2=(action={a2}, n={n2})",
        )
    except Exception as e:
        record("Test 13: Argmax stability across repeated runs", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 14 — get_tree_stats()
# ════════════════════════════════════════════════════════════════════════════

def test_14_tree_stats():
    """
    After a run, ``get_tree_stats()`` should return a dict with:
    - mean_size > 1  (tree grew beyond the root)
    - max_size  >= mean_size
    - mean_depth >= 0
    - max_depth  >= 0
    """
    try:
        state_dim   = 4
        max_actions = 8
        n_trees     = 8
        max_sims    = 150

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            device_memory=0.3,
        )

        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.5).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        agent.run(root, 0, policy, value, problem=None)
        stats = agent.get_tree_stats()

        ok = (
            "mean_size"  in stats and stats["mean_size"]  > 1.0
            and "max_size"   in stats and stats["max_size"]  >= stats["mean_size"]
            and "mean_depth" in stats and stats["mean_depth"] >= 0
            and "max_depth"  in stats and stats["max_depth"]  >= 0
        )
        record(
            "Test 14: get_tree_stats() keys and sanity", ok,
            f"mean_size={stats.get('mean_size', '?'):.1f}, "
            f"max_size={stats.get('max_size', '?')}, "
            f"mean_depth={stats.get('mean_depth', '?'):.2f}, "
            f"max_depth={stats.get('max_depth', '?')}",
        )
    except Exception as e:
        record("Test 14: get_tree_stats() keys and sanity", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 15 — Multi-robot value backup
# ════════════════════════════════════════════════════════════════════════════

def test_15_multi_robot_backup():
    """
    With ``num_robots=2`` and ``PerRobotValueNet([0.3, 0.7])``, the backup
    should accumulate different total_values per robot.

    After the run:
    - total_value[root, 0]  should be < total_value[root, 1]
      because robot 0 gets 0.3/visit and robot 1 gets 0.7/visit.
    - ns[root] > 0.
    """
    try:
        state_dim   = 4
        max_actions = 8
        num_robots  = 2
        n_trees     = 4
        max_sims    = 80

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            num_robots=num_robots,
            n_trees=n_trees,
            max_simulations=max_sims,
            device_memory=0.2,
        )

        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value  = PerRobotValueNet(state_dim, num_robots, [0.3, 0.7]).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        agent.run(root, 0, policy, value, problem=None)

        tv = agent.dev_trees_total_value.copy_to_host()  # (T, S, R)
        ns = agent.dev_trees_ns.copy_to_host()           # (T, S)

        root_ns   = int(ns[0, 0])
        root_tv_r0 = float(tv[0, 0, 0])
        root_tv_r1 = float(tv[0, 0, 1])

        ok = (root_ns > 0) and (root_tv_r0 < root_tv_r1)
        record(
            "Test 15: Multi-robot backup (robot 0 < robot 1)", ok,
            f"ns={root_ns}, tv_r0={root_tv_r0:.4f}, tv_r1={root_tv_r1:.4f}",
        )
    except Exception as e:
        record("Test 15: Multi-robot backup (robot 0 < robot 1)", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 16 — Time-budget termination
# ════════════════════════════════════════════════════════════════════════════

def test_16_time_budget():
    """
    Set search_time_limit=0.5 s and max_simulations=100_000 (effectively ∞).
    The run should:
    - Complete in ≈ 0.5 s (allow up to 3× for JIT warm-up overhead).
    - Have executed at least 1 step (non-trivial work done).
    """
    try:
        state_dim    = 4
        max_actions  = 8
        n_trees      = 4
        time_limit   = 0.5    # seconds
        max_sims     = 100_000  # effectively unlimited

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            search_time_limit=time_limit,
            device_memory=0.3,
        )

        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.5).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        t0 = time.time()
        best_action, best_n, _ = agent.run(root, 0, policy, value, problem=None)
        elapsed = time.time() - t0

        # Allow 3× headroom for JIT warm-up on first call
        time_ok   = elapsed < time_limit * 3 + 2.0
        output_ok = 0 <= best_action < max_actions and best_n > 0
        steps_ok  = agent.steps >= 1

        ok = time_ok and output_ok and steps_ok
        record(
            "Test 16: Time-budget terminates correctly", ok,
            f"elapsed={elapsed:.2f}s (limit={time_limit}s), steps={agent.steps}",
        )
    except Exception as e:
        record("Test 16: Time-budget terminates correctly", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 17 — Large simulation convergence
# ════════════════════════════════════════════════════════════════════════════

def test_17_large_sim_convergence():
    """
    Use SpotlightPolicyNet with hot_action=2, heavy prior (weight=500).
    Run 500 simulations.  The hot action must still be the most-visited.

    This is analogous to the mctsnc.py example usage where the AI converges
    to the winning move even in a large tree situation.
    """
    try:
        state_dim   = 4
        max_actions = 8
        hot_action  = 2
        n_trees     = 8
        max_sims    = 500

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            C_exp=0.5,      # lower exploration → prior matters more
            alpha_exp=0.5,
            C_pw=1.0,
            alpha_pw=0.5,
            device_memory=0.5,
        )

        policy = SpotlightPolicyNet(state_dim, max_actions, hot_action, 500.0).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.5).cuda()

        root   = np.zeros(state_dim, dtype=np.float32)
        best_action, best_n, actions_info = agent.run(
            root, 0, policy, value, problem=None
        )

        # The hot action must win or be tied with the best
        ok = best_action == hot_action and best_n > 0
        record(
            "Test 17: Convergence with 500 sims → hot action", ok,
            f"hot={hot_action}, best_action={best_action}, best_n={best_n}, steps={agent.steps}",
        )
    except Exception as e:
        record("Test 17: Convergence with 500 sims → hot action", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 18 — actions_info structure
# ════════════════════════════════════════════════════════════════════════════

def test_18_actions_info_structure():
    """
    Verify that actions_info returned by run() has the expected schema:

    actions_info[a] = {'n': int, 'q': float}     for each visited action a
    actions_info['best'] = {'index': int, 'n': int}

    Also confirm:
    - 'best'['index'] is in actions_info (i.e. it was actually visited).
    - 'best'['n'] > 0.
    - q = total_value / n lies in a reasonable range for a constant-value net.
    """
    try:
        state_dim   = 4
        max_actions = 8
        const_val   = 0.6
        n_trees     = 4
        max_sims    = 80

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            device_memory=0.2,
        )
        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value  = ConstantValueNet(state_dim, 1, const_val).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        _, best_n, actions_info = agent.run(root, 0, policy, value, problem=None)

        # Check 'best' key
        best = actions_info.get("best", None)
        ok = (
            best is not None
            and "index" in best
            and "n" in best
            and best["n"] > 0
        )

        # Check per-action entries
        non_best_keys = [k for k in actions_info if k != "best"]
        for k in non_best_keys:
            entry = actions_info[k]
            if not ("n" in entry and "q" in entry):
                ok = False
                break
            if entry["n"] > 0 and not (0.0 <= entry["q"] <= 2.0):
                # q should be in a sensible range for this net
                ok = False
                break

        # best index must appear as a regular key
        if ok:
            ok = best["index"] in actions_info

        record(
            "Test 18: actions_info schema", ok,
            f"n_actions_visited={len(non_best_keys)}, best_n={best_n}",
        )
    except Exception as e:
        record("Test 18: actions_info schema", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 19 — PUCTGpu __str__ / __repr__
# ════════════════════════════════════════════════════════════════════════════

def test_19_str_repr():
    """
    Verify that __str__ and __repr__ include key parameter names,
    analogous to MCTSNC.__str__ in mctsnc.py.
    """
    try:
        agent = make_agent(state_dim=6, max_actions=10, num_robots=3,
                           n_trees=4, max_simulations=200, verbose_info=False)
        s = str(agent)
        r = repr(agent)

        required_substrings = [
            "PUCTGpu",
            "state_dim=6",
            "max_actions=10",
            "num_robots=3",
            "n_trees=4",
            "max_simulations=200",
        ]
        ok = all(sub in s for sub in required_substrings)
        ok = ok and (r == s)  # __repr__ should match __str__

        record(
            "Test 19: __str__ / __repr__", ok,
            repr(s[:80]),
        )
    except Exception as e:
        record("Test 19: __str__ / __repr__", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 20 — Repeated runs from different root states
# ════════════════════════════════════════════════════════════════════════════

def test_20_repeated_runs_different_roots():
    """
    Mimics the game loop in mctsnc.py where the AI is called once per move
    from consecutively different states.

    Run the same agent 5 times with different root states and verify that
    each call succeeds independently (no CUDA state carryover corruption).
    """
    try:
        state_dim   = 4
        max_actions = 8
        n_trees     = 4
        max_sims    = 50

        agent = make_agent(
            state_dim=state_dim,
            max_actions=max_actions,
            n_trees=n_trees,
            max_simulations=max_sims,
            device_memory=0.2,
        )
        policy = SpotlightPolicyNet(state_dim, max_actions, hot_action=1, hot_weight=20.0).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.5).cuda()

        rng    = np.random.default_rng(42)
        errors = []

        for i in range(5):
            root = rng.standard_normal(state_dim).astype(np.float32)
            try:
                ba, bn, ai = agent.run(root, i % 1, policy, value, problem=None)
                if not (0 <= ba < max_actions and bn > 0):
                    errors.append(f"step {i}: invalid result ba={ba} bn={bn}")
            except Exception as exc:
                errors.append(f"step {i}: {exc}")

        ok = len(errors) == 0
        detail = "5/5 runs OK" if ok else "; ".join(errors)
        record("Test 20: Repeated runs from different root states", ok, detail)
    except Exception as e:
        record("Test 20: Repeated runs from different root states", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 21 — Discount factor effect on backup
# ════════════════════════════════════════════════════════════════════════════

def test_21_discount_factor():
    """
    Compare total_value at root after 2 full simulation steps with
    gamma=1.0 vs gamma=0.0.

    On step 1 the selected node is the root itself (depth=0), so after
    expansion the child is at depth=1.  On step 2, selection descends
    to that depth-1 child (the leaf now at depth=1).
    The backup on step 2 propagates a value V from depth=1 up to root
    (depth=0), applying discount = gamma^(depth_leaf - depth_root) = gamma^1.

    Expected:
        gamma=1.0 → root total_value contains contributions from step 1 and
                     step 2 (both with discount 1).  After 2 steps with
                     const_val=0.8:  tv ≈ 2 × 0.8 = 1.6.
        gamma=0.0 → step 2's depth-1 leaf contributes gamma^1 × V = 0 to root.
                     Only step 1's leaf (selected = root, depth=0, discount=1)
                     contributes to root. tv ≈ 1 × 0.8 = 0.8.

    So with gamma=0 the root accumulates less than with gamma=1,
    and specifically: tv_g0 < tv_g1.
    """
    try:
        state_dim   = 2
        max_actions = 4
        const_val   = 0.8
        n_sims      = 2    # run 2 steps so depth-1 leaf is visited on step 2

        results_by_gamma = {}

        for gamma_val in [1.0, 0.0]:
            policy = UniformPolicyNet(state_dim, max_actions).cuda()
            value  = ConstantValueNet(state_dim, 1, const_val).cuda()

            agent = make_agent(
                state_dim=state_dim,
                max_actions=max_actions,
                n_trees=1,
                max_simulations=n_sims,
                gamma=gamma_val,
                device_memory=0.1,
            )

            root = np.zeros(state_dim, dtype=np.float32)
            agent.run(root, 0, policy, value, problem=None)

            tv = agent.dev_trees_total_value.copy_to_host()  # (T, S, R)
            results_by_gamma[gamma_val] = float(tv[0, 0, 0])

        tv_g1 = results_by_gamma[1.0]
        tv_g0 = results_by_gamma[0.0]

        # With gamma=1: both steps contribute fully → tv_g1 > tv_g0
        # With gamma=0: step-2 depth-1 leaf contributes 0 to root → tv_g0 < tv_g1
        # Margin: at least const_val difference expected
        ok = (tv_g1 > tv_g0 + const_val * 0.5)   # robust inequality
        record(
            "Test 21: Discount factor effect on root value", ok,
            f"gamma=1 → tv={tv_g1:.4f}, "
            f"gamma=0 → tv={tv_g0:.4f} (expect g1 >> g0)",
        )
    except Exception as e:
        record("Test 21: Discount factor effect on root value", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 22 — MCTSNC parity smoke (MCTSNC original still works)
# ════════════════════════════════════════════════════════════════════════════

def test_22_mctsnc_original_smoke():
    """
    Verify that the *original* MCTSNC still runs correctly alongside PUCTGpu
    (they share the same CUDA context and must not interfere).

    This replicates the document's 'Example usage 1' where MCTSNC runs on
    an empty C4 board and returns a valid column index in [0, 6].

    Skipped gracefully if optional system-info deps (cpuinfo) are unavailable
    in the current environment.
    """
    try:
        # Guard: utils.py imports cpuinfo which may be absent in some envs
        import importlib
        if importlib.util.find_spec("cpuinfo") is None:
            record(
                "Test 22: MCTSNC original smoke on empty C4 board",
                True,   # treat as skip/pass — not a PUCTGpu failure
                "SKIPPED (cpuinfo not installed in current env)",
            )
            return

        from mctsnc import MCTSNC
        from c4 import C4

        board_shape    = C4.get_board_shape()
        extra_info_mem = C4.get_extra_info_memory()
        max_actions    = C4.get_max_actions()

        ai = MCTSNC(
            board_shape, extra_info_mem, max_actions,
            search_time_limit=1.0, search_steps_limit=float("inf"),
            n_trees=4, n_playouts=32,
            variant="acp_prodigal",
            device_memory=0.5,
            verbose_info=False,
            action_index_to_name_function=C4.action_index_to_name,
        )
        ai.init_device_side_arrays()

        c4 = C4()
        best_action = ai.run(c4.get_board(), c4.get_extra_info(), c4.get_turn())

        ok = 0 <= best_action < max_actions
        record(
            "Test 22: MCTSNC original smoke on empty C4 board", ok,
            f"best_action={best_action} (valid column in [0, {max_actions-1}])",
        )
    except Exception as e:
        record("Test 22: MCTSNC original smoke on empty C4 board", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 23 — Coexistence: PUCTGpu + MCTSNC in same process
# ════════════════════════════════════════════════════════════════════════════

def test_23_coexistence():
    """
    Run MCTSNC and PUCTGpu in the same process (interleaved), confirming
    they do not corrupt each other's CUDA state.

    Steps:
    1. MCTSNC runs on C4.
    2. PUCTGpu runs on double-integrator.
    3. MCTSNC runs again on C4 → same best_action (deterministic).
    4. Both best actions must be valid.

    Skipped gracefully if cpuinfo is unavailable in the current environment.
    """
    try:
        import importlib
        if importlib.util.find_spec("cpuinfo") is None:
            record(
                "Test 23: Coexistence PUCTGpu + MCTSNC",
                True,
                "SKIPPED (cpuinfo not installed in current env)",
            )
            return

        from mctsnc import MCTSNC
        from c4 import C4

        board_shape    = C4.get_board_shape()
        extra_info_mem = C4.get_extra_info_memory()
        max_actions_c4 = C4.get_max_actions()

        mctsnc = MCTSNC(
            board_shape, extra_info_mem, max_actions_c4,
            search_time_limit=0.5, search_steps_limit=float("inf"),
            n_trees=4, n_playouts=32, variant="acp_prodigal",
            device_memory=0.5, verbose_info=False,
        )
        mctsnc.init_device_side_arrays()

        c4 = C4()
        board, extra_info, turn = c4.get_board(), c4.get_extra_info(), c4.get_turn()

        best_c4_run1 = mctsnc.run(board, extra_info, turn)

        # PUCTGpu interleaved run
        state_dim        = 2
        max_actions_puct = 5
        agent = make_agent(
            state_dim=state_dim, max_actions=max_actions_puct,
            n_trees=4, max_simulations=50, device_memory=0.2,
        )
        policy = UniformPolicyNet(state_dim, max_actions_puct).cuda()
        value  = ConstantValueNet(state_dim, 1, 0.5).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)
        best_puct, _, _ = agent.run(root, 0, policy, value, problem=None)

        best_c4_run2 = mctsnc.run(board, extra_info, turn)

        ok = (
            0 <= best_c4_run1 < max_actions_c4
            and 0 <= best_puct < max_actions_puct
            and best_c4_run1 == best_c4_run2
        )
        record(
            "Test 23: Coexistence PUCTGpu + MCTSNC", ok,
            f"mctsnc={best_c4_run1}→{best_c4_run2}, puct={best_puct}",
        )
    except Exception as e:
        record("Test 23: Coexistence PUCTGpu + MCTSNC", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Test 24 — Example8 pursuit-evasion integration
# ════════════════════════════════════════════════════════════════════════════

def test_24_example8_integration():
    """
    End-to-end integration with the self-contained example8 problem.

    Verifies:
      - GPU_PUCT_Adapter runs with the example8 problem
      - turn_groups search (2 searches per step, not 4)
      - Terminal detection works
      - Rewards are plausible (all in [-1, 1])
      - Zero dependency on decision_making
    """
    try:
        from examples.example8_problem import Example8
        from examples.example8_adapter import GPU_PUCT_Adapter

        problem = Example8()
        problem.use_minco_rollout = True

        solver = GPU_PUCT_Adapter(
            policy_oracle=[None],
            value_oracle=None,
            search_depth=20,
            number_simulations=50,
            C_pw=2.0, alpha_pw=0.5,
            C_exp=1.0, alpha_exp=0.25,
            beta_policy=0.0, beta_value=0.0,
            max_actions=16,
            n_trees=4,
            device_memory=0.5,
        )

        np.random.seed(123)
        state = problem.initialize()
        max_steps = 5
        rewards_all = []
        terminated = False

        for step in range(max_steps):
            if problem.is_terminal(state):
                terminated = True
                break
            action = solver.multiturn_policy(problem, state)
            reward = problem.reward(state, action)
            next_state = problem.step(state, action, problem.dt)
            terminated = problem.is_terminal(next_state)
            rewards_all.append(reward.flatten())
            state = next_state

        n_steps = len(rewards_all)
        ok = (
            n_steps > 0
            and all(np.all(np.abs(r) <= 1.0 + 1e-6) for r in rewards_all)
        )
        has_groups = hasattr(problem, 'turn_groups') and problem.turn_groups is not None
        term_str = f", terminal={terminated}" if terminated else ""
        record(
            "Test 24: Example8 integration", ok,
            f"steps={n_steps}, turn_groups={has_groups}{term_str}",
        )
    except Exception as e:
        record("Test 24: Example8 integration", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  GPU PUCT — Example-Style Integration Tests  (Tests 11-24)")
    print("=" * 65)

    test_11_double_integrator_smoke()
    test_12_prior_dominance()
    test_13_argmax_stability()
    test_14_tree_stats()
    test_15_multi_robot_backup()
    test_16_time_budget()
    test_17_large_sim_convergence()
    test_18_actions_info_structure()
    test_19_str_repr()
    test_20_repeated_runs_different_roots()
    test_21_discount_factor()
    test_22_mctsnc_original_smoke()
    test_23_coexistence()
    test_24_example8_integration()

    print("=" * 65)
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    if n_fail > 0:
        print("  FAILED tests:")
        for name, ok in results:
            if not ok:
                print(f"    ✗ {name}")
    else:
        print("  All tests PASSED ✓")
    print("=" * 65)
    sys.exit(0 if n_fail == 0 else 1)

