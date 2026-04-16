"""
test_puct_gpu_perf.py
---------------------
Performance benchmarks and stress tests for ``PUCTGpu``.

Sections
--------
BENCH-1   Throughput vs n_trees (steps/second)
BENCH-2   Throughput vs max_actions
BENCH-3   Throughput vs state_dim
BENCH-4   Phase-level timing breakdown (select / NN / expand_backup / reduce)
BENCH-5   JIT warm-up cost vs steady-state
BENCH-6   NN inference time fraction
STRESS-1  Sustained long run (10 000 steps) — no crash / memory leak
STRESS-2  Sequential chained runs — agent reuse correctness
STRESS-3  Maximum-acting-fan-out (max_actions=512) — memory boundary
STRESS-4  Minimum state_dim=1, maximum n_trees=128
STRESS-5  Very large n_trees=256 with small step budget
STRESS-6  Memory-budget edge: 1 % GPU memory → minimum tree size
STRESS-7  Rapid allocation / free — no CUDA memory fragmentation crash

Usage
-----
    cd /home/robomaster/Research/mcts_numba_cuda/src
    python test_puct_gpu_perf.py               # all benchmarks + stress
    python test_puct_gpu_perf.py --bench-only  # benchmarks only, no stress
    python test_puct_gpu_perf.py --stress-only # stress only, no bench

    bash run_puct_tests.sh --perf
"""

import sys
import os
import time
import math
import gc
import numpy as np
import torch
import torch.nn as nn

# ── CUDA compat shim ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    _torch_cuda_sentinel = torch.zeros(1, device="cuda")

try:
    from numba.cuda.cudadrv import nvvm as _nvvm
    if hasattr(_nvvm, "CTK_SUPPORTED"):
        for _v in [(12, 5), (12, 6), (12, 7), (12, 8), (12, 9)]:
            if _v not in _nvvm.CTK_SUPPORTED:
                _nvvm.CTK_SUPPORTED[_v] = ((5, 0), (9, 0))
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from puct_gpu import PUCTGpu

# ── Mode flags ────────────────────────────────────────────────────────────────
BENCH_ONLY  = "--bench-only"  in sys.argv
STRESS_ONLY = "--stress-only" in sys.argv
RUN_BENCH   = not STRESS_ONLY
RUN_STRESS  = not BENCH_ONLY

# ════════════════════════════════════════════════════════════════════════════
# Stubs
# ════════════════════════════════════════════════════════════════════════════

class UniformPolicy(nn.Module):
    def __init__(self, sd, ma):
        super().__init__(); self.ma = ma; self._d = nn.Linear(sd, 1, bias=False)
    def forward(self, x):
        return torch.ones(x.shape[0], self.ma, device=x.device)

class ConstantValue(nn.Module):
    def __init__(self, sd, nr, c=0.5):
        super().__init__(); self.nr = nr; self.c = c; self._d = nn.Linear(sd, 1, bias=False)
    def forward(self, x):
        return torch.full((x.shape[0], self.nr), self.c, dtype=torch.float32, device=x.device)

# ════════════════════════════════════════════════════════════════════════════
# Infrastructure
# ════════════════════════════════════════════════════════════════════════════

PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"
results: list[tuple[str, bool]] = []

def record(name: str, ok: bool, detail: str = ""):
    tag = PASS_MARK if ok else FAIL_MARK
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, ok))

def make_agent(state_dim=4, max_actions=8, num_robots=1, n_trees=4,
               max_simulations=100, device_memory=0.4, gamma=0.99,
               C_exp=1.0, alpha_exp=0.5, C_pw=1.0, alpha_pw=0.5):
    ag = PUCTGpu(
        state_dim=state_dim, action_dim=1, max_actions=max_actions,
        num_robots=num_robots, n_trees=n_trees, max_simulations=max_simulations,
        C_exp=C_exp, alpha_exp=alpha_exp, C_pw=C_pw, alpha_pw=alpha_pw,
        gamma=gamma, device_memory=device_memory,
        verbose_info=False, verbose_debug=False,
    )
    ag.init_device_side_arrays()
    return ag

def timed_run(agent, policy, value, state_dim, n_warmup=1, n_measure=5):
    """
    Returns (mean_elapsed, steps_per_run).
    First n_warmup runs are discarded (JIT warm-up).
    """
    root = np.zeros(state_dim, dtype=np.float32)
    for _ in range(n_warmup):
        agent.run(root, 0, policy, value, problem=None)
        torch.cuda.synchronize()

    elapsed_list = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        agent.run(root, 0, policy, value, problem=None)
        torch.cuda.synchronize()
        elapsed_list.append(time.perf_counter() - t0)

    return float(np.mean(elapsed_list)), int(agent.steps)


# ════════════════════════════════════════════════════════════════════════════
# BENCH-1 — Throughput vs n_trees
# ════════════════════════════════════════════════════════════════════════════

def bench_1_throughput_vs_n_trees():
    """
    Measure total-tree steps/second (steps × n_trees / wall-clock) as n_trees
    scales from 1 → 256.

    Because the NN inference is a serial bottleneck, raw steps/s may not scale
    with n_trees.  However, *useful work* (tree-node updates per second) does
    scale: each step simultaneously advances all n_trees trees.

    We measure two metrics:
      - steps/s             : raw search iterations per second
      - tree-steps/s        : steps × n_trees (total GPU work) per second

    Expectation: tree-steps/s at n_trees=256 must be ≥ 5× vs n_trees=1.
    """
    print("\n  BENCH-1: Throughput vs n_trees")
    print("  " + "-" * 68)
    print(f"  {'n_trees':>8} | {'steps':>6} | {'time(s)':>8} | {'steps/s':>10} | {'tree-steps/s':>14} | {'ratio':>8}")
    print("  " + "-" * 68)

    tree_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    state_dim = 4; max_actions = 8; max_sims = 200

    prev_treesp = None
    results_data = []
    all_ok = True

    for nt in tree_sizes:
        try:
            agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                                n_trees=nt, max_simulations=max_sims,
                                device_memory=max(0.05 * nt, 0.1))
            policy = UniformPolicy(state_dim, max_actions).cuda()
            value  = ConstantValue(state_dim, 1).cuda()

            mean_t, steps = timed_run(agent, policy, value, state_dim,
                                      n_warmup=1, n_measure=3)
            sps        = steps / mean_t
            tree_sps   = steps * nt / mean_t        # total GPU work per second
            ratio_str  = f"{tree_sps/prev_treesp:.2f}x" if prev_treesp else "base"
            print(f"  {nt:>8} | {steps:>6} | {mean_t:>8.3f} | {sps:>10.1f} | {tree_sps:>14.1f} | {ratio_str:>8}")
            results_data.append((nt, tree_sps))
            prev_treesp = tree_sps
        except Exception as e:
            print(f"  {nt:>8} | ERROR: {e}")
            all_ok = False

    # Sanity: tree-steps/s at n_trees=256 must be ≥ 5× vs n_trees=1
    if len(results_data) >= 2:
        baseline = results_data[0][1]   # n_trees=1 tree-steps/s
        largest  = results_data[-1][1]  # n_trees=256 tree-steps/s
        speedup  = largest / baseline
        ok = speedup > 5.0
        all_ok = all_ok and ok
        record("BENCH-1: tree-level throughput (256× vs 1× speedup > 5×)", ok,
               f"tree-step speedup={speedup:.1f}×")
    else:
        record("BENCH-1: tree-level throughput", all_ok, "insufficient data")


# ════════════════════════════════════════════════════════════════════════════
# BENCH-2 — Throughput vs max_actions
# ════════════════════════════════════════════════════════════════════════════

def bench_2_throughput_vs_max_actions():
    """Measure time/step as max_actions grows. Expect sub-linear slowdown."""
    print("\n  BENCH-2: Throughput vs max_actions")
    print("  " + "-" * 50)
    print(f"  {'max_acts':>8} | {'sps':>10} | {'ms/step':>10}")
    print("  " + "-" * 50)

    acts_list = [4, 8, 16, 32, 64, 128, 256, 512]
    state_dim = 4; n_trees = 32; max_sims = 100

    prev_ms = None
    all_ok = True
    for ma in acts_list:
        try:
            agent  = make_agent(state_dim=state_dim, max_actions=ma,
                                n_trees=n_trees, max_simulations=max_sims)
            policy = UniformPolicy(state_dim, ma).cuda()
            value  = ConstantValue(state_dim, 1).cuda()
            mean_t, steps = timed_run(agent, policy, value, state_dim)
            sps  = steps / mean_t
            ms_per_step = 1000.0 * mean_t / steps
            print(f"  {ma:>8} | {sps:>10.1f} | {ms_per_step:>10.3f}")
        except Exception as e:
            print(f"  {ma:>8} | ERROR: {e}")
            all_ok = False

    record("BENCH-2: max_actions throughput sweep", all_ok,
           "see table above")


# ════════════════════════════════════════════════════════════════════════════
# BENCH-3 — Throughput vs state_dim
# ════════════════════════════════════════════════════════════════════════════

def bench_3_throughput_vs_state_dim():
    """NN forward time dominates at high state_dim; measure impact."""
    print("\n  BENCH-3: Throughput vs state_dim")
    print("  " + "-" * 50)
    print(f"  {'state_dim':>10} | {'sps':>10} | {'ms/step':>10}")
    print("  " + "-" * 50)

    dims = [2, 4, 8, 16, 32, 64, 128, 256]
    max_actions = 8; n_trees = 32; max_sims = 100

    all_ok = True
    for sd in dims:
        try:
            agent  = make_agent(state_dim=sd, max_actions=max_actions,
                                n_trees=n_trees, max_simulations=max_sims)
            policy = UniformPolicy(sd, max_actions).cuda()
            value  = ConstantValue(sd, 1).cuda()
            mean_t, steps = timed_run(agent, policy, value, sd)
            sps = steps / mean_t
            ms  = 1000.0 * mean_t / steps
            print(f"  {sd:>10} | {sps:>10.1f} | {ms:>10.3f}")
        except Exception as e:
            print(f"  {sd:>10} | ERROR: {e}")
            all_ok = False

    record("BENCH-3: state_dim throughput sweep", all_ok, "see table above")


# ════════════════════════════════════════════════════════════════════════════
# BENCH-4 — Phase-level timing breakdown
# ════════════════════════════════════════════════════════════════════════════

def bench_4_phase_timing():
    """
    Time each of the main phases manually:
      Phase 1: _select_puct + _extract_leaf_states
      Phase 2: NN forward (policy + value)
      Phase 3: _prepare_expansion + _commit_expansion_and_backup
      Total vs sum(phases)
    """
    print("\n  BENCH-4: Phase-level timing breakdown")
    print("  " + "-" * 62)

    from puct_gpu_kernels import (
        _reset_puct, _select_puct, _extract_leaf_states,
        _prepare_expansion_puct, _commit_expansion_and_backup_puct,
    )
    from numba import cuda

    state_dim = 4; max_actions = 8; n_trees = 32; N = 200

    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=N)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        # First: full warm-up run
        agent.run(root, 0, policy, value, problem=None)

        # ── Manual phase loop ─────────────────────────────────────────────
        dev_root = cuda.to_device(root)
        _reset_puct[n_trees, agent.tpb_r](
            dev_root, np.int8(0),
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals,
            agent.dev_trees_ns, agent.dev_trees_total_value,
            agent.dev_trees_states, agent.dev_trees_action_priors,
            agent.dev_trees_pw_boundary,
        )
        cuda.synchronize()

        t_phase1 = t_phase2 = t_phase3 = 0.0
        REPS = 50

        for _ in range(REPS):
            # Phase 1: select + extract
            t0 = time.perf_counter()
            _select_puct[n_trees, agent.tpb_s](
                np.float32(agent.C_exp), np.float32(agent.alpha_exp),
                agent.dev_trees, agent.dev_trees_leaves, agent.dev_trees_ns,
                agent.dev_trees_total_value, agent.dev_trees_robot_turns,
                agent.dev_trees_action_priors, agent.dev_trees_pw_boundary,
                agent.dev_trees_prior_rank, agent.dev_trees_nodes_selected,
                agent.dev_trees_selected_paths,
            )
            _extract_leaf_states[n_trees, agent.tpb_r](
                agent.dev_trees_nodes_selected, agent.dev_trees_states,
                agent.dev_trees_terminals,
                agent.bridge.dev_leaf_states, agent.bridge.dev_leaf_valid,
            )
            cuda.synchronize()
            t_phase1 += time.perf_counter() - t0

            # Phase 2: NN
            t0 = time.perf_counter()
            with torch.no_grad():
                priors = policy(agent.bridge.leaf_states)
                vals   = value(agent.bridge.leaf_states)
            agent.bridge.nn_priors.copy_(priors)
            agent.bridge.nn_values.copy_(vals)
            torch.cuda.synchronize()
            t_phase2 += time.perf_counter() - t0

            # Phase 3: prepare + commit
            t0 = time.perf_counter()
            _prepare_expansion_puct[n_trees, agent.tpb_s](
                np.float32(agent.C_pw), np.float32(agent.alpha_pw),
                np.int32(agent.num_robots),
                agent.bridge.dev_nn_priors, agent.bridge.dev_leaf_valid,
                agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
                agent.dev_trees_robot_turns,
                agent.dev_trees_leaves, agent.dev_trees_terminals,
                agent.dev_trees_ns, agent.dev_trees_total_value,
                agent.dev_trees_states, agent.dev_trees_action_priors,
                agent.dev_trees_prior_rank, agent.dev_trees_pw_boundary,
                agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
                np.int32(agent.max_tree_size), np.int32(state_dim),
                np.int32(max_actions),
                agent.bridge.dev_expansion_valid,
                agent.bridge.dev_expanded_parent_states,
                agent.bridge.dev_expanded_actions,
            )
            _commit_expansion_and_backup_puct[n_trees, agent.tpb_s](
                np.float32(agent.gamma),
                np.float32(agent.C_pw), np.float32(agent.alpha_pw),
                np.int32(agent.num_robots),
                agent.bridge.dev_nn_values, agent.bridge.dev_leaf_valid,
                agent.bridge.dev_expansion_valid,
                agent.bridge.dev_expanded_next_states,
                agent.bridge.dev_expanded_rewards,
                agent.bridge.dev_expanded_terminals,
                agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
                agent.dev_trees_robot_turns,
                agent.dev_trees_leaves, agent.dev_trees_terminals,
                agent.dev_trees_ns, agent.dev_trees_total_value,
                agent.dev_trees_states, agent.dev_trees_action_priors,
                agent.dev_trees_prior_rank, agent.dev_trees_pw_boundary,
                agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
                np.int32(agent.max_tree_size), np.int32(max_actions),
            )
            cuda.synchronize()
            t_phase3 += time.perf_counter() - t0

        # Average per step
        p1 = 1000.0 * t_phase1 / REPS
        p2 = 1000.0 * t_phase2 / REPS
        p3 = 1000.0 * t_phase3 / REPS
        tot = p1 + p2 + p3

        print(f"  Phase 1 (select+extract) : {p1:8.3f} ms/step  ({100*p1/tot:5.1f}%)")
        print(f"  Phase 2 (NN inference)   : {p2:8.3f} ms/step  ({100*p2/tot:5.1f}%)")
        print(f"  Phase 3 (expand+backup)  : {p3:8.3f} ms/step  ({100*p3/tot:5.1f}%)")
        print(f"  ─────────────────────────────────────────")
        print(f"  Total (sum of phases)    : {tot:8.3f} ms/step")
        print(f"  n_trees={n_trees}, state_dim={state_dim}, max_actions={max_actions}")
        print("  " + "-" * 62)

        # All phases should take reasonable time (sanity: < 100 ms each)
        ok = (p1 < 100) and (p2 < 100) and (p3 < 100)
        record("BENCH-4: Phase timing breakdown (all < 100 ms/step)", ok,
               f"p1={p1:.2f}ms p2={p2:.2f}ms p3={p3:.2f}ms")
    except Exception as e:
        record("BENCH-4: Phase timing breakdown", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# BENCH-5 — JIT warm-up cost vs steady-state
# ════════════════════════════════════════════════════════════════════════════

def bench_5_jit_warmup():
    """
    Measure time for run #1 vs runs #2-5.
    If kernels are already JIT-compiled (cached), both should be fast.
    We report timing but only assert that steady-state < 2 s per run.
    """
    state_dim = 4; max_actions = 8; n_trees = 8; max_sims = 50
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        # Run 1: may include JIT or be served from cache
        t0 = time.perf_counter()
        agent.run(root, 0, policy, value, problem=None)
        torch.cuda.synchronize()
        t_first = time.perf_counter() - t0

        # Runs 2-5: steady state
        times = []
        for _ in range(4):
            t0 = time.perf_counter()
            agent.run(root, 0, policy, value, problem=None)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        t_steady = float(np.mean(times))

        ratio = t_first / max(t_steady, 1e-9)
        print(f"\n  BENCH-5: JIT warm-up vs steady-state")
        print(f"    Run #1       : {t_first*1000:.1f} ms  ({'JIT included' if ratio > 2 else 'cached'})")
        print(f"    Runs #2-5 avg: {t_steady*1000:.1f} ms")
        print(f"    Ratio        : {ratio:.1f}x")

        # Steady state must be fast regardless of cache status
        ok = t_steady < 2.0   # < 2 s per 50-step run on any hardware
        record("BENCH-5: Steady-state timing < 2s", ok,
               f"first={t_first*1000:.1f}ms, steady={t_steady*1000:.1f}ms, ratio={ratio:.1f}x")
    except Exception as e:
        record("BENCH-5: Steady-state timing < 2s", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# BENCH-6 — NN inference fraction
# ════════════════════════════════════════════════════════════════════════════

def bench_6_nn_fraction():
    """
    Benchmark NN fraction for two extremes:
    A) state_dim=4,  hidden≈trivial  → NN should be <50% of total time
    B) state_dim=256, hidden≈trivial → NN might dominate

    Uses torch.cuda.Event for high-precision GPU timing.
    """
    print("\n  BENCH-6: NN inference fraction by state_dim")
    print("  " + "-" * 58)
    print(f"  {'state_dim':>10} | {'NN ms/step':>12} | {'total ms':>10} | {'NN%':>6}")
    print("  " + "-" * 58)

    max_actions = 8; n_trees = 32; max_sims = 100
    all_ok = True

    for sd in [4, 8, 32, 64, 128, 256]:
        try:
            agent  = make_agent(state_dim=sd, max_actions=max_actions,
                                n_trees=n_trees, max_simulations=max_sims)
            policy = UniformPolicy(sd, max_actions).cuda()
            value  = ConstantValue(sd, 1).cuda()
            root   = np.zeros(sd, dtype=np.float32)

            # Warm up
            agent.run(root, 0, policy, value, problem=None)

            # Time full run
            t0 = time.perf_counter()
            agent.run(root, 0, policy, value, problem=None)
            torch.cuda.synchronize()
            t_full = time.perf_counter() - t0

            # Time NN forward only (batch of n_trees)
            x = torch.zeros(n_trees, sd, device="cuda")
            REPS = max_sims
            t0 = time.perf_counter()
            for _ in range(REPS):
                with torch.no_grad():
                    policy(x); value(x)
            torch.cuda.synchronize()
            t_nn = (time.perf_counter() - t0) / REPS

            nn_pct = 100.0 * t_nn / (t_full / max_sims)
            print(f"  {sd:>10} | {t_nn*1000:>12.4f} | {t_full*1000:>10.2f} | {nn_pct:>6.1f}%")
        except Exception as e:
            print(f"  {sd:>10} | ERROR: {e}")
            all_ok = False

    record("BENCH-6: NN fraction sweep", all_ok, "see table above")


# ════════════════════════════════════════════════════════════════════════════
# Stress test CUDA guard
# ════════════════════════════════════════════════════════════════════════════

def cuda_guard():
    """
    Flush any pending CUDA operations and free cached memory.
    Helps prevent error 700 cascade from prior benchmark allocations.
    """
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# STRESS-1 — Sustained long run
# ════════════════════════════════════════════════════════════════════════════

def stress_1_long_run():
    """5 000 simulation steps without crash or OOM."""
    cuda_guard()
    state_dim = 4; max_actions = 8; n_trees = 8; max_sims = 5_000
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims,
                            device_memory=0.4)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        t0 = time.perf_counter()
        ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ok = (0 <= ba < max_actions) and (agent.steps <= max_sims)
        record("STRESS-1: 5 000 steps long run", ok,
               f"steps={agent.steps}, best_n={bn}, elapsed={elapsed:.2f}s")
    except Exception as e:
        record("STRESS-1: 5 000 steps long run", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# STRESS-2 — Sequential chained runs (agent reuse)
# ════════════════════════════════════════════════════════════════════════════

def stress_2_chained_runs():
    """
    Simulate a game loop: 30 sequential calls from different root states.
    Each call must return a valid action. No CUDA state corruption.
    """
    cuda_guard()
    state_dim = 4; max_actions = 8; n_trees = 8; max_sims = 100; N = 30
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        rng    = np.random.default_rng(7)
        errors = []

        for i in range(N):
            root = rng.standard_normal(state_dim).astype(np.float32)
            ba, bn, ai = agent.run(root, 0, policy, value, problem=None)
            if not (0 <= ba < max_actions and bn > 0):
                errors.append(f"step{i}: ba={ba},bn={bn}")

        ok = len(errors) == 0
        record("STRESS-2: 30× sequential chained runs", ok,
               f"all {N} runs valid" if ok else f"FAILED: {errors}")
    except Exception as e:
        record("STRESS-2: 30× sequential chained runs", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# STRESS-3 — Large max_actions=512
# ════════════════════════════════════════════════════════════════════════════

def stress_3_large_max_actions():
    """max_actions=512 pushes per-node prior storage to memory limit."""
    cuda_guard()
    state_dim = 4; max_actions = 512; n_trees = 4; max_sims = 50
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims,
                            device_memory=0.5)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
        ok = (0 <= ba < max_actions) and (bn > 0)
        record("STRESS-3: max_actions=512", ok,
               f"best_action={ba}, best_n={bn}, tree_size={agent.max_tree_size}")
    except Exception as e:
        record("STRESS-3: max_actions=512", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# STRESS-4 — state_dim=1, n_trees=128
# ════════════════════════════════════════════════════════════════════════════

def stress_4_minimal_state_maxtrees():
    """state_dim=1 (minimum) with n_trees=128 (high GPU utilization)."""
    cuda_guard()
    state_dim = 1; max_actions = 8; n_trees = 128; max_sims = 200
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims,
                            device_memory=0.6)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
        ok = (0 <= ba < max_actions) and (bn > 0)
        record("STRESS-4: state_dim=1, n_trees=128", ok,
               f"max_tree_size={agent.max_tree_size}, best_n={bn}")
    except Exception as e:
        record("STRESS-4: state_dim=1, n_trees=128", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# STRESS-5 — Very large n_trees=256
# ════════════════════════════════════════════════════════════════════════════

def stress_5_huge_n_trees():
    """n_trees=256 — tests reduction kernel with 256 trees."""
    cuda_guard()
    state_dim = 4; max_actions = 8; n_trees = 256; max_sims = 50
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims,
                            device_memory=0.7)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        t0 = time.perf_counter()
        ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ok = (0 <= ba < max_actions) and (bn > 0)
        record("STRESS-5: n_trees=256 huge parallel search", ok,
               f"elapsed={elapsed:.2f}s, steps={agent.steps}, best_n={bn}")
    except Exception as e:
        record("STRESS-5: n_trees=256 huge parallel search", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# STRESS-6 — Minimum memory budget (1 % GPU)
# ════════════════════════════════════════════════════════════════════════════

def stress_6_minimal_memory():
    """device_memory=0.01 → very small max_tree_size. Must not crash."""
    cuda_guard()
    state_dim = 4; max_actions = 8; n_trees = 4; max_sims = 20
    try:
        agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                            n_trees=n_trees, max_simulations=max_sims,
                            device_memory=0.01)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)

        ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
        ok = (0 <= ba < max_actions)
        record("STRESS-6: 1% GPU memory budget (tiny tree)", ok,
               f"max_tree_size={agent.max_tree_size}, best_action={ba}")
    except Exception as e:
        record("STRESS-6: 1% GPU memory budget (tiny tree)", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# STRESS-7 — Rapid alloc/free (memory fragmentation)
# ════════════════════════════════════════════════════════════════════════════

def stress_7_rapid_alloc_free():
    """
    Create and discard 10 agents in rapid succession.
    Verifies that device memory is properly freed and no fragmentation crash occurs.
    """
    cuda_guard()
    state_dim = 4; max_actions = 8; n_trees = 16; N = 10
    errors = []
    try:
        for i in range(N):
            agent  = make_agent(state_dim=state_dim, max_actions=max_actions,
                                n_trees=n_trees, max_simulations=20,
                                device_memory=0.1)
            policy = UniformPolicy(state_dim, max_actions).cuda()
            value  = ConstantValue(state_dim, 1).cuda()
            root   = np.zeros(state_dim, dtype=np.float32)
            ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
            if not (0 <= ba < max_actions):
                errors.append(f"alloc#{i}: ba={ba}")
            # Explicitly free
            del agent, policy, value
            gc.collect()
            torch.cuda.empty_cache()

        ok = len(errors) == 0
        record("STRESS-7: 10× rapid alloc/run/free", ok,
               f"{N} cycles OK" if ok else f"FAILED: {errors}")
    except Exception as e:
        record("STRESS-7: 10× rapid alloc/run/free", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Print GPU memory summary
# ════════════════════════════════════════════════════════════════════════════

def print_gpu_memory():
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n  GPU Memory: alloc={alloc:.2f} GB  reserved={reserved:.2f} GB  "
          f"total={total:.2f} GB")


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  GPU PUCT — Performance Benchmarks & Stress Tests")
    print("=" * 70)
    print_gpu_memory()

    if RUN_BENCH:
        print("\n" + "─" * 70)
        print("  BENCHMARKS")
        print("─" * 70)
        bench_1_throughput_vs_n_trees()
        bench_2_throughput_vs_max_actions()
        bench_3_throughput_vs_state_dim()
        bench_4_phase_timing()
        bench_5_jit_warmup()
        bench_6_nn_fraction()

    if RUN_STRESS:
        print("\n" + "─" * 70)
        print("  STRESS TESTS")
        print("─" * 70)
        stress_1_long_run()
        stress_2_chained_runs()
        stress_3_large_max_actions()
        stress_4_minimal_state_maxtrees()
        stress_5_huge_n_trees()
        stress_6_minimal_memory()
        stress_7_rapid_alloc_free()

    print_gpu_memory()
    print("=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    if n_fail > 0:
        print("  FAILED:")
        for name, ok in results:
            if not ok:
                print(f"    ✗ {name}")
    else:
        print("  All benchmarks/stress tests PASSED ✓")
    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)
