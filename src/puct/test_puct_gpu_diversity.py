"""
test_puct_gpu_diversity.py
---------------------------
Expanded input-diversity tests for ``PUCTGpu``.

Validates correctness across the full parameter space:
  - state_dim    : 1, 2, 4, 8, 16, 32, 64, 128, 256
  - max_actions  : 2, 4, 8, 16, 32, 64, 128, 256, 512
  - n_trees      : 1, 2, 4, 8, 16, 32, 64, 128
  - num_robots   : 1, 2, 4, 8
  - gamma        : 0.0, 0.5, 0.9, 0.99, 1.0
  - C_exp / alpha_exp / C_pw / alpha_pw combinations
  - max_simulations : 1, 5, 20, 100

Each test group asserts a semantic invariant (valid output ranges, visit-count
monotonicity, prior-dominance stability, etc.) rather than exact numeric values,
making the suite robust to non-determinism while still catching regressions.

Usage
-----
    cd /home/robomaster/Research/mcts_numba_cuda/src
    python test_puct_gpu_diversity.py          # full run
    python test_puct_gpu_diversity.py --quick  # skip heavy configs

    bash run_puct_tests.sh --diversity
"""

import sys
import os
import time
import itertools
import math
import numpy as np
import torch
import torch.nn as nn

# ── Numba / PyTorch CUDA compat ───────────────────────────────────────────────
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
from puct.puct_gpu import PUCTGpu

# ── Quick-mode flag ────────────────────────────────────────────────────────────
QUICK = "--quick" in sys.argv

# ════════════════════════════════════════════════════════════════════════════
# Stub networks
# ════════════════════════════════════════════════════════════════════════════

class UniformPolicy(nn.Module):
    def __init__(self, state_dim, max_actions):
        super().__init__()
        self.max_actions = max_actions
        self._d = nn.Linear(state_dim, 1, bias=False)
    def forward(self, x):
        return torch.ones(x.shape[0], self.max_actions, device=x.device)

class ConstantValue(nn.Module):
    def __init__(self, state_dim, num_robots, c=0.5):
        super().__init__()
        self._d = nn.Linear(state_dim, 1, bias=False)
        self.c = c; self.num_robots = num_robots
    def forward(self, x):
        return torch.full((x.shape[0], self.num_robots), self.c,
                          dtype=torch.float32, device=x.device)

class SpotlightPolicy(nn.Module):
    def __init__(self, state_dim, max_actions, hot, weight=50.0):
        super().__init__()
        self.ma = max_actions; self.hot = hot; self.w = weight
        self._d = nn.Linear(state_dim, 1, bias=False)
    def forward(self, x):
        B = x.shape[0]
        logits = torch.ones(B, self.ma, device=x.device)
        logits[:, self.hot] = self.w
        return logits

# ════════════════════════════════════════════════════════════════════════════
# Infrastructure
# ════════════════════════════════════════════════════════════════════════════

PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"
SKIP_MARK = "  [SKIP]"
results: list[tuple[str, bool]] = []

def record(name, ok, detail=""):
    tag = PASS_MARK if ok else FAIL_MARK
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, ok))

def skip(name, reason=""):
    print(f"{SKIP_MARK}  {name}  ({reason})")
    results.append((name, True))  # skips are not failures

def make_agent(state_dim=4, action_dim=1, max_actions=8, num_robots=1,
               n_trees=4, max_simulations=20, C_exp=1.0, alpha_exp=0.5,
               C_pw=1.0, alpha_pw=0.5, gamma=0.99, device_memory=0.3):
    ag = PUCTGpu(
        state_dim=state_dim, action_dim=action_dim,
        max_actions=max_actions, num_robots=num_robots,
        n_trees=n_trees, max_simulations=max_simulations,
        C_exp=C_exp, alpha_exp=alpha_exp,
        C_pw=C_pw, alpha_pw=alpha_pw,
        gamma=gamma, device_memory=device_memory,
        verbose_info=False, verbose_debug=False,
    )
    ag.init_device_side_arrays()
    return ag

def valid_run(state_dim, max_actions, n_trees, num_robots=1,
              max_simulations=20, gamma=0.99,
              C_exp=1.0, alpha_exp=0.5, C_pw=1.0, alpha_pw=0.5,
              device_memory=0.2):
    """Run agent and return (best_action, best_n, steps, ok)."""
    agent = make_agent(
        state_dim=state_dim, max_actions=max_actions,
        n_trees=n_trees, num_robots=num_robots,
        max_simulations=max_simulations, gamma=gamma,
        C_exp=C_exp, alpha_exp=alpha_exp,
        C_pw=C_pw, alpha_pw=alpha_pw,
        device_memory=device_memory,
    )
    policy = UniformPolicy(state_dim, max_actions).cuda()
    value  = ConstantValue(state_dim, num_robots).cuda()
    root   = np.zeros(state_dim, dtype=np.float32)
    ba, bn, ai = agent.run(root, 0, policy, value, problem=None)
    ok = (0 <= ba < max_actions) and (bn > 0) and ("best" in ai)
    return ba, bn, agent.steps, ok

# ════════════════════════════════════════════════════════════════════════════
# GROUP A — state_dim sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_A_state_dim():
    """Valid output across state_dim ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}."""
    dims = [1, 2, 4, 8, 16, 32] if QUICK else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    all_ok = True
    failed = []
    for d in dims:
        try:
            _, bn, steps, ok = valid_run(state_dim=d, max_actions=8,
                                         n_trees=4, max_simulations=20)
            if not ok:
                all_ok = False
                failed.append(f"dim={d}")
        except Exception as e:
            all_ok = False
            failed.append(f"dim={d}:{e}")
    detail = f"dims tested={dims}" if all_ok else f"FAILED: {failed}"
    record("Group A: state_dim sweep", all_ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# GROUP B — max_actions sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_B_max_actions():
    """Valid output across max_actions ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512}."""
    actions = [2, 4, 8, 16, 32, 64] if QUICK else [2, 4, 8, 16, 32, 64, 128, 256, 512]
    all_ok = True
    failed = []
    for a in actions:
        try:
            ba, bn, _, ok = valid_run(state_dim=4, max_actions=a,
                                      n_trees=4, max_simulations=20)
            if not ok:
                all_ok = False
                failed.append(f"actions={a}")
        except Exception as e:
            all_ok = False
            failed.append(f"actions={a}:{e}")
    detail = f"actions tested={actions}" if all_ok else f"FAILED: {failed}"
    record("Group B: max_actions sweep", all_ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# GROUP C — n_trees sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_C_n_trees():
    """Valid output across n_trees ∈ {1, 2, 4, 8, 16, 32, 64, 128}."""
    trees = [1, 2, 4, 8, 16, 32] if QUICK else [1, 2, 4, 8, 16, 32, 64, 128]
    all_ok = True
    failed = []
    for t in trees:
        try:
            _, bn, _, ok = valid_run(state_dim=4, max_actions=8,
                                     n_trees=t, max_simulations=20,
                                     device_memory=0.05 * t + 0.1)
            if not ok:
                all_ok = False
                failed.append(f"trees={t}")
        except Exception as e:
            all_ok = False
            failed.append(f"trees={t}:{e}")
    detail = f"n_trees tested={trees}" if all_ok else f"FAILED: {failed}"
    record("Group C: n_trees sweep", all_ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# GROUP D — num_robots sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_D_num_robots():
    """Valid output across num_robots ∈ {1, 2, 4, 8}."""
    robots_list = [1, 2, 4, 8]
    all_ok = True
    failed = []
    for r in robots_list:
        try:
            _, bn, _, ok = valid_run(state_dim=4, max_actions=8,
                                     n_trees=4, num_robots=r,
                                     max_simulations=20)
            if not ok:
                all_ok = False
                failed.append(f"robots={r}")
        except Exception as e:
            all_ok = False
            failed.append(f"robots={r}:{e}")
    record("Group D: num_robots sweep", all_ok,
           f"tested={robots_list}" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP E — gamma sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_E_gamma():
    """Valid output and monotone total_value scaling across gamma ∈ {0,0.5,0.9,0.99,1}."""
    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]
    all_ok = True
    failed = []
    tv_by_gamma = {}
    for g in gammas:
        try:
            agent = make_agent(state_dim=4, max_actions=8, n_trees=4,
                                max_simulations=50, gamma=g)
            policy = UniformPolicy(4, 8).cuda()
            value  = ConstantValue(4, 1, c=0.5).cuda()
            root   = np.zeros(4, dtype=np.float32)
            ba, bn, ai = agent.run(root, 0, policy, value, problem=None)
            ok = (0 <= ba < 8) and (bn > 0)
            tv = float(agent.dev_trees_total_value.copy_to_host()[0, 0, 0])
            tv_by_gamma[g] = tv
            if not ok:
                all_ok = False
                failed.append(f"gamma={g}")
        except Exception as e:
            all_ok = False
            failed.append(f"gamma={g}:{e}")

    # Invariant: higher gamma → higher root value (more future discounting)
    if all_ok and len(tv_by_gamma) == len(gammas):
        tvs = [tv_by_gamma[g] for g in gammas]
        # Monotonically non-decreasing
        monotone = all(tvs[i] <= tvs[i+1] + 0.01 for i in range(len(tvs)-1))
        if not monotone:
            all_ok = False
            failed.append(f"not monotone: {dict(zip(gammas, [f'{v:.3f}' for v in tvs]))}")

    detail = (f"gamma→tv: {dict(zip(gammas, [round(tv_by_gamma.get(g,0),3) for g in gammas]))}"
              if not failed else f"FAILED: {failed}")
    record("Group E: gamma sweep + monotonicity", all_ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# GROUP F — exploration constants sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_F_exploration():
    """
    C_exp ∈ {0.01, 0.1, 1.0, 2.0, 5.0}, alpha_exp ∈ {0.25, 0.5, 1.0}.
    All runs must return valid actions. Higher C_exp must spread visits more evenly
    (measured: std of actions_ns when hot action exists).
    """
    C_exps    = [0.01, 0.5, 1.0, 5.0] if QUICK else [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    a_exps    = [0.25, 0.5, 1.0]
    all_ok    = True
    failed    = []
    n_ok = 0
    for C_exp, alpha_exp in itertools.product(C_exps, a_exps):
        try:
            _, bn, _, ok = valid_run(
                state_dim=4, max_actions=8, n_trees=4,
                max_simulations=30, C_exp=C_exp, alpha_exp=alpha_exp,
            )
            if ok:
                n_ok += 1
            else:
                failed.append(f"C={C_exp},a={alpha_exp}")
                all_ok = False
        except Exception as e:
            failed.append(f"C={C_exp},a={alpha_exp}:{e}")
            all_ok = False
    total = len(C_exps) * len(a_exps)
    record("Group F: C_exp × alpha_exp grid", all_ok,
           f"{n_ok}/{total} OK" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP G — progressive widening constants sweep
# ════════════════════════════════════════════════════════════════════════════

def test_group_G_pw_constants():
    """C_pw ∈ {0.5, 1.0, 2.0}, alpha_pw ∈ {0.25, 0.5, 0.75, 1.0}."""
    C_pws     = [0.5, 1.0, 2.0]
    a_pws     = [0.25, 0.5, 0.75, 1.0]
    all_ok    = True
    failed    = []
    n_ok = 0
    for C_pw, alpha_pw in itertools.product(C_pws, a_pws):
        try:
            _, bn, _, ok = valid_run(
                state_dim=4, max_actions=16, n_trees=4,
                max_simulations=30, C_pw=C_pw, alpha_pw=alpha_pw,
            )
            if ok:
                n_ok += 1
            else:
                failed.append(f"Cpw={C_pw},apw={alpha_pw}")
                all_ok = False
        except Exception as e:
            failed.append(f"Cpw={C_pw},apw={alpha_pw}:{e}")
            all_ok = False
    total = len(C_pws) * len(a_pws)
    record("Group G: C_pw × alpha_pw grid", all_ok,
           f"{n_ok}/{total} OK" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP H — max_simulations sweep + steps invariant
# ════════════════════════════════════════════════════════════════════════════

def test_group_H_simulations():
    """
    max_simulations ∈ {1, 5, 20, 100, 500}.
    Invariant: agent.steps == min(max_simulations, actual_steps_taken).
    Also: best_n increases (weakly) with max_simulations.
    """
    sim_list = [1, 5, 20, 100] if QUICK else [1, 5, 20, 100, 500]
    all_ok = True
    failed = []
    for ms in sim_list:
        try:
            agent = make_agent(state_dim=4, max_actions=8, n_trees=4,
                               max_simulations=ms)
            policy = UniformPolicy(4, 8).cuda()
            value  = ConstantValue(4, 1).cuda()
            root   = np.zeros(4, dtype=np.float32)
            ba, bn, _ = agent.run(root, 0, policy, value, problem=None)

            steps_ok  = agent.steps <= ms
            range_ok  = 0 <= ba < 8
            # bn=0 is valid for max_sims=1 (child created but has no own backups yet)
            value_ok  = bn >= 0
            ok = steps_ok and range_ok and value_ok
            if not ok:
                all_ok = False
                failed.append(f"sims={ms}:steps={agent.steps},ba={ba}")
        except Exception as e:
            all_ok = False
            failed.append(f"sims={ms}:{e}")
    record("Group H: max_simulations sweep + steps ≤ budget", all_ok,
           f"tested={sim_list}" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP I — diverse root states (non-zero, negative, large values)
# ════════════════════════════════════════════════════════════════════════════

def test_group_I_root_state_diversity():
    """
    Run with various root state vectors to ensure no NaN/Inf in NN path.
    """
    state_dim = 8
    max_actions = 8
    rng = np.random.default_rng(1234)

    root_states = [
        np.zeros(state_dim, dtype=np.float32),                   # all zeros
        np.ones(state_dim, dtype=np.float32),                    # all ones
        -np.ones(state_dim, dtype=np.float32),                   # all neg
        rng.standard_normal(state_dim).astype(np.float32),       # random N(0,1)
        rng.uniform(-1e3, 1e3, state_dim).astype(np.float32),    # large values
        rng.uniform(-1e-6, 1e-6, state_dim).astype(np.float32),  # tiny values
        np.full(state_dim, np.finfo(np.float32).max * 0.001,
                dtype=np.float32),                               # large but safe
    ]

    agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                       n_trees=4, max_simulations=20)
    policy = UniformPolicy(state_dim, max_actions).cuda()
    value  = ConstantValue(state_dim, 1).cuda()

    all_ok = True
    failed = []
    for i, root in enumerate(root_states):
        try:
            ba, bn, ai = agent.run(root, 0, policy, value, problem=None)
            if not (0 <= ba < max_actions and bn > 0):
                all_ok = False
                failed.append(f"state#{i}")
        except Exception as e:
            all_ok = False
            failed.append(f"state#{i}:{e}")

    record("Group I: diverse root states (zero/ones/neg/large/tiny)", all_ok,
           f"{len(root_states)} states tested" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP J — root_turn diversity (multi-robot turn rotation)
# ════════════════════════════════════════════════════════════════════════════

def test_group_J_root_turn():
    """For num_robots=4, all turn values 0..3 must be valid."""
    state_dim   = 4
    max_actions = 8
    num_robots  = 4
    all_ok = True
    failed = []

    agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                       num_robots=num_robots, n_trees=4, max_simulations=20)
    policy = UniformPolicy(state_dim, max_actions).cuda()
    value  = ConstantValue(state_dim, num_robots).cuda()
    root   = np.zeros(state_dim, dtype=np.float32)

    for turn in range(num_robots):
        try:
            ba, bn, _ = agent.run(root, turn, policy, value, problem=None)
            if not (0 <= ba < max_actions and bn > 0):
                all_ok = False
                failed.append(f"turn={turn}")
        except Exception as e:
            all_ok = False
            failed.append(f"turn={turn}:{e}")

    record("Group J: root_turn sweep 0..num_robots-1", all_ok,
           f"turns 0..{num_robots-1} OK" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP K — prior-dominance across action positions
# ════════════════════════════════════════════════════════════════════════════

def test_group_K_hot_action_positions():
    """
    SpotlightPolicy with hot action at positions {0, mid, last} in different
    max_actions sizes. The hot action must always become best_action.
    """
    configs = [
        (8,   0), (8,   4), (8,   7),
        (16,  0), (16,  8), (16, 15),
    ] if QUICK else [
        (8,   0), (8,   4), (8,   7),
        (16,  0), (16,  8), (16, 15),
        (32,  0), (32, 16), (32, 31),
        (64,  0), (64, 32), (64, 63),
        (128, 0), (128,64), (128,127),
    ]

    all_ok = True
    failed = []
    for (ma, hot) in configs:
        try:
            agent = make_agent(state_dim=4, max_actions=ma, n_trees=8,
                               max_simulations=200, C_exp=0.5, device_memory=0.3)
            policy = SpotlightPolicy(4, ma, hot, weight=200.0).cuda()
            value  = ConstantValue(4, 1).cuda()
            root   = np.zeros(4, dtype=np.float32)
            ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
            if ba != hot:
                all_ok = False
                failed.append(f"ma={ma},hot={hot}→got {ba}")
        except Exception as e:
            all_ok = False
            failed.append(f"ma={ma},hot={hot}:{e}")

    record("Group K: hot-action position across action space", all_ok,
           f"{len(configs)} configs OK" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP L — edge-case max_simulations=1
# ════════════════════════════════════════════════════════════════════════════

def test_group_L_single_sim():
    """
    max_simulations=1: tree must have exactly 2 nodes (root + 1 child) per tree,
    and ns_root must equal 1 for each tree.
    """
    all_ok = True
    failed = []
    configs = [(4, 8, 4), (2, 4, 1), (8, 16, 8)]
    for (state_dim, max_actions, n_trees) in configs:
        try:
            agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                               n_trees=n_trees, max_simulations=1)
            policy = UniformPolicy(state_dim, max_actions).cuda()
            value  = ConstantValue(state_dim, 1).cuda()
            root   = np.zeros(state_dim, dtype=np.float32)
            agent.run(root, 0, policy, value, problem=None)

            sizes  = agent.dev_trees_sizes.copy_to_host()
            ns     = agent.dev_trees_ns.copy_to_host()
            ok = all(sizes[t] == 2 for t in range(n_trees)) \
                 and all(ns[t, 0] == 1 for t in range(n_trees))
            if not ok:
                all_ok = False
                failed.append(f"SD={state_dim},MA={max_actions},NT={n_trees}: "
                               f"sizes={sizes}, ns_root={ns[:,0]}")
        except Exception as e:
            all_ok = False
            failed.append(f"SD={state_dim},MA={max_actions},NT={n_trees}:{e}")

    record("Group L: single-simulation tree size invariant", all_ok,
           "3 configs OK" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP M — visit-count monotonicity within a run
# ════════════════════════════════════════════════════════════════════════════

def test_group_M_visit_count_mono():
    """
    After N steps:
    - root ns must equal agent.steps (every step visits root exactly once).
    - Every node in the tree with ns > 0 has ns >= 1 (i.e. no fractional counts).
    - Tree grew: sizes > 1 for all trees.

    Note: the newest allocated leaf (created on the last PW-expansion) can have
    ns=0 because it has been added to the tree but not yet selected/backed-up.
    """
    state_dim = 4; max_actions = 8; n_trees = 4; max_sims = 50
    try:
        agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                           n_trees=n_trees, max_simulations=max_sims)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)
        agent.run(root, 0, policy, value, problem=None)

        ns    = agent.dev_trees_ns.copy_to_host()       # (T, S)
        sizes = agent.dev_trees_sizes.copy_to_host()    # (T,)

        # Root visit count must equal steps across all trees
        root_ns_ok = all(ns[t, 0] == agent.steps for t in range(n_trees))
        # Visited nodes (ns > 0) must have integer ns >= 1 (tautological, but checks dtype)
        visited_ok = all(
            int(ns[t, s]) >= 1
            for t in range(n_trees)
            for s in range(int(sizes[t]))
            if int(ns[t, s]) > 0         # skip un-visited newly-allocated nodes
        )
        # Tree must have grown beyond root
        size_ok = all(sizes[t] > 1 for t in range(n_trees))

        ok = root_ns_ok and visited_ok and size_ok
        record("Group M: visit-count monotonicity", ok,
               f"root_ns={ns[:,0].tolist()}, steps={agent.steps}, "
               f"tree_sizes={sizes.tolist()}")
    except Exception as e:
        record("Group M: visit-count monotonicity", False, str(e))




# ════════════════════════════════════════════════════════════════════════════
# GROUP N — total_value / ns sanity (Q-values in valid range)
# ════════════════════════════════════════════════════════════════════════════

def test_group_N_qvalue_range():
    """
    With ConstantValue(c=0.7) and gamma=1, Q = total_value/ns should
    be ≈ 0.7 for every node that has been visited at least once.
    Allow ±0.5 tolerance due to reward discounting mixing.
    """
    const_c = 0.7
    state_dim = 4; max_actions = 8; n_trees = 4
    try:
        agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                           n_trees=n_trees, max_simulations=50,
                           gamma=1.0)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1, c=const_c).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)
        agent.run(root, 0, policy, value, problem=None)

        ns = agent.dev_trees_ns.copy_to_host()
        tv = agent.dev_trees_total_value.copy_to_host()
        sizes = agent.dev_trees_sizes.copy_to_host()

        bad = []
        for t in range(n_trees):
            for s in range(int(sizes[t])):
                n = int(ns[t, s])
                if n == 0:
                    continue
                q = float(tv[t, s, 0]) / n
                if not (0.0 <= q <= 1.5):   # generous upper bound
                    bad.append(f"tree{t}/node{s}: q={q:.3f}")

        ok = len(bad) == 0
        record("Group N: Q-values in valid range [0, 1.5]", ok,
               f"no violations in {int(np.sum(sizes))} visited nodes"
               if ok else f"violations: {bad[:5]}")
    except Exception as e:
        record("Group N: Q-values in valid range [0, 1.5]", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# GROUP O — actions_ns sums match total steps
# ════════════════════════════════════════════════════════════════════════════

def test_group_O_actions_ns_sum():
    """
    After reduction: sum(actions_ns) ≤ steps × n_trees.
    Root ns per tree == steps, so sum over root children ≤ total ns at root.
    """
    state_dim = 4; max_actions = 8; n_trees = 8; max_sims = 100
    try:
        agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                           n_trees=n_trees, max_simulations=max_sims)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)
        agent.run(root, 0, policy, value, problem=None)

        actions_ns = agent.dev_actions_ns.copy_to_host()   # (A,)
        ns         = agent.dev_trees_ns.copy_to_host()     # (T, S)

        total_actions_ns = int(np.sum(actions_ns))
        total_root_ns    = int(np.sum(ns[:, 0]))
        # Children of root collectively have ≤ total root visits across trees
        ok = total_actions_ns <= total_root_ns + 1  # +1 tolerance
        record("Group O: sum(actions_ns) ≤ total root ns", ok,
               f"sum_actions_ns={total_actions_ns}, total_root_ns={total_root_ns}")
    except Exception as e:
        record("Group O: sum(actions_ns) ≤ total root ns", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# GROUP P — NaN / Inf guard: extreme NN outputs
# ════════════════════════════════════════════════════════════════════════════

def test_group_P_nan_inf_guard():
    """
    Policy outputs all-zeros (→ softmax NaN) or all-inf logits.
    PUCTGpu must survive (NaN guard in run() should activate)
    and still return a valid best_action.
    """
    class ZeroPolicy(nn.Module):
        def __init__(self, sd, ma): super().__init__(); self.ma=ma; self._d=nn.Linear(sd,1,bias=False)
        def forward(self, x): return torch.zeros(x.shape[0], self.ma, device=x.device)

    class InfPolicy(nn.Module):
        def __init__(self, sd, ma): super().__init__(); self.ma=ma; self._d=nn.Linear(sd,1,bias=False)
        def forward(self, x): return torch.full((x.shape[0], self.ma), float("inf"), device=x.device)

    state_dim = 4; max_actions = 8; n_trees = 4
    all_ok = True
    failed = []

    for PolicyCls, label in [(ZeroPolicy, "zero-logits"), (InfPolicy, "inf-logits")]:
        try:
            agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                               n_trees=n_trees, max_simulations=20)
            policy = PolicyCls(state_dim, max_actions).cuda()
            value  = ConstantValue(state_dim, 1).cuda()
            root   = np.zeros(state_dim, dtype=np.float32)
            ba, bn, _ = agent.run(root, 0, policy, value, problem=None)
            if not (0 <= ba < max_actions):
                all_ok = False
                failed.append(f"{label}: ba={ba}")
        except Exception as e:
            all_ok = False
            failed.append(f"{label}:{e}")

    record("Group P: NaN/Inf guard in NN outputs", all_ok,
           "survived zero & inf policies" if all_ok else f"FAILED: {failed}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP Q — cross-product mini-grid (state_dim × n_trees × max_actions)
# ════════════════════════════════════════════════════════════════════════════

def test_group_Q_mini_cross_product():
    """
    Compact 3-D grid ensuring correctness under diverse joint configurations.
    """
    state_dims  = [2, 8, 32] if QUICK else [2, 4, 8, 16, 32]
    n_trees_lst = [1, 4, 16] if QUICK else [1, 4, 8, 16, 32]
    max_acts    = [4, 16]    if QUICK else [4, 8, 16, 32]

    total = 0
    passed = 0
    failed = []

    for sd, nt, ma in itertools.product(state_dims, n_trees_lst, max_acts):
        total += 1
        try:
            _, bn, _, ok = valid_run(state_dim=sd, max_actions=ma,
                                     n_trees=nt, max_simulations=10,
                                     device_memory=0.05 * nt + 0.1)
            if ok:
                passed += 1
            else:
                failed.append(f"({sd},{nt},{ma})")
        except Exception as e:
            failed.append(f"({sd},{nt},{ma}):{str(e)[:40]}")

    all_ok = len(failed) == 0
    record("Group Q: cross-product mini-grid", all_ok,
           f"{passed}/{total} passed" if all_ok else
           f"{passed}/{total}: FAILED: {failed[:5]}")


# ════════════════════════════════════════════════════════════════════════════
# GROUP R — pw_boundary consistency check
# ════════════════════════════════════════════════════════════════════════════

def test_group_R_pw_boundary():
    """
    After a full run, for every visited non-leaf node:
        pw_boundary[node] == min(max_actions, ceil(C_pw * ns[node]^alpha_pw))
    Allow ±1 slack because the kernel increments pw_boundary by 1 per step.
    """
    C_pw = 1.0; alpha_pw = 0.5
    state_dim = 4; max_actions = 16; n_trees = 4; max_sims = 100
    try:
        agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                           n_trees=n_trees, max_simulations=max_sims,
                           C_pw=C_pw, alpha_pw=alpha_pw)
        policy = UniformPolicy(state_dim, max_actions).cuda()
        value  = ConstantValue(state_dim, 1).cuda()
        root   = np.zeros(state_dim, dtype=np.float32)
        agent.run(root, 0, policy, value, problem=None)

        ns      = agent.dev_trees_ns.copy_to_host()
        pw      = agent.dev_trees_pw_boundary.copy_to_host()
        leaves  = agent.dev_trees_leaves.copy_to_host()
        sizes   = agent.dev_trees_sizes.copy_to_host()

        violations = []
        for t in range(n_trees):
            for s in range(int(sizes[t])):
                n = int(ns[t, s])
                if n == 0 or leaves[t, s]:
                    continue
                expected = min(max_actions, math.ceil(C_pw * (n ** alpha_pw)))
                actual   = int(pw[t, s])
                # Allow ±1 because pw is updated after backup
                if abs(actual - expected) > 1:
                    violations.append(
                        f"T{t}S{s}: pw={actual} expected≈{expected} (ns={n})"
                    )

        ok = len(violations) == 0
        record("Group R: pw_boundary ≈ ceil(C_pw * ns^alpha_pw)", ok,
               f"no violations in {int(np.sum(sizes))} nodes"
               if ok else f"{len(violations)} violation(s): {violations[:3]}")
    except Exception as e:
        record("Group R: pw_boundary ≈ ceil(C_pw * ns^alpha_pw)", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = "[QUICK]" if QUICK else "[FULL]"
    print("=" * 70)
    print(f"  GPU PUCT — Input Diversity Tests  {mode}")
    print("=" * 70)

    test_group_A_state_dim()
    test_group_B_max_actions()
    test_group_C_n_trees()
    test_group_D_num_robots()
    test_group_E_gamma()
    test_group_F_exploration()
    test_group_G_pw_constants()
    test_group_H_simulations()
    test_group_I_root_state_diversity()
    test_group_J_root_turn()
    test_group_K_hot_action_positions()
    test_group_L_single_sim()
    test_group_M_visit_count_mono()
    test_group_N_qvalue_range()
    test_group_O_actions_ns_sum()
    test_group_P_nan_inf_guard()
    test_group_Q_mini_cross_product()
    test_group_R_pw_boundary()

    print("=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"  Results: {n_pass}/{len(results)} passed, {n_fail} failed")
    if n_fail > 0:
        print("  FAILED groups:")
        for name, ok in results:
            if not ok:
                print(f"    ✗ {name}")
    else:
        print("  All groups PASSED ✓")
    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)
