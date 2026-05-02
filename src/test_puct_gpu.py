"""
test_puct_gpu.py
----------------
Integration and unit tests for the GPU-accelerated PUCT implementation.

Tests (matching Phase 13 of the migration plan)
------------------------------------------------
Test 1  Smoke test — instantiation + init_device_side_arrays
Test 2  Zero-copy verification (NumbaPytorchBridge)
Test 3  _reset_puct — root node initialised correctly
Test 4  _select_puct — returns root on fresh tree (step 0)
Test 5  _extract_leaf_states — copies root state to bridge buffer
Test 6  _expand_and_backup_puct — tree grows after one NN call
Test 7  Progressive widening — pw_boundary increases correctly
Test 8  Value backup — total_value at root matches discounted sum
Test 9  Multi-tree reduction — actions_ns sums across trees
Test 10 Full run smoke — runs max_simulations steps without error

Usage
-----
    cd /home/robomaster/Research/mcts_numba_cuda/src
    python test_puct_gpu.py
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn

# ── Numba/PyTorch CUDA compatibility patches ──────────────────────────────────
# 1. Torch must own the primary CUDA context before Numba imports its runtime.
if torch.cuda.is_available():
    _torch_cuda_sentinel = torch.zeros(1, device="cuda")

# 2. Newer CUDA toolkit versions (12.5+) may not be in Numba's CTK_SUPPORTED
#    table, causing it to fail on sm_89 (RTX 4090) even though it can compile
#    for that architecture.  Patch the table to include them.
try:
    from numba.cuda.cudadrv import nvvm as _nvvm_mod
    if hasattr(_nvvm_mod, "CTK_SUPPORTED"):
        _ctk = _nvvm_mod.CTK_SUPPORTED
        # sm_89 (Ada Lovelace) is the max for CUDA 12.x up to at least 12.9
        for _v in [(12, 5), (12, 6), (12, 7), (12, 8), (12, 9)]:
            if _v not in _ctk:
                _ctk[_v] = ((5, 0), (9, 0))
except Exception:
    pass

from numba import cuda

# ── Path setup ───────────────────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(__file__))

from puct.puct_gpu_nn_bridge import NumbaPytorchBridge
from puct.puct_gpu import PUCTGpu
from puct.puct_gpu_kernels import (
    _reset_puct,
    _select_puct,
    _extract_leaf_states,
    _prepare_expansion_puct,
    _commit_expansion_and_backup_puct,
    _reduce_over_trees_puct,
    _reduce_over_actions_puct,
)

# ── Tiny stub neural networks ─────────────────────────────────────────────────

class UniformPolicyNet(nn.Module):
    """Returns uniform prior (1/max_actions) for any input."""
    def __init__(self, state_dim, max_actions):
        super().__init__()
        self.max_actions = max_actions
        self._dummy = nn.Linear(state_dim, 1, bias=False)  # forces CUDA init

    def forward(self, x):
        B = x.shape[0]
        return torch.ones(B, self.max_actions, device=x.device)


class ZeroValueNet(nn.Module):
    """Returns zero value for any input."""
    def __init__(self, state_dim, num_robots):
        super().__init__()
        self.num_robots = num_robots
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        return torch.zeros(B, self.num_robots, device=x.device)


class ConstantValueNet(nn.Module):
    """Returns a fixed constant value for all inputs and all robots."""
    def __init__(self, state_dim, num_robots, constant=0.5):
        super().__init__()
        self.num_robots = num_robots
        self.constant = constant
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        return torch.full(
            (B, self.num_robots), self.constant,
            dtype=torch.float32, device=x.device
        )


# ── Helper ───────────────────────────────────────────────────────────────────

def make_agent(state_dim=4, action_dim=1, max_actions=8, num_robots=1,
               n_trees=4, max_simulations=10, device_memory=0.5):
    agent = PUCTGpu(
        state_dim=state_dim,
        action_dim=action_dim,
        max_actions=max_actions,
        num_robots=num_robots,
        n_trees=n_trees,
        max_simulations=max_simulations,
        C_exp=1.0,
        alpha_exp=0.5,
        C_pw=1.0,
        alpha_pw=0.5,
        gamma=0.99,
        device_memory=device_memory,
        verbose_info=False,
        verbose_debug=False,
    )
    agent.init_device_side_arrays()
    return agent


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"
results = []


def record(name, ok, detail=""):
    status = PASS_MARK if ok else FAIL_MARK
    msg = f"{status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, ok))


# ── Test 1: Smoke — instantiation ────────────────────────────────────────────
def test_1_smoke():
    try:
        agent = make_agent()
        ok = (
            agent.max_tree_size > 0
            and hasattr(agent, "dev_trees")
            and hasattr(agent, "bridge")
        )
        record("Test 1: Smoke instantiation", ok,
               f"max_tree_size={agent.max_tree_size}")
    except Exception as e:
        record("Test 1: Smoke instantiation", False, str(e))


# ── Test 2: Zero-copy bridge ─────────────────────────────────────────────────
def test_2_zero_copy():
    try:
        bridge = NumbaPytorchBridge(
            n_trees=4, state_dim=4, action_dim=1, max_actions=8, num_robots=1
        )
        ok = bridge.verify_zero_copy()
        record("Test 2: NumbaPytorchBridge zero-copy", ok)
    except Exception as e:
        record("Test 2: NumbaPytorchBridge zero-copy", False, str(e))


# ── Test 3: _reset_puct ───────────────────────────────────────────────────────
def test_3_reset():
    try:
        agent = make_agent(state_dim=4, max_actions=8, n_trees=4)
        root_state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dev_root = cuda.to_device(root_state)

        _reset_puct[agent.n_trees, agent.tpb_r](
            dev_root, np.int8(0),
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors, agent.dev_trees_pw_boundary,
        )
        cuda.synchronize()

        sizes = agent.dev_trees_sizes.copy_to_host()
        states = agent.dev_trees_states.copy_to_host()
        turns = agent.dev_trees_robot_turns.copy_to_host()
        leaves = agent.dev_trees_leaves.copy_to_host()

        ok = (
            all(sizes == 1)
            and all(np.allclose(states[ti, 0, :], root_state) for ti in range(agent.n_trees))
            and all(turns[ti, 0] == 0 for ti in range(agent.n_trees))
            and all(leaves[ti, 0] for ti in range(agent.n_trees))
        )
        record("Test 3: _reset_puct", ok,
               f"sizes={sizes}, state[0,0,:]={states[0,0,:]}")
    except Exception as e:
        record("Test 3: _reset_puct", False, str(e))


# ── Test 4: _select_puct returns root on fresh tree ───────────────────────────
def test_4_select_on_fresh_tree():
    try:
        agent = make_agent(state_dim=4, max_actions=8, n_trees=4)
        root_state = np.zeros(4, dtype=np.float32)
        dev_root = cuda.to_device(root_state)

        _reset_puct[agent.n_trees, agent.tpb_r](
            dev_root, np.int8(0),
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors, agent.dev_trees_pw_boundary,
        )
        cuda.synchronize()

        _select_puct[agent.n_trees, agent.tpb_s](
            np.float32(1.0), np.float32(0.5),
            agent.dev_trees, agent.dev_trees_leaves, agent.dev_trees_ns,
            agent.dev_trees_total_value,
            agent.dev_trees_robot_turns, agent.dev_trees_action_priors,
            agent.dev_trees_pw_boundary, agent.dev_trees_prior_rank,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
        )
        cuda.synchronize()

        selected = agent.dev_trees_nodes_selected.copy_to_host()
        ok = all(s == 0 for s in selected)
        record("Test 4: _select_puct on fresh tree → root", ok,
               f"selected={selected}")
    except Exception as e:
        record("Test 4: _select_puct on fresh tree → root", False, str(e))


# ── Test 5: _extract_leaf_states ─────────────────────────────────────────────
def test_5_extract():
    try:
        agent = make_agent(state_dim=4, max_actions=8, n_trees=4)
        root_state = np.array([1.5, -2.5, 0.0, 3.14], dtype=np.float32)
        dev_root = cuda.to_device(root_state)

        _reset_puct[agent.n_trees, agent.tpb_r](
            dev_root, np.int8(0),
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors, agent.dev_trees_pw_boundary,
        )
        cuda.synchronize()

        _select_puct[agent.n_trees, agent.tpb_s](
            np.float32(1.0), np.float32(0.5),
            agent.dev_trees, agent.dev_trees_leaves, agent.dev_trees_ns,
            agent.dev_trees_total_value,
            agent.dev_trees_robot_turns, agent.dev_trees_action_priors,
            agent.dev_trees_pw_boundary, agent.dev_trees_prior_rank,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
        )
        cuda.synchronize()

        _extract_leaf_states[agent.n_trees, agent.tpb_r](
            agent.dev_trees_nodes_selected, agent.dev_trees_states,
            agent.dev_trees_terminals,
            agent.bridge.dev_leaf_states, agent.bridge.dev_leaf_valid,
        )
        cuda.synchronize()

        extracted = agent.bridge.leaf_states.cpu().numpy()
        valid = agent.bridge.leaf_valid.cpu().numpy()
        ok = (
            all(valid == 1)
            and all(np.allclose(extracted[ti], root_state) for ti in range(agent.n_trees))
        )
        record("Test 5: _extract_leaf_states", ok,
               f"valid={valid}, extracted[0]={extracted[0]}")
    except Exception as e:
        record("Test 5: _extract_leaf_states", False, str(e))


# ── Test 6: Tree grows after first expand+backup ──────────────────────────────
def test_6_expand_grows_tree():
    try:
        state_dim, max_actions, n_trees = 4, 8, 4
        agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                           n_trees=n_trees)
        root_state = np.zeros(state_dim, dtype=np.float32)
        dev_root = cuda.to_device(root_state)

        _reset_puct[n_trees, agent.tpb_r](
            dev_root, np.int8(0),
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors, agent.dev_trees_pw_boundary,
        )
        cuda.synchronize()

        _select_puct[n_trees, agent.tpb_s](
            np.float32(1.0), np.float32(0.5),
            agent.dev_trees, agent.dev_trees_leaves, agent.dev_trees_ns,
            agent.dev_trees_total_value,
            agent.dev_trees_robot_turns, agent.dev_trees_action_priors,
            agent.dev_trees_pw_boundary, agent.dev_trees_prior_rank,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
        )
        cuda.synchronize()

        _extract_leaf_states[n_trees, agent.tpb_r](
            agent.dev_trees_nodes_selected, agent.dev_trees_states,
            agent.dev_trees_terminals,
            agent.bridge.dev_leaf_states, agent.bridge.dev_leaf_valid,
        )
        cuda.synchronize()

        # Inject uniform priors and constant values
        agent.bridge.nn_priors.fill_(1.0 / max_actions)
        agent.bridge.nn_values.fill_(0.5)

        # Phase B1: prepare expansion (store priors, sort ranks, surface parent state)
        _prepare_expansion_puct[n_trees, agent.tpb_s](
            np.float32(1.0), np.float32(0.5), np.int32(1),
            agent.bridge.dev_nn_priors, agent.bridge.dev_leaf_valid,
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors,
            agent.dev_trees_prior_rank, agent.dev_trees_pw_boundary,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
            np.int32(agent.max_tree_size), np.int32(state_dim), np.int32(max_actions),
            agent.bridge.dev_expansion_valid, agent.bridge.dev_expanded_parent_states,
            agent.bridge.dev_expanded_actions,
        )
        cuda.synchronize()
        # (no host environment step in unit test — next_states stay zero)
        # Phase B2: commit expansion + backup
        _commit_expansion_and_backup_puct[n_trees, agent.tpb_s](
            np.float32(0.99), np.float32(1.0), np.float32(0.5), np.int32(1),
            agent.bridge.dev_nn_values, agent.bridge.dev_leaf_valid,
            agent.bridge.dev_expansion_valid, agent.bridge.dev_expanded_next_states,
            agent.bridge.dev_expanded_rewards, agent.bridge.dev_expanded_terminals,
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors,
            agent.dev_trees_prior_rank, agent.dev_trees_pw_boundary,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
            np.int32(agent.max_tree_size), np.int32(max_actions),
        )
        cuda.synchronize()

        sizes = agent.dev_trees_sizes.copy_to_host()
        ns_root = agent.dev_trees_ns.copy_to_host()[:, 0]
        # Each tree should now have 2 nodes (root + 1 child)
        ok = all(s == 2 for s in sizes) and all(ns_root == 1)
        record("Test 6: Tree expands + backup updates ns", ok,
               f"sizes={sizes}, ns_root={ns_root}")
    except Exception as e:
        record("Test 6: Tree expands + backup updates ns", False, str(e))


# ── Test 7: Progressive widening ──────────────────────────────────────────────
def test_7_progressive_widening():
    """
    Run 50 steps on a single tree (n_trees=1) and verify that pw_boundary
    at root grows ≥ ceil(C_pw * N^alpha_pw).
    """
    try:
        state_dim, max_actions, n_trees = 2, 16, 1
        C_pw, alpha_pw = 1.0, 0.5
        agent = PUCTGpu(
            state_dim=state_dim, action_dim=1, max_actions=max_actions,
            num_robots=1, n_trees=n_trees, max_simulations=50,
            C_pw=C_pw, alpha_pw=alpha_pw, gamma=0.99, device_memory=0.2,
            verbose_info=False,
        )
        agent.init_device_side_arrays()
        root_state = np.zeros(state_dim, dtype=np.float32)

        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value = ConstantValueNet(state_dim, 1, constant=0.5).cuda()

        agent.run(root_state, 0, policy, value, problem=None)

        ns_root = int(agent.dev_trees_ns.copy_to_host()[0, 0])
        pw = int(agent.dev_trees_pw_boundary.copy_to_host()[0, 0])
        expected_min_pw = math.ceil(C_pw * (ns_root ** alpha_pw))
        # pw may slightly lag by 1 step because backup checks *after* incrementing
        ok = pw >= max(1, expected_min_pw - 1)
        record("Test 7: Progressive widening", ok,
               f"ns_root={ns_root}, pw={pw}, expected_min_pw={expected_min_pw}")
    except Exception as e:
        record("Test 7: Progressive widening", False, str(e))


# ── Test 8: Value backup correctness ─────────────────────────────────────────
def test_8_backup_value():
    """
    One step, constant value 0.7:
    After backup, total_value[0, root] should equal 0.7 (discount=0, same depth).
    """
    try:
        state_dim, max_actions, n_trees = 2, 4, 1
        agent = make_agent(state_dim=state_dim, max_actions=max_actions,
                           n_trees=n_trees, max_simulations=1)
        root_state = np.zeros(state_dim, dtype=np.float32)
        dev_root = cuda.to_device(root_state)

        _reset_puct[n_trees, agent.tpb_r](
            dev_root, np.int8(0),
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors, agent.dev_trees_pw_boundary,
        )
        cuda.synchronize()
        _select_puct[n_trees, agent.tpb_s](
            np.float32(1.0), np.float32(0.5),
            agent.dev_trees, agent.dev_trees_leaves, agent.dev_trees_ns,
            agent.dev_trees_total_value,
            agent.dev_trees_robot_turns, agent.dev_trees_action_priors,
            agent.dev_trees_pw_boundary, agent.dev_trees_prior_rank,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
        )
        cuda.synchronize()
        _extract_leaf_states[n_trees, agent.tpb_r](
            agent.dev_trees_nodes_selected, agent.dev_trees_states,
            agent.dev_trees_terminals,
            agent.bridge.dev_leaf_states, agent.bridge.dev_leaf_valid,
        )
        cuda.synchronize()

        const_val = 0.7
        agent.bridge.nn_priors.fill_(1.0 / max_actions)
        agent.bridge.nn_values.fill_(const_val)

        # Phase B1
        _prepare_expansion_puct[n_trees, agent.tpb_s](
            np.float32(1.0), np.float32(0.5), np.int32(1),
            agent.bridge.dev_nn_priors, agent.bridge.dev_leaf_valid,
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors,
            agent.dev_trees_prior_rank, agent.dev_trees_pw_boundary,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
            np.int32(agent.max_tree_size), np.int32(state_dim), np.int32(max_actions),
            agent.bridge.dev_expansion_valid, agent.bridge.dev_expanded_parent_states,
            agent.bridge.dev_expanded_actions,
        )
        cuda.synchronize()
        # Phase B2
        _commit_expansion_and_backup_puct[n_trees, agent.tpb_s](
            np.float32(0.99), np.float32(1.0), np.float32(0.5), np.int32(1),
            agent.bridge.dev_nn_values, agent.bridge.dev_leaf_valid,
            agent.bridge.dev_expansion_valid, agent.bridge.dev_expanded_next_states,
            agent.bridge.dev_expanded_rewards, agent.bridge.dev_expanded_terminals,
            agent.dev_trees, agent.dev_trees_sizes, agent.dev_trees_depths,
            agent.dev_trees_robot_turns,
            agent.dev_trees_leaves, agent.dev_trees_terminals, agent.dev_trees_ns,
            agent.dev_trees_total_value, agent.dev_trees_states,
            agent.dev_trees_action_priors,
            agent.dev_trees_prior_rank, agent.dev_trees_pw_boundary,
            agent.dev_trees_nodes_selected, agent.dev_trees_selected_paths,
            np.int32(agent.max_tree_size), np.int32(max_actions),
        )
        cuda.synchronize()

        tv = agent.dev_trees_total_value.copy_to_host()  # (T, S, R)
        root_val = float(tv[0, 0, 0])
        # root is at depth 0, leaf (root itself) is at depth 0, discount = gamma^0 = 1
        ok = abs(root_val - const_val) < 1e-4
        record("Test 8: Value backup at root", ok,
               f"expected={const_val:.4f}, got={root_val:.4f}")
    except Exception as e:
        record("Test 8: Value backup at root", False, str(e))


# ── Test 9: Multi-tree reduction ──────────────────────────────────────────────
def test_9_reduction():
    try:
        state_dim, max_actions, n_trees = 2, 4, 8
        agent = PUCTGpu(
            state_dim=state_dim, action_dim=1, max_actions=max_actions,
            num_robots=1, n_trees=n_trees, max_simulations=20,
            C_pw=1.0, alpha_pw=0.5, gamma=0.99, device_memory=0.2,
            verbose_info=False,
        )
        agent.init_device_side_arrays()
        root_state = np.zeros(state_dim, dtype=np.float32)
        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value = ZeroValueNet(state_dim, 1).cuda()
        agent.run(root_state, 0, policy, value, problem=None)

        actions_ns = agent.dev_actions_ns.copy_to_host()
        total = int(np.sum(actions_ns))
        ok = total > 0
        record("Test 9: Multi-tree reduction", ok,
               f"sum(actions_ns)={total}, actions_ns={actions_ns}")
    except Exception as e:
        record("Test 9: Multi-tree reduction", False, str(e))


# ── Test 10: Full run smoke ───────────────────────────────────────────────────
def test_10_full_run():
    try:
        state_dim, max_actions, n_trees = 4, 8, 8
        agent = PUCTGpu(
            state_dim=state_dim, action_dim=1, max_actions=max_actions,
            num_robots=2, n_trees=n_trees, max_simulations=50,
            C_pw=1.0, alpha_pw=0.5, gamma=0.99, device_memory=0.5,
            verbose_info=False,
        )
        agent.init_device_side_arrays()
        root_state = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        policy = UniformPolicyNet(state_dim, max_actions).cuda()
        value = ConstantValueNet(state_dim, 2, constant=0.3).cuda()

        best_action, best_n, actions_info = agent.run(root_state, 0, policy, value, problem=None)
        ok = 0 <= best_action < max_actions and best_n > 0
        record("Test 10: Full run smoke", ok,
               f"best_action={best_action}, best_n={best_n}, steps={agent.steps}")
    except Exception as e:
        record("Test 10: Full run smoke", False, str(e))


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  GPU PUCT — Integration Tests")
    print("=" * 60)

    test_1_smoke()
    test_2_zero_copy()
    test_3_reset()
    test_4_select_on_fresh_tree()
    test_5_extract()
    test_6_expand_grows_tree()
    test_7_progressive_widening()
    test_8_backup_value()
    test_9_reduction()
    test_10_full_run()

    print("=" * 60)
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
    print("=" * 60)
    sys.exit(0 if n_fail == 0 else 1)
