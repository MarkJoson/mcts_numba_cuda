"""
puct_gpu_kernels.py
-------------------
All ``@cuda.jit`` kernel functions for the GPU-accelerated PUCT algorithm.

Kernel naming follows the plan phases:

    _reset_puct,
    _select_puct,
    _extract_leaf_states,
    _prepare_expansion_puct,
    _commit_expansion_and_backup_puct,
    _reduce_over_trees_puct,
    _reduce_over_actions_puct,

Architecture
------------
Every kernel is a **static free function** (not a method) to avoid Numba
class-method compilation issues.  The ``PUCTGpu`` class in ``puct_gpu.py``
calls them as module-level symbols.

One CUDA block per tree in all kernels.  Threads within a block cooperate
via shared memory for reductions (argmax for select, sum for reduce).
There are **no atomic operations** and **no mutexes** — each tree is
completely independent.

References
----------
- Migration plan: Phase 4-10 code snippets (direct transcription + fixes).
- mctsnc.py: ``_reset`` (L1259-1290), ``_select`` (L1292-1365) as structural
  references.
"""

import math
from numpy import inf

# PyTorch must initialise the CUDA primary context before Numba does.
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _torch_init_sentinel = _torch.zeros(1, device="cuda")
except ImportError:
    pass

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

from numba import cuda, void, int8, int16, int32, int64, float32, boolean


# ════════════════════════════════════════════════════════════════════════════
# Phase 4 — _reset_puct
# ════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _reset_puct(root_state, root_turn,
                trees, trees_sizes, trees_depths, trees_robot_turns,
                trees_leaves, trees_terminals, trees_ns,
                trees_total_value, trees_states, trees_action_priors,
                trees_pw_boundary):
    """
    Reset all trees to a single root node carrying *root_state*.

    One block per tree; threads cooperate to copy state and zero value arrays.

    Parameters
    ----------
    root_state : float32[state_dim]
        The initial continuous state vector.
    root_turn : int8
        Which robot acts first (0 .. num_robots-1).
    trees : int32[n_trees, max_tree_size, 1+max_actions]
        Node topology.
    trees_sizes : int32[n_trees]
    trees_depths : int16[n_trees, max_tree_size]
    trees_robot_turns : int8[n_trees, max_tree_size]
    trees_leaves : bool[n_trees, max_tree_size]
    trees_terminals : bool[n_trees, max_tree_size]
    trees_ns : int32[n_trees, max_tree_size]
    trees_total_value : float32[n_trees, max_tree_size, num_robots]
    trees_states : float32[n_trees, max_tree_size, state_dim]
    trees_action_priors : float32[n_trees, max_tree_size, max_actions]
    trees_pw_boundary : int16[n_trees, max_tree_size]
    """
    ti = cuda.blockIdx.x
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    state_dim = root_state.shape[0]
    max_actions = trees.shape[2] - 1
    num_robots = trees_total_value.shape[2]

    # ── Scalar root-node metadata (thread 0 only) ───────────────────────────
    if t == 0:
        trees[ti, 0, 0] = int32(-1)        # root has no parent
        trees_sizes[ti] = int32(1)
        trees_depths[ti, 0] = int16(0)
        trees_robot_turns[ti, 0] = int8(root_turn)
        trees_leaves[ti, 0] = True
        trees_terminals[ti, 0] = False
        trees_ns[ti, 0] = int32(0)
        trees_pw_boundary[ti, 0] = int16(0)
        # Children slots: -1 (no child)
        for a in range(max_actions):
            trees[ti, 0, 1 + a] = int32(-1)

    # ── Parallel copy of root state ─────────────────────────────────────────
    e = t
    while e < state_dim:
        trees_states[ti, 0, e] = root_state[e]
        e += tpb

    # ── Zero total_value for root ────────────────────────────────────────────
    e = t
    while e < num_robots:
        trees_total_value[ti, 0, e] = float32(0.0)
        e += tpb

    # ── Zero action priors for root ──────────────────────────────────────────
    e = t
    while e < max_actions:
        trees_action_priors[ti, 0, e] = float32(0.0)
        e += tpb


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 — _select_puct
# ════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _select_puct(C_exp, alpha_exp,
                 trees, trees_leaves, trees_ns, trees_total_value,
                 trees_robot_turns, trees_action_priors, trees_pw_boundary,
                 trees_prior_rank,
                 trees_nodes_selected, trees_selected_paths):
    """
    Walk each tree from root to a leaf using the PUCT UCB formula.

    PUCT UCB (from puct_v1.hpp L140):
        Q(child, robot)  =  total_value[child, robot] / ns[child]
        U(child, parent) =  C_exp * sqrt(ns[parent]^alpha_exp / ns[child])
        UCB              =  Q + U

    Progressive widening: only children with rank < pw_boundary[node] are
    considered.  The ``trees_prior_rank`` array maps rank → action_index.

    Path is stored in ``trees_selected_paths[ti, 0..depth]``; the path
    length is stored in ``trees_selected_paths[ti, -1]``.
    """
    # Shared memory — 512 slots covers max_actions ≤ 512 (MCTSNC convention)
    shared_ucbs = cuda.shared.array(512, dtype=float32)
    shared_best_child = cuda.shared.array(512, dtype=int32)
    shared_selected_path = cuda.shared.array(2050, dtype=int32)

    ti = cuda.blockIdx.x
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    max_actions = trees.shape[2] - 1

    node = int32(0)
    depth = int16(0)

    if t == 0:
        shared_selected_path[0] = int32(0)  # root is always on path

    cuda.syncthreads()

    max_depth = int16(trees_selected_paths.shape[1] - 2)  # shape[-1] = MAX_TREE_DEPTH + 2
    while not trees_leaves[ti, node] and depth < max_depth:
        pw_limit = int32(trees_pw_boundary[ti, node])
        robot_turn = int32(trees_robot_turns[ti, node])
        parent_n = int32(trees_ns[ti, node])

        # Each thread evaluates one action slot
        if t < max_actions:
            rank = t
            if rank < pw_limit:
                action = int32(trees_prior_rank[ti, node, rank])
                child = int32(trees[ti, node, 1 + action])
                shared_best_child[t] = child
                if child == int32(-1):
                    shared_ucbs[t] = -float32(inf)
                else:
                    child_n = int32(trees_ns[ti, child])
                    if child_n == int32(0):
                        shared_ucbs[t] = float32(inf)
                    else:
                        q = trees_total_value[ti, child, robot_turn] / float32(child_n)
                        exploration = C_exp * math.sqrt(
                            math.pow(float32(parent_n), alpha_exp) / float32(child_n)
                        )
                        shared_ucbs[t] = q + exploration
            else:
                shared_ucbs[t] = -float32(inf)
                shared_best_child[t] = int32(-1)
        else:
            shared_ucbs[t] = -float32(inf)
            shared_best_child[t] = int32(-1)

        cuda.syncthreads()

        # Max-argmax reduction (identical pattern to mctsnc.py L1343-1351)
        stride = tpb >> 1
        while stride > 0:
            if t < stride:
                t_stride = t + stride
                if shared_ucbs[t] < shared_ucbs[t_stride]:
                    shared_ucbs[t] = shared_ucbs[t_stride]
                    shared_best_child[t] = shared_best_child[t_stride]
            cuda.syncthreads()
            stride >>= 1

        # Descend to best child
        node = shared_best_child[0]
        depth += int16(1)
        if t == 0:
            shared_selected_path[depth] = node

        cuda.syncthreads()

    # Write path back to global memory
    path_length = int32(depth) + int32(1)
    e = t
    while e < path_length:
        trees_selected_paths[ti, e] = shared_selected_path[e]
        e += tpb
    if t == 0:
        trees_nodes_selected[ti] = node
        trees_selected_paths[ti, -1] = path_length


# ════════════════════════════════════════════════════════════════════════════
# Phase 6 — _extract_leaf_states
# ════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _extract_leaf_states(trees_nodes_selected, trees_states, trees_terminals,
                         leaf_states_buffer, leaf_valid_buffer):
    """
    Copy each tree's selected leaf state into the shared NN input buffer.

    Terminal leaves set ``leaf_valid_buffer[ti] = 0`` so that the host skip
    NN evaluation for them.  Non-terminal leaves set it to ``1``.
    """
    ti = cuda.blockIdx.x
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    selected = int32(trees_nodes_selected[ti])
    state_dim = trees_states.shape[2]

    if trees_terminals[ti, selected]:
        if t == 0:
            leaf_valid_buffer[ti] = int32(0)
        return

    if t == 0:
        leaf_valid_buffer[ti] = int32(1)

    # Parallel copy
    e = t
    while e < state_dim:
        leaf_states_buffer[ti, e] = trees_states[ti, selected, e]
        e += tpb


# ════════════════════════════════════════════════════════════════════════════
# Phase 8+9 — _prepare_expansion_puct & _commit_expansion_and_backup_puct
# ════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _prepare_expansion_puct(
        C_pw, alpha_pw, num_robots,
        nn_priors, leaf_valid,
        trees, trees_sizes, trees_depths, trees_robot_turns,
        trees_leaves, trees_terminals, trees_ns,
        trees_total_value, trees_states, trees_action_priors,
        trees_prior_rank, trees_pw_boundary,
        trees_nodes_selected, trees_selected_paths,
        max_tree_size, state_dim, max_actions,
        expansion_valid_out, parent_states_out, actions_out):
    """
    Expansion Phase 1: Store priors, sort ranks, prep the best action.
    """
    ti = cuda.blockIdx.x
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    selected = int32(trees_nodes_selected[ti])

    if t == 0:
        expansion_valid_out[ti] = int32(0)

    # ── EXPANSION PREP ────────────────────────────────────────────────────────
    if t == 0 and leaf_valid[ti] == int32(1):
        for a in range(max_actions):
            trees_action_priors[ti, selected, a] = nn_priors[ti, a]

        current_size = trees_sizes[ti]
        if current_size < max_tree_size and not trees_terminals[ti, selected]:
            # Sort priors
            for a in range(max_actions):
                trees_prior_rank[ti, selected, a] = int16(a)
            for i in range(1, max_actions):
                key_rank = trees_prior_rank[ti, selected, i]
                key_prior = nn_priors[ti, key_rank]
                j = i - 1
                while j >= 0 and nn_priors[ti, int32(trees_prior_rank[ti, selected, j])] < key_prior:
                    trees_prior_rank[ti, selected, j + 1] = trees_prior_rank[ti, selected, j]
                    j -= 1
                trees_prior_rank[ti, selected, j + 1] = int16(key_rank)

            best_action = int32(trees_prior_rank[ti, selected, 0])
            
            # Export to Host for Simulation
            expansion_valid_out[ti] = int32(1)
            actions_out[ti, 0] = float32(best_action) # Can map to continuous if needed by host
            for e in range(state_dim):
                parent_states_out[ti, e] = trees_states[ti, selected, e]

@cuda.jit
def _commit_expansion_and_backup_puct(
        gamma, C_pw, alpha_pw, num_robots,
        nn_values, leaf_valid, expansion_valid,
        expanded_next_states, expanded_rewards, expanded_terminals,
        trees, trees_sizes, trees_depths, trees_robot_turns,
        trees_leaves, trees_terminals, trees_ns,
        trees_total_value, trees_states, trees_action_priors,
        trees_prior_rank, trees_pw_boundary,
        trees_nodes_selected, trees_selected_paths,
        max_tree_size, max_actions):
    """
    Expansion Phase 2: Consume Host simulated data, allocate child, run backup with exact rewards.
    """
    ti = cuda.blockIdx.x
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    selected = int32(trees_nodes_selected[ti])
    state_dim = expanded_next_states.shape[1]

    # ── COMMIT EXPANSION ─────────────────────────────────────────────────────
    if t == 0 and leaf_valid[ti] == int32(1):
        if expansion_valid[ti] == int32(1):
            trees_leaves[ti, selected] = False
            trees_pw_boundary[ti, selected] = int16(1)

            best_action = int32(trees_prior_rank[ti, selected, 0])
            child_idx = trees_sizes[ti]

            # ── bounds guard: skip expansion if tree is at capacity ──────────
            # Without this check a full tree causes a CUDA illegal-address
            # error (error 700) because child_idx == max_tree_size is OOB.
            if child_idx < max_tree_size:
                trees[ti, selected, 1 + best_action] = child_idx
                trees[ti, child_idx, 0] = selected
                for a in range(max_actions):
                    if a != best_action:
                        trees[ti, selected, 1 + a] = int32(-1)

                trees_leaves[ti, child_idx] = True
                trees_terminals[ti, child_idx] = expanded_terminals[ti]
                trees_ns[ti, child_idx] = int32(0)
                trees_depths[ti, child_idx] = trees_depths[ti, selected] + int16(1)
                parent_turn = int32(trees_robot_turns[ti, selected])
                trees_robot_turns[ti, child_idx] = int8((parent_turn + 1) % num_robots)
                trees_pw_boundary[ti, child_idx] = int16(0)

                for r in range(num_robots):
                    trees_total_value[ti, child_idx, r] = float32(0.0)
                for a in range(max_actions):
                    trees_action_priors[ti, child_idx, a] = float32(0.0)
                    trees_prior_rank[ti, child_idx, a] = int16(a)

                for e in range(state_dim):
                    trees_states[ti, child_idx, e] = expanded_next_states[ti, e]

                trees_sizes[ti] = child_idx + 1
            # else: tree is at capacity — skip expansion silently
        else:
            trees_pw_boundary[ti, selected] = int16(0)

    cuda.syncthreads()

    # ── BACKUP ───────────────────────────────────────────────────────────────
    # Always backup ns even for terminal leaves (leaf_valid==0 means terminal,
    # but we still traversed a path that needs ns updated).
    if t == 0:
        path_length = int32(trees_selected_paths[ti, -1])
        leaf_depth = int32(trees_depths[ti, selected])

        for p in range(path_length):
            node = int32(trees_selected_paths[ti, p])
            node_depth = int32(trees_depths[ti, node])
            trees_ns[ti, node] += int32(1)

            # Only accumulate value if this tree had a valid (non-terminal) leaf
            if leaf_valid[ti] == int32(1):
                # Backup formula matching puct_v0.hpp / puct_v1.hpp:
                #   Each ancestor on the path gets:
                #     gamma^(leaf_depth - node_depth) * leaf_value
                #   + gamma^(leaf_depth - node_depth - 1) * step_reward   [if expanded]
                discount = math.pow(float32(gamma), float32(leaf_depth - node_depth))
                for r in range(num_robots):
                    backup_val = discount * nn_values[ti, r]
                    if expansion_valid[ti] == int32(1):
                        rel = leaf_depth - node_depth - 1
                        if rel >= 0:
                            reward_discount = math.pow(float32(gamma), float32(rel))
                        else:
                            reward_discount = float32(1.0)
                        backup_val += reward_discount * expanded_rewards[ti, r]
                    trees_total_value[ti, node, r] += backup_val

            # ── Progressive Widening update ──────────────────────────────────
            node_n = float32(trees_ns[ti, node])
            old_pw = int32(trees_pw_boundary[ti, node])
            new_pw = int32(math.ceil(C_pw * math.pow(node_n, alpha_pw)))
            if new_pw > old_pw and old_pw < max_actions:
                trees_pw_boundary[ti, node] = int16(old_pw + 1)


# ════════════════════════════════════════════════════════════════════════════
# Phase 10 — _reduce_over_trees_puct
# ════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _reduce_over_trees_puct(trees, trees_ns, trees_total_value,
                             num_robots,
                             actions_ns_out, actions_total_value_out):
    """
    Sum-reduce visit counts and total-value over all trees for a given action.

    Grid: one block per root-action index (b = blockIdx.x).
    Block: one thread per tree (t = threadIdx.x).

    Outputs
    -------
    actions_ns_out : int64[max_actions]
        Total visit count for each root action summed over all trees.
    actions_total_value_out : float32[max_actions, num_robots]
        Total value for each (root_action, robot) summed over all trees.
    """
    shared_ns = cuda.shared.array(512, dtype=int64)
    # Flat shared memory for values: 512 trees × up to 8 robots
    # We store as [thread, robot] in a 1-D array of length 512*8
    shared_values = cuda.shared.array(4096, dtype=float32)   # 512 × 8

    b = cuda.blockIdx.x    # action index
    t = cuda.threadIdx.x   # tree index
    tpb = cuda.blockDim.x
    n_trees = trees.shape[0]

    max_robots = 8   # hard limit matching shared_values layout

    if t < n_trees:
        action_node = int32(trees[t, 0, 1 + b])
        if action_node != int32(-1):
            shared_ns[t] = int64(trees_ns[t, action_node])
            for r in range(num_robots):
                shared_values[t * max_robots + r] = trees_total_value[t, action_node, r]
        else:
            shared_ns[t] = int64(0)
            for r in range(num_robots):
                shared_values[t * max_robots + r] = float32(0.0)
    else:
        shared_ns[t] = int64(0)
        for r in range(num_robots):
            shared_values[t * max_robots + r] = float32(0.0)
    cuda.syncthreads()

    # Sum reduction
    stride = tpb >> 1
    while stride > 0:
        if t < stride:
            t2 = t + stride
            shared_ns[t] += shared_ns[t2]
            for r in range(num_robots):
                shared_values[t * max_robots + r] += shared_values[t2 * max_robots + r]
        cuda.syncthreads()
        stride >>= 1

    if t == 0:
        actions_ns_out[b] = shared_ns[0]
        for r in range(num_robots):
            actions_total_value_out[b, r] = shared_values[r]


# ════════════════════════════════════════════════════════════════════════════
# Phase 10 — _reduce_over_actions_puct
# ════════════════════════════════════════════════════════════════════════════

@cuda.jit
def _reduce_over_actions_puct(actions_ns, best_action_out, best_n_out):
    """
    Find the most-visited root action (matching ``puct_v1.hpp::most_visited``).

    Grid: 1 block.
    Block: one thread per action (tpb = next-power-of-2 ≥ max_actions).

    Inputs
    ------
    actions_ns : int64[max_actions]

    Outputs
    -------
    best_action_out : int32[1]
    best_n_out : int64[1]
    """
    shared_ns = cuda.shared.array(512, dtype=int64)
    shared_action = cuda.shared.array(512, dtype=int32)

    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    max_actions = actions_ns.shape[0]

    if t < max_actions:
        shared_ns[t] = actions_ns[t]
    else:
        shared_ns[t] = int64(0)
    shared_action[t] = int32(t)
    cuda.syncthreads()

    # Max-argmax reduction
    stride = tpb >> 1
    while stride > 0:
        if t < stride:
            t2 = t + stride
            if shared_ns[t] < shared_ns[t2]:
                shared_ns[t] = shared_ns[t2]
                shared_action[t] = shared_action[t2]
        cuda.syncthreads()
        stride >>= 1

    if t == 0:
        best_action_out[0] = shared_action[0]
        best_n_out[0] = shared_ns[0]
