import math
import os

import numba.cuda as cuda
from numba import void, float32, int32

from puct.puct_gpu_v3 import (
    FULL_MASK,
    INT32_MAX,
    NEG_INF_F32,
    NODE_EXPANDED_TERMINAL,
    PACKED_EDGE_MASK,
    PACKED_KIND_MASK,
    PACKED_KIND_SHIFT,
    PACKED_NODE_LIMIT,
    PACKED_NODE_MASK,
    PACKED_SLOT_SHIFT,
    REASON_BUSY_EXPAND_INFLIGHT,
    REASON_BUSY_WINNER_RECALC,
    REASON_INVALID_CHILD_OOB,
    REASON_INVALID_NODE_INFO,
    REASON_INVALID_NO_VALID_EDGE,
    REASON_INVALID_SHAPE,
    REASON_INVALID_UNKNOWN,
    REASON_OK_DEPTH_LIMIT,
    REASON_OK_EXPAND,
    REASON_OK_TERMINAL,
    SELECT_BUSY,
    SELECT_DEPTH_LIMIT,
    SELECT_EXPAND,
    SELECT_INVALID,
    SELECT_TERMINAL,
    MAX_RECALC_RETRY,
    _allowed_children,
    _init_select_output,
    _pack_selection,
    _score_better_block_local,
    _write_select_output,
)


POS_INF_F32 = 3.4028234663852886e38
DUCT_PLAYERS = 2
DUCT_ACTION_BITS = 4
DUCT_MARGINAL_ACTIONS = 1 << DUCT_ACTION_BITS
DUCT_ACTION_MASK = DUCT_MARGINAL_ACTIONS - 1
DUCT_JOINT_ACTIONS = DUCT_MARGINAL_ACTIONS * DUCT_MARGINAL_ACTIONS
DUCT_EDGE_UNEXPANDED = -1
DUCT_EDGE_EXPANDING = -2
DUCT_PLAYER0_MASK = 0x0000FFFF
DUCT_PLAYER1_MASK = 0xFFFF0000
DUCT_EXPAND_TOPK = 4
DUCT_EXPAND_TOPK_BITS = 2
DUCT_EXPAND_TOPK_MASK = DUCT_EXPAND_TOPK - 1
DUCT_EXPAND_CANDIDATES = DUCT_EXPAND_TOPK * DUCT_EXPAND_TOPK
DUCT_EXPAND_JOB_EMPTY = 0
DUCT_EXPAND_JOB_READY = 1
DUCT_EXPAND_JOB_FAILED = -1
DUCT_SOFT_WINNER = int(os.environ.get("PUCT_DUCT_SOFT_WINNER", "0") == "1")
DUCT_SOFT_EXPAND = int(os.environ.get(
    "PUCT_DUCT_SOFT_EXPAND",
    os.environ.get("PUCT_DUCT_SOFT_WINNER", "0"),
) == "1")


# ============================================================
# DUCT data structure documentation
# ============================================================

#& DUCT新增: DUCT SELECT data layout (two-player simultaneous move)
# -----------------------------------------------------
#& 理由: 单人 PUCT 的 edge_* 统计绑定到一条边；DUCT 需要先让每个玩家
#&      独立选择自己的动作，再把动作组合成 joint move，所以需要记录
#&      “edge_slot 对应哪个 joint action”。
#
# edge_slot is the fixed joint action id:
#   edge_slot = (action_player0 << 4) | action_player1
# edge_tgt_node[tree, node, edge_slot] stores the transition target:
#   -1: no rollout/state transition has been created for this joint action.
#   -2: another warp has claimed this edge for Expand.
#  >=0: child node id.
# edge_actions[tree, node, edge_slot, player] mirrors the decoded joint action
# for Expand/backup code that consumes action tensors directly.
#
#& DUCT新增: 每个玩家独立维护动作边际统计，而不是维护 joint action 的 Q/N。
#& 理由: DUCT UPDATE 只更新本玩家被选中的动作 a_i，不更新对方动作；
#&      select 也必须按玩家维度分别套 UCB。
# action_W/action_N/action_inflight[tree, node, player, action] are per-player
# marginal statistics:
#   action_W[..., i, a] accumulates player i's return for choosing action a.
#   action_N[..., i, a] counts how often player i chose action a.
#   action_inflight[..., i, a] holds virtual visits while select traversals are
#   in flight.
#
#& DUCT新增: path 需要额外保存两个玩家的动作。
#& 理由: backup 时要对 X_{s,a_i}^i / n_{s,a_i}^i 做边际更新，仅靠
#&      单人 PUCT 的 (parent_node << 8) | edge_slot 无法知道每个玩家动作。
# out_path_actions[tree, warp, depth, player] records the per-player actions
# selected at each traversed node. Backup can use this directly to update only
# the chosen action for each player. out_path_eids keeps the resolved joint edge
# slot.


# ============================================================
# DUCT device helpers
# ============================================================

@cuda.jit(device=True, inline=True)
def _valid_duct_select_shape(max_edge_steps, node_capacity, node_cnt,
                             max_edges, action_capacity,
                             action_players, action_count_players,
                             edge_action_players, path_action_players):
    return (
        max_edge_steps >= int32(1) and
        node_capacity > int32(0) and
        node_cnt > int32(0) and
        node_cnt <= node_capacity and
        node_cnt <= int32(PACKED_NODE_LIMIT) and
        max_edges == int32(DUCT_JOINT_ACTIONS) and
        action_capacity > int32(0) and
        action_players >= int32(DUCT_PLAYERS) and
        action_count_players >= int32(DUCT_PLAYERS) and
        edge_action_players >= int32(DUCT_PLAYERS) and
        path_action_players >= int32(DUCT_PLAYERS)
    )


@cuda.jit(device=True, inline=True)
def _valid_duct_action_counts(count0, count1, action_capacity):
    return (
        count0 > int32(0) and count0 <= action_capacity and
        count0 <= int32(DUCT_MARGINAL_ACTIONS) and
        count1 > int32(0) and count1 <= action_capacity and
        count1 <= int32(DUCT_MARGINAL_ACTIONS)
    )


@cuda.jit(int32(int32, int32), device=True, inline=True)
def _duct_joint_slot(action0, action1):
    return (action0 << int32(DUCT_ACTION_BITS)) | action1


@cuda.jit(device=True, inline=True)
def _duct_parent_n_eff_local(node_N, action_inflight,
                             tree, node, action_count, lane):
    """Node visits for this lane's player, including pending visits."""
    #& DUCT重构: 两个玩家各自最多 16 个动作，用两个 half-warp 同时算各自 inflight。
    #& 理由: 每个 lane 只需要自己玩家的 PW/UCB 输入，不必持有两份玩家状态。
    player = int32(0)
    half_lane = lane
    mask = int32(DUCT_PLAYER0_MASK)

    if lane >= int32(DUCT_MARGINAL_ACTIONS):
        player = int32(1)
        half_lane = lane - int32(DUCT_MARGINAL_ACTIONS)
        mask = int32(DUCT_PLAYER1_MASK)

    inflight_total = int32(0)
    if half_lane < action_count:
        inflight_total = action_inflight[tree, node, player, half_lane]

    other = cuda.shfl_xor_sync(mask, inflight_total, 8)
    inflight_total += other
    other = cuda.shfl_xor_sync(mask, inflight_total, 4)
    inflight_total += other
    other = cuda.shfl_xor_sync(mask, inflight_total, 2)
    inflight_total += other
    other = cuda.shfl_xor_sync(mask, inflight_total, 1)
    inflight_total += other

    base_n = int32(0)
    if lane == int32(0):
        base_n = node_N[tree, node]
    base_n = cuda.shfl_sync(FULL_MASK, base_n, 0)

    return max(base_n + inflight_total, int32(1))


@cuda.jit(float32(float32, float32, int32, int32, float32), device=True, inline=True)
def _duct_ucb_score(c_uct, w, n_action, n_action_inflight, log_parent_n_eff):
    """DUCT action UCB with virtual-visit-aware counts.

    Pure DUCT treats unvisited real actions as +inf. Once a concurrent warp
    has placed a virtual visit on such an action, the effective count becomes
    finite so sibling warps can spread out instead of all chasing the same
    zero-count action.
    """
    #& DUCT新增: 未访问动作返回 +inf，已被其他 warp 虚拟占用后用有限 UCB。
    #& 理由: 符合 DUCT 对 n_{s,a}^i=0 优先探索的定义，同时保留并行去重能力。
    n_eff = n_action + n_action_inflight
    if n_eff <= int32(0):
        return float32(POS_INF_F32)

    q = float32(0.0)
    if n_action > int32(0):
        q = w / float32(n_action)

    exploration = float32(0.0)
    if log_parent_n_eff > float32(0.0):
        exploration = c_uct * math.sqrt(log_parent_n_eff / float32(n_eff))
    return q + exploration


@cuda.jit(device=True, inline=True)
def _duct_reduce_best_action_halfwarp(c_uct, action_W, action_N, action_inflight,
                                      tree, node, lane,
                                      action_count, log_parent_n_eff,
                                      tie_offset):
    #& DUCT重构: 两个玩家各自最多 16 个动作，正好用一个 warp 的两个 half-warp 并行 reduce。
    #& 理由: 旧实现用整 warp 先算 player0 再算 player1，16 动作时一半 lane 空闲且串行两遍。
    player = int32(0)
    half_lane = lane
    mask = int32(DUCT_PLAYER0_MASK)

    if lane >= int32(DUCT_MARGINAL_ACTIONS):
        player = int32(1)
        half_lane = lane - int32(DUCT_MARGINAL_ACTIONS)
        mask = int32(DUCT_PLAYER1_MASK)

    best_score = float32(NEG_INF_F32)
    best_action = int32(INT32_MAX)
    best_inflight = int32(0)

    if half_lane < action_count:
        inflight = action_inflight[tree, node, player, half_lane]
        score = _duct_ucb_score(
            c_uct,
            action_W[tree, node, player, half_lane],
            action_N[tree, node, player, half_lane],
            inflight,
            log_parent_n_eff,
        )
        if _score_better_block_local(score, int32(half_lane), best_score, best_action,
                                     tie_offset, action_count):
            best_score = score
            best_action = int32(half_lane)
            best_inflight = inflight

    other_score = cuda.shfl_xor_sync(mask, best_score, 8)
    other_action = cuda.shfl_xor_sync(mask, best_action, 8)
    other_inflight = cuda.shfl_xor_sync(mask, best_inflight, 8)
    if _score_better_block_local(other_score, other_action, best_score, best_action,
                                 tie_offset, action_count):
        best_score = other_score
        best_action = other_action
        best_inflight = other_inflight

    other_score = cuda.shfl_xor_sync(mask, best_score, 4)
    other_action = cuda.shfl_xor_sync(mask, best_action, 4)
    other_inflight = cuda.shfl_xor_sync(mask, best_inflight, 4)
    if _score_better_block_local(other_score, other_action, best_score, best_action,
                                 tie_offset, action_count):
        best_score = other_score
        best_action = other_action
        best_inflight = other_inflight

    other_score = cuda.shfl_xor_sync(mask, best_score, 2)
    other_action = cuda.shfl_xor_sync(mask, best_action, 2)
    other_inflight = cuda.shfl_xor_sync(mask, best_inflight, 2)
    if _score_better_block_local(other_score, other_action, best_score, best_action,
                                 tie_offset, action_count):
        best_score = other_score
        best_action = other_action
        best_inflight = other_inflight

    other_score = cuda.shfl_xor_sync(mask, best_score, 1)
    other_action = cuda.shfl_xor_sync(mask, best_action, 1)
    other_inflight = cuda.shfl_xor_sync(mask, best_inflight, 1)
    if _score_better_block_local(other_score, other_action, best_score, best_action,
                                 tie_offset, action_count):
        best_score = other_score
        best_action = other_action
        best_inflight = other_inflight

    return best_action, best_inflight


@cuda.jit(device=True, inline=True)
def _duct_sort_actions_halfwarp(c_uct, action_W, action_N, action_inflight,
                                tree, node, action_count, lane,
                                log_parent_n_eff, tie_offset):
    #& DUCT expand: half-warp bitonic sort。排序后 lane 0..3/16..19 分别持有两个玩家 top-4。
    #& 理由: 每个 lane 持有一个动作并并行 compare-exchange，不再把 top4 聚合到每个 lane。
    player = int32(0)
    half_lane = lane
    mask = int32(DUCT_PLAYER0_MASK)

    if lane >= int32(DUCT_MARGINAL_ACTIONS):
        player = int32(1)
        half_lane = lane - int32(DUCT_MARGINAL_ACTIONS)
        mask = int32(DUCT_PLAYER1_MASK)
        tie_offset += int32(17)
    if tie_offset >= action_count:
        tie_offset = tie_offset % action_count

    score = float32(NEG_INF_F32)
    action = int32(INT32_MAX)
    if half_lane < action_count:
        action = half_lane
        score = _duct_ucb_score(
            c_uct,
            action_W[tree, node, player, half_lane],
            action_N[tree, node, player, half_lane],
            action_inflight[tree, node, player, half_lane],
            log_parent_n_eff,
        )

    size = int32(2)
    while size <= int32(DUCT_MARGINAL_ACTIONS):
        stride = size >> int32(1)
        while stride > int32(0):
            other_score = cuda.shfl_xor_sync(mask, score, stride)
            other_action = cuda.shfl_xor_sync(mask, action, stride)
            want_desc = int32(0)
            if (half_lane & size) == int32(0):
                want_desc = int32(1)
            lower = int32(0)
            if (half_lane & stride) == int32(0):
                lower = int32(1)
            other_better = _score_better_block_local(
                other_score, other_action, score, action,
                tie_offset, action_count,
            )
            self_better = _score_better_block_local(
                score, action, other_score, other_action,
                tie_offset, action_count,
            )

            if want_desc == lower:
                if other_better:
                    score = other_score
                    action = other_action
            else:
                if self_better:
                    score = other_score
                    action = other_action

            stride >>= int32(1)
        size <<= int32(1)

    return action, score


@cuda.jit(device=True, inline=True)
def _duct_release_joint_slot(action_inflight, tree, node, joint_slot, lane):
    #& DUCT新增: 回滚/释放两个玩家各自动作上的 virtual loss。
    #& 理由: DUCT 的并发占用在 action_inflight[player, action]，不是 edge_inflight。
    if lane == int32(0) or lane == int32(DUCT_MARGINAL_ACTIONS):
        if joint_slot >= int32(0):
            action0 = joint_slot >> int32(DUCT_ACTION_BITS)
            action1 = joint_slot & int32(DUCT_ACTION_MASK)
            player = int32(lane != int32(0))
            action = action0 + (action1 - action0) * player
            cuda.atomic.sub(action_inflight, (tree, node, player, action), int32(1))


@cuda.jit(device=True, inline=True)
def _duct_add_joint_slot_inflight(action_inflight, tree, node, joint_slot, lane):
    #& DUCT soft-expand: fallback 到另一条 joint edge 后，把 virtual loss 挪到新动作上。
    if lane == int32(0) or lane == int32(DUCT_MARGINAL_ACTIONS):
        if joint_slot >= int32(0):
            action0 = joint_slot >> int32(DUCT_ACTION_BITS)
            action1 = joint_slot & int32(DUCT_ACTION_MASK)
            player = int32(lane != int32(0))
            action = action0 + (action1 - action0) * player
            cuda.atomic.add(action_inflight, (tree, node, player, action), int32(1))


@cuda.jit(device=True, inline=True)
def _duct_claim_two_actions(action_inflight, tree, node, action, inflight, lane,
                            action_count):
    #& DUCT新增: 两个玩家动作必须同时 claim 成功；任一侧失败时回滚已成功的一侧。
    #& 理由: 一个 select warp 持有的是 joint move，不能留下半个玩家动作的虚拟访问。
    held_local = int32(0)
    if lane == int32(0) or lane == int32(DUCT_MARGINAL_ACTIONS):
        player = int32(lane != int32(0))
        if action_count <= int32(1):
            cuda.atomic.add(action_inflight, (tree, node, player, action), int32(1))
            held_local = int32(1)
        else:
            prev = cuda.atomic.cas(
                action_inflight, (tree, node, player, action),
                inflight, inflight + int32(1),
            )
            if prev == inflight:
                held_local = int32(1)

    held0 = cuda.shfl_sync(FULL_MASK, held_local, 0)
    held1 = cuda.shfl_sync(FULL_MASK, held_local, DUCT_MARGINAL_ACTIONS)
    if held0 != int32(0) and held1 != int32(0):
        return int32(1)

    if lane == int32(0) or lane == int32(DUCT_MARGINAL_ACTIONS):
        if held_local != int32(0):
            player = int32(lane != int32(0))
            cuda.atomic.sub(action_inflight, (tree, node, player, action), int32(1))
    return int32(0)


@cuda.jit(device=True, inline=True)
def _duct_best_actions_winner_recalc(c_uct, action_W, action_N, action_inflight,
                                     tree, node, action_count, lane,
                                     log_parent_n_eff, tie_offset):
    #* 选择最优动作，得到对应的 action 与 inflight
    #& DUCT新增: 每个 half-warp 只持有自己玩家的 action/inflight，然后一起做 winner recalc。
    #& 理由: 保留单人 PUCT 的 CAS 重算模式，避免多个 warp 长时间挤在同一动作上。
    best_action = int32(INT32_MAX)
    pair_valid = int32(0)
    held = int32(0)
    retry = int32(0)

    local_tie_offset = tie_offset
    if lane >= int32(DUCT_MARGINAL_ACTIONS):
        local_tie_offset = tie_offset + int32(17)
    if local_tie_offset >= action_count:
        local_tie_offset = local_tie_offset % action_count

    while retry < int32(MAX_RECALC_RETRY):
        best_action, best_inflight = _duct_reduce_best_action_halfwarp(
            c_uct, action_W, action_N, action_inflight,
            tree, node, lane, action_count, log_parent_n_eff, local_tie_offset,
        )

        valid_local = int32(0)
        if best_action != int32(INT32_MAX):
            valid_local = int32(1)
        valid0 = cuda.shfl_sync(FULL_MASK, valid_local, 0)
        valid1 = cuda.shfl_sync(FULL_MASK, valid_local, DUCT_MARGINAL_ACTIONS)
        pair_valid = valid0 & valid1
        if pair_valid == int32(0):
            break

        #* 对 inflight 进行 CAS，并检查是否成功 Hold
        held = _duct_claim_two_actions(
            action_inflight, tree, node, best_action, best_inflight, lane,
            action_count,
        )
        #* 成功，或超过最大重试次数时退出
        if held == int32(1) or retry + int32(1) >= int32(MAX_RECALC_RETRY):
            break

        retry += int32(1)

    if (
        int32(DUCT_SOFT_WINNER) != int32(0) and held == int32(0) and
        pair_valid != int32(0)
    ):
        if lane == int32(0) or lane == int32(DUCT_MARGINAL_ACTIONS):
            cuda.atomic.add(
                action_inflight,
                (tree, node, int32(lane != int32(0)), best_action),
                int32(1),
            )
        held = int32(1)

    held = cuda.shfl_sync(FULL_MASK, held, 0)
    return best_action, pair_valid, held


@cuda.jit(device=True, inline=True)
def _duct_claim_joint_expand(edge_tgt_node, edge_actions, node_expand_inflight,
                             tree, node, joint_slot, lane):
    #& DUCT重构: joint action 固定映射到 0..255 的边槽，缺 child 时直接抢占这条边。
    #& 理由: 不再用 node_expanded/cur_expanded 分配线性 slot，避免线性扫描和重复联合动作。
    held = int32(0)
    if lane == int32(0):
        prev = cuda.atomic.cas(
            edge_tgt_node,
            (tree, node, joint_slot),
            int32(DUCT_EDGE_UNEXPANDED),
            int32(DUCT_EDGE_EXPANDING),
        )
        if prev == int32(DUCT_EDGE_UNEXPANDED):
            action0 = joint_slot >> int32(DUCT_ACTION_BITS)
            action1 = joint_slot & int32(DUCT_ACTION_MASK)
            edge_actions[tree, node, joint_slot, int32(0)] = action0
            edge_actions[tree, node, joint_slot, int32(1)] = action1
            cuda.atomic.add(node_expand_inflight, (tree, node), int32(1))
            held = int32(1)
    return cuda.shfl_sync(FULL_MASK, held, 0)


@cuda.jit(device=True, inline=True)
def _duct_best_top4_joint_slot(edge_tgt_node, tree, node, avoid_slot, lane,
                               start_offset,
                               sorted_action, sorted_score):
    #& DUCT soft-expand: lane 0..15 分别评估 top4(player0) x top4(player1) 的 joint 候选。
    #& 理由: expand 只需要找可建 child 的高价值 joint edge；16 个候选正好铺满 half-warp。
    best_score = float32(NEG_INF_F32)
    best_key = int32(INT32_MAX)
    best_slot = int32(-1)
    pair_key = lane & int32(DUCT_ACTION_MASK)
    row = pair_key >> int32(DUCT_EXPAND_TOPK_BITS)
    col = pair_key & int32(DUCT_EXPAND_TOPK_MASK)

    action0 = cuda.shfl_sync(FULL_MASK, sorted_action, row)
    score0 = cuda.shfl_sync(FULL_MASK, sorted_score, row)
    action1 = cuda.shfl_sync(
        FULL_MASK,
        sorted_action,
        int32(DUCT_MARGINAL_ACTIONS) + col,
    )
    score1 = cuda.shfl_sync(
        FULL_MASK,
        sorted_score,
        int32(DUCT_MARGINAL_ACTIONS) + col,
    )

    if lane < int32(DUCT_EXPAND_CANDIDATES):
        if action0 != int32(INT32_MAX) and action1 != int32(INT32_MAX):
            slot = _duct_joint_slot(action0, action1)
            if slot != avoid_slot:
                child = edge_tgt_node[tree, node, slot]
                if child == int32(DUCT_EDGE_UNEXPANDED):
                    best_score = score0 + score1
                    best_key = pair_key
                    best_slot = slot

    if lane < int32(DUCT_EXPAND_CANDIDATES):
        other_score = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_score, 8)
        other_key = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_key, 8)
        other_slot = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_slot, 8)
        if other_slot >= int32(0) and (
            best_slot < int32(0) or
            _score_better_block_local(
                other_score, other_key, best_score, best_key,
                start_offset, int32(DUCT_MARGINAL_ACTIONS),
            )
        ):
            best_score = other_score
            best_key = other_key
            best_slot = other_slot

        other_score = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_score, 4)
        other_key = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_key, 4)
        other_slot = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_slot, 4)
        if other_slot >= int32(0) and (
            best_slot < int32(0) or
            _score_better_block_local(
                other_score, other_key, best_score, best_key,
                start_offset, int32(DUCT_MARGINAL_ACTIONS),
            )
        ):
            best_score = other_score
            best_key = other_key
            best_slot = other_slot

        other_score = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_score, 2)
        other_key = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_key, 2)
        other_slot = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_slot, 2)
        if other_slot >= int32(0) and (
            best_slot < int32(0) or
            _score_better_block_local(
                other_score, other_key, best_score, best_key,
                start_offset, int32(DUCT_MARGINAL_ACTIONS),
            )
        ):
            best_score = other_score
            best_key = other_key
            best_slot = other_slot

        other_score = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_score, 1)
        other_key = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_key, 1)
        other_slot = cuda.shfl_xor_sync(DUCT_PLAYER0_MASK, best_slot, 1)
        if other_slot >= int32(0) and (
            best_slot < int32(0) or
            _score_better_block_local(
                other_score, other_key, best_score, best_key,
                start_offset, int32(DUCT_MARGINAL_ACTIONS),
            )
        ):
            best_slot = other_slot

    return cuda.shfl_sync(FULL_MASK, best_slot, 0)


@cuda.jit(device=True, inline=True)
def _duct_claim_soft_expand_fallback(edge_tgt_node, edge_actions,
                                     node_expand_inflight,
                                     tree, node, avoid_slot, lane, tie_offset,
                                     sorted_action, sorted_score):
    held = int32(0)
    fallback_slot = int32(-1)
    retry = int32(0)

    while retry < int32(MAX_RECALC_RETRY):
        start = (tie_offset + retry * int32(5)) & int32(15)
        candidate = _duct_best_top4_joint_slot(
            edge_tgt_node, tree, node, avoid_slot, lane, start,
            sorted_action, sorted_score,
        )

        if candidate < int32(0):
            break

        if lane == int32(0):
            prev = cuda.atomic.cas(
                edge_tgt_node,
                (tree, node, candidate),
                int32(DUCT_EDGE_UNEXPANDED),
                int32(DUCT_EDGE_EXPANDING),
            )
            if prev == int32(DUCT_EDGE_UNEXPANDED):
                action0 = candidate >> int32(DUCT_ACTION_BITS)
                action1 = candidate & int32(DUCT_ACTION_MASK)
                edge_actions[tree, node, candidate, int32(0)] = action0
                edge_actions[tree, node, candidate, int32(1)] = action1
                cuda.atomic.add(node_expand_inflight, (tree, node), int32(1))
                fallback_slot = candidate
                held = int32(1)

        held = cuda.shfl_sync(FULL_MASK, held, 0)
        fallback_slot = cuda.shfl_sync(FULL_MASK, fallback_slot, 0)
        if held != int32(0):
            break

        retry += int32(1)

    return fallback_slot, held


@cuda.jit(device=True, inline=True)
def _write_duct_path_step(out_path_eids, out_path_actions,
                          tree, wid, depth, lane, node, edge_slot):
    #* Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
    #& DUCT新增: 同步写 out_path_actions，保存每个玩家本次实际选择的动作。
    #& 理由: DUCT backup 只更新被选中的 a_i，因此 path 里必须有 action0/action1。
    if lane == int32(0):
        if edge_slot >= int32(0):
            out_path_eids[tree, wid, depth] = (node << int32(8)) | edge_slot
        else:
            out_path_eids[tree, wid, depth] = int32(-1)
        action0 = edge_slot >> int32(DUCT_ACTION_BITS)
        action1 = edge_slot & int32(DUCT_ACTION_MASK)
        out_path_actions[tree, wid, depth, int32(0)] = action0
        out_path_actions[tree, wid, depth, int32(1)] = action1


@cuda.jit(device=True, inline=True)
def _rollback_duct_vloss_path_nodes(tree, wid, action_depth, lane,
                                    out_path_eids, out_path_actions,
                                    action_inflight, virtual_loss,
                                    n_nodes, action_capacity):
    #& DUCT新增: path rollback 按玩家动作维度回滚 virtual loss。
    #& 理由: 单人 PUCT 回滚 edge_inflight；DUCT 没有 joint edge 级别的选择占用。
    player = int32(0)
    half_lane = lane
    if lane >= int32(DUCT_MARGINAL_ACTIONS):
        player = int32(1)
        half_lane = lane - int32(DUCT_MARGINAL_ACTIONS)

    d = half_lane
    while d < action_depth:
        encoded = out_path_eids[tree, wid, d]
        parent = encoded >> int32(8)
        if encoded < int32(0):
            parent = int32(0)
        action = out_path_actions[tree, wid, d, player]
        assert parent >= int32(0) and parent < n_nodes
        assert action >= int32(0) and action < action_capacity
        cuda.atomic.sub(action_inflight, (tree, parent, player, action), virtual_loss)
        d += int32(DUCT_MARGINAL_ACTIONS)


@cuda.jit(device=True, inline=True)
def _duct_warp_or_i32(value):
    #& Expand辅助: warp 内 int32 OR 归约。
    #& Shape: value 是每个 lane 的标量 flag；返回值会广播到整个 warp。
    #& 算法路径: shfl_xor 树形归约 -> lane0 广播，避免共享内存和 block 级同步。
    other = cuda.shfl_xor_sync(FULL_MASK, value, 16)
    value |= other
    other = cuda.shfl_xor_sync(FULL_MASK, value, 8)
    value |= other
    other = cuda.shfl_xor_sync(FULL_MASK, value, 4)
    value |= other
    other = cuda.shfl_xor_sync(FULL_MASK, value, 2)
    value |= other
    other = cuda.shfl_xor_sync(FULL_MASK, value, 1)
    value |= other
    return cuda.shfl_sync(FULL_MASK, value, 0)


@cuda.jit(device=True, inline=True)
def _duct_release_expand_job(edge_tgt_node, node_expand_inflight,
                             action_inflight, tree_nodes,
                             out_path_eids, out_path_actions, out_path_len,
                             tree, wid, parent, slot, lane,
                             node_cnt, action_capacity,
                             reclaim_tree_ticket):
    #& Expand失败路径公共回滚。
    #& Shape:
    #&   edge_tgt_node: [T, N, 256]，joint edge child id / sentinel。
    #&   node_expand_inflight: [T, N]，parent 上正在扩展的 edge 数。
    #&   action_inflight: [T, N, 2, 16]，select 阶段加上的边缘 virtual loss。
    #&   out_path_*: [T, W, D...]，本 warp 遍历路径，供回滚本次 claim。
    #& 关键路径:
    #&   1) lane0 CAS(EXPANDING -> UNEXPANDED)，只由 claim owner 释放 inflight。
    #&   2) 容量失败时可用 atomic.min 回收 node pool ticket。
    #&   3) 全 warp 按 path 并行回滚两个玩家动作维度上的 virtual loss。
    if lane == int32(0):
        prev = cuda.atomic.cas(
            edge_tgt_node,
            (tree, parent, slot),
            int32(DUCT_EDGE_EXPANDING),
            int32(DUCT_EDGE_UNEXPANDED),
        )
        if prev == int32(DUCT_EDGE_EXPANDING):
            cuda.atomic.sub(node_expand_inflight, (tree, parent), int32(1))
        if reclaim_tree_ticket != int32(0):
            cuda.atomic.min(tree_nodes, tree, node_cnt)

    action_depth = out_path_len[tree, wid]
    _rollback_duct_vloss_path_nodes(
        tree, wid, action_depth, lane,
        out_path_eids, out_path_actions,
        action_inflight, int32(1), node_cnt, action_capacity,
    )




# ============================================================
# DUCT selection kernels
# ============================================================
@cuda.jit(void(
    float32, float32, float32,
    int32[:, :, :], int32[:, :, :, :],
    float32[:, :, :, :], int32[:, :, :, :], int32[:, :, :, :],
    int32[:, :, :], int32[:, :], int32[:, :], int32[:, :], int32[:],
    int32[:, :], int32[:, :, :], int32[:, :, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_duct_winner_recalc(c_uct, c_pw, alpha_pw,
                                      edge_tgt_node, edge_actions,
                                      action_W, action_N, action_inflight,
                                      action_counts, node_N, node_expand_inflight,
                                      node_expanded, tree_nodes,
                                      out_selected_node, out_path_eids,
                                      out_path_actions, out_path_len):
    """
    双玩家同时决策的 DUCT select kernel。

    一个 CUDA block 负责一棵树，一个 warp 执行一次独立遍历。在每个非终止节点，
    两个玩家分别在自己的 PW 窗口内，用边缘 DUCT 统计选择动作：

        Q_i(s,a) + c_uct * sqrt(log(N_i(s)) / N_i(s,a))

    两个边缘动作直接组成固定的 joint edge slot：

        joint_slot = (action0 << 4) | action1        # 共 256 个 joint action

    Select 只处理这条确定的 joint edge：
        - edge_tgt_node == -1：该 joint edge 还没有 child，用 CAS(-1, -2)
          抢占 Expand 权限，并返回 SELECT_EXPAND(node, joint_slot)。
        - edge_tgt_node == -2：已有其他 warp 正在扩展这条边，回滚本次
          virtual loss 并返回 SELECT_BUSY。
        - edge_tgt_node >= 0：该边已有 child，记录路径后继续向 child 遍历。

    PW 是每个玩家独立维护的边缘动作可见窗口，不再按 joint edge 计数。
    node_expanded 只保留 terminal sentinel 和基本状态检查用途，不参与决定
    当前可访问多少 joint action。

    DUCT_SOFT_WINNER / DUCT_SOFT_EXPAND 是导入模块时确定的编译期常量。
    前者允许 winner CAS 失败后保留重复选择；后者在选中边正在扩展时，
    用两个玩家各自的 top-4 动作并行组成 16 个候选，改抢其中最优的
    未扩展 joint edge。

    关键输入：
        edge_tgt_node: [tree, node, 256]，child node id；-1 未扩展，-2 正在扩展。
        edge_actions: [tree, node, 256, player]，缓存 joint slot 解码后的两个动作。
        action_W/N/inflight: [tree, node, player, action]，每个玩家的边缘 DUCT 统计。
        action_counts: [tree, node, player]，每个玩家的合法边缘动作数，最大 16。
        node_N: 节点访问次数，用于 UCB/PW。
        node_expand_inflight: 当前节点已抢占但尚未完成的 edge Expand 数。

    输出：
        out_selected_node: 打包后的 SELECT_* 结果，以及被选中的 node/joint slot。
        out_path_eids: 路径边，编码为 (parent_node << 8) | joint_slot。
        out_path_actions: 每条路径边对应的 action0/action1。
        out_path_len: 路径长度或评估叶子的长度。
    """
    tree = cuda.blockIdx.x
    lane = cuda.threadIdx.x & int32(31)
    wid = cuda.threadIdx.x >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    if wid >= min(cuda.blockDim.x >> int32(5), out_selected_node.shape[1]):
        return

    _init_select_output(tree, wid, lane, out_selected_node, out_path_len)

    max_edges = edge_tgt_node.shape[2]
    max_edge_steps = out_path_eids.shape[2]
    node_capacity = node_expanded.shape[1]
    action_capacity = action_W.shape[3]
    action_players = action_W.shape[2]
    action_count_players = action_counts.shape[2]
    edge_action_players = edge_actions.shape[3]
    path_action_players = out_path_actions.shape[3]

    node_cnt = int32(0)
    if lane == int32(0):
        node_cnt = tree_nodes[tree]
    node_cnt = cuda.shfl_sync(FULL_MASK, node_cnt, 0)

    if not _valid_duct_select_shape(
        max_edge_steps, node_capacity, node_cnt, max_edges,
        action_capacity, action_players, action_count_players,
        edge_action_players, path_action_players,
    ):
        if lane == int32(0):
            out_selected_node[tree, wid] = _pack_selection(
                int32(SELECT_INVALID),
                int32(PACKED_NODE_MASK),
                int32(0),
                int32(REASON_INVALID_SHAPE),
            )
        return

    final_packed = _pack_selection(
        int32(SELECT_INVALID),
        int32(PACKED_NODE_MASK),
        int32(0),
        int32(REASON_INVALID_UNKNOWN),
    )
    final_len = int32(0)
    node = int32(0)
    depth = int32(0)

    while True:
        node_info = node_expanded[tree, node]
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_packed = _pack_selection(
                int32(SELECT_TERMINAL),
                node,
                int32(0),
                int32(REASON_OK_TERMINAL),
            )
            final_len = depth + int32(1)
            break

        #* DUCT 不再用 node_expanded 记录扩展计数；这里仅保留非法 sentinel 的防御检查。
        if node_info < int32(0) or node_info > max_edges:
            _rollback_duct_vloss_path_nodes(
                tree, wid, depth, lane,
                out_path_eids, out_path_actions, action_inflight,
                int32(1), node_cnt, action_capacity,
            )
            cuda.syncwarp(FULL_MASK)
            final_packed = _pack_selection(
                int32(SELECT_INVALID),
                int32(PACKED_NODE_MASK),
                int32(0),
                int32(REASON_INVALID_NODE_INFO),
            )
            break

        #* 超过最大 rollout 长度，直接截断；后续可在该节点重新估计价值。
        if depth >= max_edge_steps:
            final_packed = _pack_selection(
                int32(SELECT_DEPTH_LIMIT),
                node,
                int32(0),
                int32(REASON_OK_DEPTH_LIMIT),
            )
            final_len = depth + int32(1)
            break

        #& DUCT新增: 每个节点保存两个玩家各自的合法动作数。
        #& 理由: A_0(s) 和 A_1(s) 可以不同，select 时必须分别约束动作范围。
        action_count0 = action_counts[tree, node, int32(0)]
        action_count1 = action_counts[tree, node, int32(1)]
        if not _valid_duct_action_counts(action_count0, action_count1, action_capacity):
            _rollback_duct_vloss_path_nodes(
                tree, wid, depth, lane,
                out_path_eids, out_path_actions, action_inflight,
                int32(1), node_cnt, action_capacity,
            )
            cuda.syncwarp(FULL_MASK)
            final_packed = _pack_selection(
                int32(SELECT_INVALID),
                int32(PACKED_NODE_MASK),
                int32(0),
                int32(REASON_INVALID_NODE_INFO),
            )
            break


        #* 每个玩家各自执行 PW，限制当前可访问的边缘动作前缀。
        local_action_count = action_count0
        if lane >= int32(DUCT_MARGINAL_ACTIONS):
            local_action_count = action_count1
        parent_n_eff = _duct_parent_n_eff_local(
            node_N, action_inflight, tree, node, local_action_count, lane,
        )
        allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, local_action_count)
        log_parent_n_eff = math.log(float32(parent_n_eff))
        tie_offset = wid

        #& DUCT新增: 每个玩家独立用 DUCT UCB 选择动作，再组合成 joint move。这是 DUCT 和单人 PUCT select 的核心差异；
        local_best_action, pair_valid, claim_held = _duct_best_actions_winner_recalc(
            c_uct, action_W, action_N, action_inflight,
            tree, node, allowed, lane, log_parent_n_eff, tie_offset,
        )

        #* 当未取得 held 机会时直接回滚；拿到 held 后再校验 child 合法性。
        if claim_held == int32(0):
            cuda.syncwarp(FULL_MASK)
            _rollback_duct_vloss_path_nodes(
                tree, wid, depth, lane,
                out_path_eids, out_path_actions, action_inflight,
                int32(1), node_cnt, action_capacity,
            )
            cuda.syncwarp(FULL_MASK)
            if pair_valid == int32(0):
                final_packed = _pack_selection(
                    int32(SELECT_INVALID),
                    int32(PACKED_NODE_MASK),
                    int32(0),
                    int32(REASON_INVALID_NO_VALID_EDGE),
                )
            else:
                final_packed = _pack_selection(
                    int32(SELECT_BUSY),
                    node,
                    int32(0),
                    int32(REASON_BUSY_WINNER_RECALC),
                )
            break

        #& DUCT重构: joint action 数量固定为 256，两份 4-bit 边缘动作直接组成 slot。
        #& 理由: Select 已经选出两个玩家动作，定位边不需要扫描 node_expanded。
        joint_slot = int32(0)
        action1 = cuda.shfl_sync(FULL_MASK, local_best_action, DUCT_MARGINAL_ACTIONS)
        if lane == int32(0):
            joint_slot = _duct_joint_slot(local_best_action, action1)
        joint_slot = cuda.shfl_sync(FULL_MASK, joint_slot, 0)
        joint_child = edge_tgt_node[tree, node, joint_slot]
        joint_child = cuda.shfl_sync(FULL_MASK, joint_child, 0)

        expand_busy = int32(0)
        if joint_child == int32(DUCT_EDGE_UNEXPANDED):
            expand_held = _duct_claim_joint_expand(
                edge_tgt_node, edge_actions, node_expand_inflight,
                tree, node, joint_slot, lane,
            )
            if expand_held == int32(1):
                _write_duct_path_step(
                    out_path_eids, out_path_actions,
                    tree, wid, depth, lane, node, joint_slot,
                )
                final_packed = _pack_selection(
                    int32(SELECT_EXPAND),
                    node,
                    joint_slot,
                    int32(REASON_OK_EXPAND),
                )
                final_len = depth + int32(1)
                break

            expand_busy = int32(1)

        #* 其他 warp 已选择将其扩展
        if joint_child == int32(DUCT_EDGE_EXPANDING):
            expand_busy = int32(1)

        if expand_busy != int32(0) or joint_child < int32(0) or joint_child >= node_cnt:
            if (
                int32(DUCT_SOFT_EXPAND) != int32(0) and
                expand_busy != int32(0)
            ):
                sorted_action, sorted_score = _duct_sort_actions_halfwarp(
                    c_uct, action_W, action_N, action_inflight,
                    tree, node, allowed, lane, log_parent_n_eff, tie_offset,
                )
                fallback_slot, fallback_held = _duct_claim_soft_expand_fallback(
                    edge_tgt_node, edge_actions, node_expand_inflight,
                    tree, node, joint_slot, lane, wid,
                    sorted_action, sorted_score,
                )
                if fallback_held != int32(0):
                    _duct_release_joint_slot(
                        action_inflight, tree, node, joint_slot, lane,
                    )
                    _duct_add_joint_slot_inflight(
                        action_inflight, tree, node, fallback_slot, lane,
                    )
                    cuda.syncwarp(FULL_MASK)
                    _write_duct_path_step(
                        out_path_eids, out_path_actions,
                        tree, wid, depth, lane, node, fallback_slot,
                    )
                    final_packed = _pack_selection(
                        int32(SELECT_EXPAND),
                        node,
                        fallback_slot,
                        int32(REASON_OK_EXPAND),
                    )
                    final_len = depth + int32(1)
                    break

            _duct_release_joint_slot(
                action_inflight, tree, node, joint_slot, lane,
            )
            cuda.syncwarp(FULL_MASK)
            _rollback_duct_vloss_path_nodes(
                tree, wid, depth, lane,
                out_path_eids, out_path_actions, action_inflight,
                int32(1), node_cnt, action_capacity,
            )
            cuda.syncwarp(FULL_MASK)
            if expand_busy != int32(0):
                final_packed = _pack_selection(
                    int32(SELECT_BUSY),
                    node,
                    int32(0),
                    int32(REASON_BUSY_EXPAND_INFLIGHT),
                )
            else:
                final_packed = _pack_selection(
                    int32(SELECT_INVALID),
                    int32(PACKED_NODE_MASK),
                    int32(0),
                    int32(REASON_INVALID_CHILD_OOB),
                )
            final_len = int32(0)
            break

        #* Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
        #& DUCT新增: 这里额外记录 action0/action1，供 DUCT backup 更新每个玩家的动作统计。
        _write_duct_path_step(
            out_path_eids, out_path_actions,
            tree, wid, depth, lane, node, joint_slot,
        )
        cuda.syncwarp(FULL_MASK)

        node = joint_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_packed, final_len,
                         out_selected_node, out_path_len)


# ============================================================
# DUCT expand kernels
# ============================================================

@cuda.jit
def _prepare_expand_stage1_duct(edge_tgt_node,
                                node_expand_inflight,
                                action_inflight,
                                tree_nodes,
                                out_selected_node,
                                out_path_eids,
                                out_path_actions,
                                out_path_len,
                                node_states,
                                node_action_ids,
                                node_action_targets,
                                node_action_probs,
                                expand_valid,
                                expand_tree,
                                expand_parent,
                                expand_slot,
                                expand_child,
                                expand_parent_states,
                                expand_parent_targets,
                                expand_parent_action_ids,
                                expand_parent_action_probs):
    """DUCT 扩展阶段1：把 select 的 SELECT_EXPAND 输出转成 PyTorch job。

    并行粒度：
        blockIdx.x = tree，一个 block 处理一棵树。
        一个 warp 对应一次独立遍历输出 wid；lane 在线性 shape 上 stride 拷贝。

    关键输入 shape：
        edge_tgt_node: [T, N, 256]，joint edge 状态；-1 未扩展，-2 正在扩展。
        out_selected_node: [T, W]，select 输出的 packed SELECT_EXPAND(parent, slot)。
        out_path_eids/actions/len: [T, W, D...]，失败时回滚本次 virtual loss。
        node_states: [T, N, state_dim]，parent state 源数据。
        node_action_ids: [T, N, 2, 16]，每个玩家本地 slot 对应的候选 action id。
        node_action_targets: [T, N, 2, 16, A, 2]，slot 对应 MINCO target。
        node_action_probs: [T, N, 2, 16]，策略/均匀混合后的 slot 概率。

    输出 staging shape：
        expand_valid/tree/parent/slot/child: [T, W]。
        expand_parent_states: [T, W, state_dim]。
        expand_parent_targets: [T, W, 2, A, 2]，只导出被选中的两个玩家 slot target。
        expand_parent_action_ids/probs: [T, W, 2]。

    关键算法路径：
        1) 解码 SELECT_EXPAND(parent, joint_slot)，并确认 edge 仍为 EXPANDING。
        2) lane0 从 tree_nodes[T] 原子分配 child id，容量失败则重置 edge 并回滚。
        3) warp 并行导出 parent state 和两个玩家被选中的 slot target。
        4) 写完 metadata 后，最后发布 expand_valid=READY，交给 PyTorch stage2。

    发布约束：
        stage1 只分配 child/staging，不发布 edge_tgt_node[parent, slot]=child。
        child 对其他 select warp 的可见性由 stage2 commit 的 threadfence 保证。
    """
    tree = cuda.blockIdx.x
    lane = cuda.threadIdx.x & int32(31)
    wid = cuda.threadIdx.x >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return
    if wid >= min(cuda.blockDim.x >> int32(5), out_selected_node.shape[1]):
        return

    #& 先清空本 warp job header，避免上一轮残留被 Python bridge 误读。
    if lane == int32(0):
        expand_valid[tree, wid] = int32(DUCT_EXPAND_JOB_EMPTY)
        expand_tree[tree, wid] = int32(-1)
        expand_parent[tree, wid] = int32(-1)
        expand_slot[tree, wid] = int32(-1)
        expand_child[tree, wid] = int32(-1)

    raw = out_selected_node[tree, wid]
    kind = (raw >> int32(PACKED_KIND_SHIFT)) & int32(PACKED_KIND_MASK)
    if kind != int32(SELECT_EXPAND):
        return

    #& 解码 select packed 输出；slot 是 8-bit joint action: (a0 << 4) | a1。
    parent = raw & int32(PACKED_NODE_MASK)
    slot = (raw >> int32(PACKED_SLOT_SHIFT)) & int32(PACKED_EDGE_MASK)
    node_capacity = node_states.shape[1]
    action_capacity = action_inflight.shape[3]

    node_cnt = int32(0)
    if lane == int32(0):
        node_cnt = tree_nodes[tree]
    node_cnt = cuda.shfl_sync(FULL_MASK, node_cnt, 0)

    if (
        parent < int32(0) or parent >= node_cnt or
        parent >= edge_tgt_node.shape[1] or
        slot < int32(0) or slot >= int32(DUCT_JOINT_ACTIONS)
    ):
        return

    #& 二次验证 edge ownership：select 抢到的 EXPANDING 可能已被其他路径失败释放。
    child_state = int32(DUCT_EDGE_UNEXPANDED)
    if lane == int32(0):
        child_state = edge_tgt_node[tree, parent, slot]
    child_state = cuda.shfl_sync(FULL_MASK, child_state, 0)
    if child_state != int32(DUCT_EDGE_EXPANDING):
        return

    #& node pool 分配只由 lane0 执行；child id 经 shfl 广播给整个 warp。
    child = int32(-1)
    if lane == int32(0):
        child = cuda.atomic.add(tree_nodes, tree, int32(1))
    child = cuda.shfl_sync(FULL_MASK, child, 0)

    if child < int32(0) or child >= node_capacity or child >= int32(PACKED_NODE_LIMIT):
        alloc_limit = node_capacity
        if alloc_limit > int32(PACKED_NODE_LIMIT):
            alloc_limit = int32(PACKED_NODE_LIMIT)
        #& 容量/OOB 失败: 不发布 child，释放 edge claim，并回滚本 warp virtual loss。
        _duct_release_expand_job(
            edge_tgt_node, node_expand_inflight, action_inflight, tree_nodes,
            out_path_eids, out_path_actions, out_path_len,
            tree, wid, parent, slot, lane,
            alloc_limit, action_capacity, int32(1),
        )
        if lane == int32(0):
            expand_valid[tree, wid] = int32(DUCT_EXPAND_JOB_EMPTY)
        return

    state_dim = node_states.shape[2]
    #& Shape拷贝: parent state [state_dim] -> expand_parent_states[tree, wid, :].
    idx = lane
    while idx < state_dim:
        expand_parent_states[tree, wid, idx] = node_states[tree, parent, idx]
        idx += int32(32)

    action0 = slot >> int32(DUCT_ACTION_BITS)
    action1 = slot & int32(DUCT_ACTION_MASK)
    num_agents = node_action_targets.shape[4]

    #& Shape拷贝: 两个玩家选中的 marginal slot target
    #&   node_action_targets[tree,parent,player,action,A,2]
    #&   -> expand_parent_targets[tree,wid,player,A,2].
    idx = lane
    while idx < int32(DUCT_PLAYERS) * num_agents * int32(2):
        player = idx // (num_agents * int32(2))
        rem = idx - player * num_agents * int32(2)
        agent = rem >> int32(1)
        dim = rem & int32(1)
        action = action0
        if player == int32(1):
            action = action1
        expand_parent_targets[tree, wid, player, agent, dim] = (
            node_action_targets[tree, parent, player, action, agent, dim]
        )
        idx += int32(32)

    #& lane0 写 job metadata；READY 必须最后写，Python 只扫描 READY job。
    if lane == int32(0):
        expand_parent_action_ids[tree, wid, int32(0)] = (
            node_action_ids[tree, parent, int32(0), action0]
        )
        expand_parent_action_ids[tree, wid, int32(1)] = (
            node_action_ids[tree, parent, int32(1), action1]
        )
        expand_parent_action_probs[tree, wid, int32(0)] = (
            node_action_probs[tree, parent, int32(0), action0]
        )
        expand_parent_action_probs[tree, wid, int32(1)] = (
            node_action_probs[tree, parent, int32(1), action1]
        )
        expand_tree[tree, wid] = tree
        expand_parent[tree, wid] = parent
        expand_slot[tree, wid] = slot
        expand_child[tree, wid] = child

    cuda.syncwarp(FULL_MASK)
    if lane == int32(0):
        #& 发布 stage1 job。此时 parent state/targets/metadata 已对本 warp 写完。
        expand_valid[tree, wid] = int32(DUCT_EXPAND_JOB_READY)


@cuda.jit
def _commit_expand_stage2_duct(edge_tgt_node,
                               edge_actions,
                               action_W,
                               action_N,
                               action_inflight,
                               action_counts,
                               node_N,
                               node_expand_inflight,
                               node_expanded,
                               tree_nodes,
                               out_path_eids,
                               out_path_actions,
                               out_path_len,
                               node_states,
                               node_action_ids,
                               node_action_targets,
                               node_action_probs,
                               expand_valid,
                               expand_parent,
                               expand_slot,
                               expand_child,
                               expand_next_states,
                               expand_done,
                               expand_child_action_ids,
                               expand_child_action_targets,
                               expand_child_action_probs):
    """DUCT 扩展阶段2：提交 PyTorch/MINCO 结果并发布 child edge。

    并行粒度：
        blockIdx.x = tree，一个 warp 对应一个 stage1 job wid。
        lane 在线性 shape 上 stride 初始化 child 节点和 slot 表。

    输入 staging shape：
        expand_valid: [T, W]，READY 表示 Python 已填好 stage2 buffer。
        expand_parent/slot/child: [T, W]，stage1 分配和声明的 tree metadata。
        expand_next_states: [T, W, state_dim]，MINCO transition 后的 child state。
        expand_done: [T, W]，terminal 标记。
        expand_child_action_ids/probs: [T, W, 2, 16]，新 child 的本地候选 slot。
        expand_child_action_targets: [T, W, 2, 16, A, 2]，slot 对应连续 target。

    树内目标 shape：
        node_states: [T, N, state_dim]。
        edge_tgt_node/edge_actions: [T, N, 256] / [T, N, 256, 2]。
        action_W/N/inflight: [T, N, 2, 16]。
        action_counts: [T, N, 2]。

    关键算法路径：
        1) 校验 READY job、parent/child/slot 范围、edge 仍为 EXPANDING、next_state 有限。
        2) 失败 job 重置 edge，并回滚 select 加上的 virtual loss。
        3) 成功 job 先写 child state，再初始化 256 个 joint edge 和 2x16 边缘动作统计。
        4) 拷贝 PyTorch 采样出的 child slot ids/targets/probs。
        5) 设置 node_N/node_expanded/action_counts，threadfence 后发布 edge_tgt_node=child。
        6) 发布后释放 node_expand_inflight，并清空 expand_valid。

    发布约束：
        edge_tgt_node[parent, slot] 是最后的树可见发布点；发布前 child 的统计、
        action slot 和 terminal sentinel 必须已全部写入，避免 select 读到半初始化节点。
    """
    tree = cuda.blockIdx.x
    lane = cuda.threadIdx.x & int32(31)
    wid = cuda.threadIdx.x >> int32(5)

    if tree >= expand_valid.shape[0]:
        return
    if wid >= min(cuda.blockDim.x >> int32(5), expand_valid.shape[1]):
        return

    status = expand_valid[tree, wid]
    if status == int32(DUCT_EXPAND_JOB_EMPTY):
        return

    parent = expand_parent[tree, wid]
    slot = expand_slot[tree, wid]
    child = expand_child[tree, wid]
    node_capacity = node_states.shape[1]
    action_capacity = action_inflight.shape[3]
    state_dim = node_states.shape[2]
    num_agents = node_action_targets.shape[4]

    #& 读取当前 node pool 上界，用于检查 child ticket 是否仍在有效范围内。
    node_cnt = int32(0)
    if lane == int32(0):
        node_cnt = tree_nodes[tree]
    node_cnt = cuda.shfl_sync(FULL_MASK, node_cnt, 0)

    invalid = int32(0)
    if (
        parent < int32(0) or parent >= node_cnt or
        child < int32(0) or child >= node_capacity or
        child >= node_cnt or
        slot < int32(0) or slot >= int32(DUCT_JOINT_ACTIONS) or
        status != int32(DUCT_EXPAND_JOB_READY)
    ):
        invalid = int32(1)

    edge_state = int32(DUCT_EDGE_UNEXPANDED)
    if lane == int32(0) and parent >= int32(0) and parent < edge_tgt_node.shape[1]:
        if slot >= int32(0) and slot < int32(DUCT_JOINT_ACTIONS):
            edge_state = edge_tgt_node[tree, parent, slot]
    edge_state = cuda.shfl_sync(FULL_MASK, edge_state, 0)
    if edge_state != int32(DUCT_EDGE_EXPANDING):
        invalid = int32(1)

    #& finite guard: PyTorch/MINCO 若产生 NaN/Inf，commit 失败并走 rollback。
    idx = lane
    while idx < state_dim:
        v = expand_next_states[tree, wid, idx]
        if not (v == v and abs(v) <= float32(POS_INF_F32)):
            invalid = int32(1)
        idx += int32(32)
    invalid = _duct_warp_or_i32(invalid)

    if invalid != int32(0):
        #& 失败路径只释放仍然由本 job 持有的 EXPANDING edge。
        if (
            edge_state == int32(DUCT_EDGE_EXPANDING) and
            parent >= int32(0) and parent < edge_tgt_node.shape[1]
        ):
            _duct_release_expand_job(
                edge_tgt_node, node_expand_inflight, action_inflight, tree_nodes,
                out_path_eids, out_path_actions, out_path_len,
                tree, wid, parent, slot, lane,
                node_cnt, action_capacity, int32(0),
            )
        if lane == int32(0):
            expand_valid[tree, wid] = int32(DUCT_EXPAND_JOB_EMPTY)
        return

    #& Shape拷贝: expand_next_states[tree,wid,state_dim] -> node_states[tree,child,:].
    idx = lane
    while idx < state_dim:
        node_states[tree, child, idx] = expand_next_states[tree, wid, idx]
        idx += int32(32)

    #& 初始化 child 的固定 16x16 joint edge 表。edge_actions 延迟到 select 命中时也可解码，
    #& 这里先置 -1，避免调试/断言看到旧值。
    idx = lane
    while idx < int32(DUCT_JOINT_ACTIONS):
        edge_tgt_node[tree, child, idx] = int32(DUCT_EDGE_UNEXPANDED)
        edge_actions[tree, child, idx, int32(0)] = int32(-1)
        edge_actions[tree, child, idx, int32(1)] = int32(-1)
        idx += int32(32)

    #& 初始化两个玩家的 16 个 marginal slot 统计，并写入策略采样后的 action id/prob。
    idx = lane
    while idx < int32(DUCT_PLAYERS) * int32(DUCT_MARGINAL_ACTIONS):
        player = idx // int32(DUCT_MARGINAL_ACTIONS)
        action = idx - player * int32(DUCT_MARGINAL_ACTIONS)
        action_W[tree, child, player, action] = float32(0.0)
        action_N[tree, child, player, action] = int32(0)
        action_inflight[tree, child, player, action] = int32(0)
        node_action_ids[tree, child, player, action] = (
            expand_child_action_ids[tree, wid, player, action]
        )
        node_action_probs[tree, child, player, action] = (
            expand_child_action_probs[tree, wid, player, action]
        )
        idx += int32(32)

    #& Shape拷贝: expand_child_action_targets[T,W,2,16,A,2]
    #&   -> node_action_targets[T,child,2,16,A,2].
    idx = lane
    total_target = int32(DUCT_PLAYERS) * int32(DUCT_MARGINAL_ACTIONS) * num_agents * int32(2)
    while idx < total_target:
        player = idx // (int32(DUCT_MARGINAL_ACTIONS) * num_agents * int32(2))
        rem0 = idx - player * int32(DUCT_MARGINAL_ACTIONS) * num_agents * int32(2)
        action = rem0 // (num_agents * int32(2))
        rem1 = rem0 - action * num_agents * int32(2)
        agent = rem1 >> int32(1)
        dim = rem1 & int32(1)
        node_action_targets[tree, child, player, action, agent, dim] = (
            expand_child_action_targets[tree, wid, player, action, agent, dim]
        )
        idx += int32(32)

    #& lane0 写节点级状态。terminal child 设置防御性 action_count=1，
    #& select 会通过 NODE_EXPANDED_TERMINAL 跳过继续扩展。
    if lane == int32(0):
        node_N[tree, child] = int32(0)
        if expand_done[tree, wid] != int32(0):
            node_expanded[tree, child] = int32(NODE_EXPANDED_TERMINAL)
            action_counts[tree, child, int32(0)] = int32(1)
            action_counts[tree, child, int32(1)] = int32(1)
        else:
            node_expanded[tree, child] = int32(0)
            action_counts[tree, child, int32(0)] = int32(DUCT_MARGINAL_ACTIONS)
            action_counts[tree, child, int32(1)] = int32(DUCT_MARGINAL_ACTIONS)

    #& 发布屏障: child 数据全部写完后，再让 parent edge 指向 child。
    cuda.syncwarp(FULL_MASK)
    cuda.threadfence()

    if lane == int32(0):
        #& 树可见发布点。之后 select 可能立刻沿 parent/slot 进入 child。
        edge_tgt_node[tree, parent, slot] = child
        cuda.atomic.sub(node_expand_inflight, (tree, parent), int32(1))
        expand_valid[tree, wid] = int32(DUCT_EXPAND_JOB_EMPTY)


# ============================================================
# DUCT PyTorch bridge helpers
# ============================================================

class DuctExpandBridge:
    """DUCT expand 的 PyTorch/Numba 共享缓冲区。

    树内持久 storage shape：
        node_states: [T, N, state_dim]。
        node_action_ids: [T, N, 2, 16]。
        node_action_targets: [T, N, 2, 16, A, 2]。
        node_action_probs: [T, N, 2, 16]。

    job staging shape：
        expand_valid/tree/parent/slot/child: [T, W]。
        expand_parent_states: [T, W, state_dim]。
        expand_parent_targets: [T, W, 2, A, 2]。
        expand_next_states: [T, W, state_dim]。
        expand_child_action_ids/probs: [T, W, 2, 16]。
        expand_child_action_targets: [T, W, 2, 16, A, 2]。

    关键路径：
        CUDA stage1 写 parent/slot/child 和 parent target staging；
        Python 调 MINCO projection/transition 和 policy sampling；
        CUDA stage2 把 next_state/slot 表提交回 tree 并发布 edge。

    注意：
        PyTorch tensor 是 owner；cuda.as_cuda_array 只是零拷贝 view，
        因此 bridge 对象必须在 kernel 使用期间保持存活。
    """

    def __init__(self, n_trees: int, node_capacity: int, state_dim: int,
                 num_agents: int, max_warps: int, device: str = "cuda"):
        import torch

        self.n_trees = int(n_trees)
        self.node_capacity = int(node_capacity)
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.max_warps = int(max_warps)
        self.device = torch.device(device)

        #& 树内持久 storage：child commit 后 select/backup 都从这些 tensor 读取。
        self.node_states = torch.zeros(
            (self.n_trees, self.node_capacity, self.state_dim),
            dtype=torch.float32, device=self.device,
        )
        self.node_action_ids = torch.full(
            (self.n_trees, self.node_capacity, DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS),
            -1, dtype=torch.int32, device=self.device,
        )
        self.node_action_targets = torch.zeros(
            (self.n_trees, self.node_capacity, DUCT_PLAYERS,
             DUCT_MARGINAL_ACTIONS, self.num_agents, 2),
            dtype=torch.float32, device=self.device,
        )
        self.node_action_probs = torch.zeros(
            (self.n_trees, self.node_capacity, DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS),
            dtype=torch.float32, device=self.device,
        )

        #& stage1 -> Python -> stage2 的 job staging。
        #& expand_valid 是唯一的 job 状态发布字段：EMPTY / READY / FAILED。
        self.expand_valid = torch.zeros(
            (self.n_trees, self.max_warps), dtype=torch.int32, device=self.device,
        )
        self.expand_tree = torch.full_like(self.expand_valid, -1)
        self.expand_parent = torch.full_like(self.expand_valid, -1)
        self.expand_slot = torch.full_like(self.expand_valid, -1)
        self.expand_child = torch.full_like(self.expand_valid, -1)
        self.expand_parent_states = torch.zeros(
            (self.n_trees, self.max_warps, self.state_dim),
            dtype=torch.float32, device=self.device,
        )
        self.expand_parent_targets = torch.zeros(
            (self.n_trees, self.max_warps, DUCT_PLAYERS, self.num_agents, 2),
            dtype=torch.float32, device=self.device,
        )
        self.expand_parent_action_ids = torch.full(
            (self.n_trees, self.max_warps, DUCT_PLAYERS),
            -1, dtype=torch.int32, device=self.device,
        )
        self.expand_parent_action_probs = torch.zeros(
            (self.n_trees, self.max_warps, DUCT_PLAYERS),
            dtype=torch.float32, device=self.device,
        )
        self.expand_next_states = torch.zeros(
            (self.n_trees, self.max_warps, self.state_dim),
            dtype=torch.float32, device=self.device,
        )
        self.expand_done = torch.zeros(
            (self.n_trees, self.max_warps), dtype=torch.int32, device=self.device,
        )
        self.expand_child_action_ids = torch.full(
            (self.n_trees, self.max_warps, DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS),
            -1, dtype=torch.int32, device=self.device,
        )
        self.expand_child_action_targets = torch.zeros(
            (self.n_trees, self.max_warps, DUCT_PLAYERS,
             DUCT_MARGINAL_ACTIONS, self.num_agents, 2),
            dtype=torch.float32, device=self.device,
        )
        self.expand_child_action_probs = torch.zeros(
            (self.n_trees, self.max_warps, DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS),
            dtype=torch.float32, device=self.device,
        )

        self._refresh_dev_views()

    def _refresh_dev_views(self) -> None:
        """刷新 Numba CUDA view；shape 与同名 PyTorch tensor 完全一致。"""
        self.dev_node_states = cuda.as_cuda_array(self.node_states)
        self.dev_node_action_ids = cuda.as_cuda_array(self.node_action_ids)
        self.dev_node_action_targets = cuda.as_cuda_array(self.node_action_targets)
        self.dev_node_action_probs = cuda.as_cuda_array(self.node_action_probs)
        self.dev_expand_valid = cuda.as_cuda_array(self.expand_valid)
        self.dev_expand_tree = cuda.as_cuda_array(self.expand_tree)
        self.dev_expand_parent = cuda.as_cuda_array(self.expand_parent)
        self.dev_expand_slot = cuda.as_cuda_array(self.expand_slot)
        self.dev_expand_child = cuda.as_cuda_array(self.expand_child)
        self.dev_expand_parent_states = cuda.as_cuda_array(self.expand_parent_states)
        self.dev_expand_parent_targets = cuda.as_cuda_array(self.expand_parent_targets)
        self.dev_expand_parent_action_ids = cuda.as_cuda_array(self.expand_parent_action_ids)
        self.dev_expand_parent_action_probs = cuda.as_cuda_array(self.expand_parent_action_probs)
        self.dev_expand_next_states = cuda.as_cuda_array(self.expand_next_states)
        self.dev_expand_done = cuda.as_cuda_array(self.expand_done)
        self.dev_expand_child_action_ids = cuda.as_cuda_array(self.expand_child_action_ids)
        self.dev_expand_child_action_targets = cuda.as_cuda_array(self.expand_child_action_targets)
        self.dev_expand_child_action_probs = cuda.as_cuda_array(self.expand_child_action_probs)

    def reset_jobs(self) -> None:
        """清空 staging job，不改树内 node_* storage。"""
        #& 用于一轮 expand 前/后清理。tree storage 保留，避免误删已提交 child。
        self.expand_valid.zero_()
        self.expand_tree.fill_(-1)
        self.expand_parent.fill_(-1)
        self.expand_slot.fill_(-1)
        self.expand_child.fill_(-1)
        self.expand_parent_states.zero_()
        self.expand_parent_targets.zero_()
        self.expand_parent_action_ids.fill_(-1)
        self.expand_parent_action_probs.zero_()
        self.expand_next_states.zero_()
        self.expand_done.zero_()
        self.expand_child_action_ids.fill_(-1)
        self.expand_child_action_targets.zero_()
        self.expand_child_action_probs.zero_()


def _as_cuda_float_tensor(value, *, device):
    """把输入规范化为 CUDA float32 contiguous tensor。"""
    import torch

    return torch.as_tensor(value, dtype=torch.float32, device=device).contiguous()


def _duct_policy_probs(logits, uniform_sample_prob: float):
    """把策略 logits 转成 DUCT marginal slot 采样分布。

    Shape:
        logits: [B, 16]，每行对应一个节点、一个玩家的候选 action logits。
        return: [B, 16]，已归一化的概率分布。

    关键路径：
        softmax -> NaN/Inf guard -> 坏行退回 uniform ->
        mixed = epsilon * uniform + (1 - epsilon) * policy。
    """
    import torch

    if logits.ndim != 2 or logits.shape[1] != DUCT_MARGINAL_ACTIONS:
        raise ValueError(
            f"policy logits must have shape (batch, {DUCT_MARGINAL_ACTIONS}), "
            f"got {tuple(logits.shape)}"
        )
    eps = float(uniform_sample_prob)
    if eps < 0.0 or eps > 1.0:
        raise ValueError(f"uniform_sample_prob must be in [0, 1], got {eps}")

    probs = torch.softmax(logits.float(), dim=-1)
    finite = torch.isfinite(probs).all(dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sums = probs.sum(dim=-1, keepdim=True)
    good = finite & torch.isfinite(sums.squeeze(-1)) & (sums.squeeze(-1) > 0.0)
    uniform = torch.full_like(probs, 1.0 / DUCT_MARGINAL_ACTIONS)
    probs = torch.where(good.unsqueeze(-1), probs / torch.clamp(sums, min=1e-30), uniform)
    mixed = eps * uniform + (1.0 - eps) * probs
    mixed = torch.clamp(mixed, min=torch.finfo(mixed.dtype).tiny)
    mixed = mixed / mixed.sum(dim=-1, keepdim=True)
    return mixed


def _sample_policy_action_table(states, action_table, policy0, policy1,
                                uniform_sample_prob: float = 0.0,
                                generator=None):
    """按双玩家策略分布采样并排列 child 的 16 个 marginal slot。

    Shape:
        states: [B, state_dim]，即 child state batch。
        action_table: [2, 16, A, 2]，外部候选 target 表。
        policy0/1(states): [B, 16] logits。
        return ids: [B, 2, 16]，每个玩家的候选 action id 排列。
        return targets: [B, 2, 16, A, 2]，按 ids gather 后的连续 target。
        return ordered_probs: [B, 2, 16]，ids 顺序对应的混合概率。

    关键路径：
        两个玩家分别 softmax+epsilon 混合 -> multinomial 无放回采样 16 个 slot ->
        从 action_table[player, id] gather target，供 commit 写入 child。
    """
    import torch

    table = _as_cuda_float_tensor(action_table, device=states.device)
    if table.ndim != 4:
        raise ValueError(
            "action_table must have shape "
            f"({DUCT_PLAYERS}, {DUCT_MARGINAL_ACTIONS}, num_agents, 2), "
            f"got {tuple(table.shape)}"
        )
    expected = (DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS, table.shape[-2], 2)
    if tuple(table.shape) != expected:
        raise ValueError(
            "action_table must have shape "
            f"({DUCT_PLAYERS}, {DUCT_MARGINAL_ACTIONS}, num_agents, 2), "
            f"got {tuple(table.shape)}"
        )

    logits0 = policy0(states)
    logits1 = policy1(states)
    probs0 = _duct_policy_probs(logits0, uniform_sample_prob)
    probs1 = _duct_policy_probs(logits1, uniform_sample_prob)
    ids0 = torch.multinomial(
        probs0, DUCT_MARGINAL_ACTIONS, replacement=False, generator=generator,
    )
    ids1 = torch.multinomial(
        probs1, DUCT_MARGINAL_ACTIONS, replacement=False, generator=generator,
    )
    ids = torch.stack((ids0, ids1), dim=1)

    batch = states.shape[0]
    targets = torch.empty(
        (batch, DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS, table.shape[-2], 2),
        dtype=torch.float32, device=states.device,
    )
    targets[:, 0] = table[0].index_select(0, ids0.reshape(-1)).view(
        batch, DUCT_MARGINAL_ACTIONS, table.shape[-2], 2,
    )
    targets[:, 1] = table[1].index_select(0, ids1.reshape(-1)).view(
        batch, DUCT_MARGINAL_ACTIONS, table.shape[-2], 2,
    )
    ordered_probs = torch.stack(
        (torch.gather(probs0, 1, ids0), torch.gather(probs1, 1, ids1)),
        dim=1,
    )
    return ids.to(torch.int32), targets, ordered_probs


def fill_duct_node_action_slots(bridge: DuctExpandBridge, tree_ids, node_ids,
                                states, action_table, policy0, policy1,
                                uniform_sample_prob: float = 0.0,
                                generator=None):
    """给已有节点填充策略排序的 DUCT marginal action slots。

    Shape:
        tree_ids/node_ids: [B]，指定要填的树和节点。
        states: [B, state_dim]，会同步写入 bridge.node_states。
        写入:
            node_action_ids[tree, node]: [2, 16]。
            node_action_targets[tree, node]: [2, 16, A, 2]。
            node_action_probs[tree, node]: [2, 16]。

    关键路径：
        用当前 state 调 policy0/policy1 -> 采样 slot 排列 ->
        写入 tree storage。root 初始化和 child commit 前的 slot 生成共用这条路径。
    """
    import torch

    with torch.no_grad():
        states_t = _as_cuda_float_tensor(states, device=bridge.device)
        tree_t = torch.as_tensor(tree_ids, dtype=torch.long, device=bridge.device).view(-1)
        node_t = torch.as_tensor(node_ids, dtype=torch.long, device=bridge.device).view(-1)
        if states_t.ndim != 2 or states_t.shape[0] != tree_t.numel():
            raise ValueError("states must be (batch, state_dim) and match tree_ids")
        if states_t.shape[1] != bridge.state_dim:
            raise ValueError(f"states must have dim {bridge.state_dim}, got {states_t.shape[1]}")
        if node_t.numel() != tree_t.numel():
            raise ValueError("node_ids must match tree_ids")

        ids, targets, probs = _sample_policy_action_table(
            states_t, action_table, policy0, policy1,
            uniform_sample_prob=uniform_sample_prob,
            generator=generator,
        )
        bridge.node_states[tree_t, node_t] = states_t
        bridge.node_action_ids[tree_t, node_t] = ids
        bridge.node_action_targets[tree_t, node_t] = targets
        bridge.node_action_probs[tree_t, node_t] = probs
        return ids, targets, probs


def initialize_duct_root_slots(bridge: DuctExpandBridge, root_states,
                               action_table, policy0, policy1,
                               uniform_sample_prob: float = 0.0,
                               generator=None):
    """初始化每棵树 root(node=0) 的 DUCT action slots。

    Shape:
        root_states: [T, state_dim]，T 必须等于 bridge.n_trees。
        action_table/policy 输出 shape 与 fill_duct_node_action_slots 相同。

    关键路径：
        tree_ids = arange(T)，node_ids = 0，然后复用统一的策略填槽逻辑。
    """
    import torch

    tree_ids = torch.arange(bridge.n_trees, dtype=torch.long, device=bridge.device)
    node_ids = torch.zeros_like(tree_ids)
    return fill_duct_node_action_slots(
        bridge, tree_ids, node_ids, root_states, action_table, policy0, policy1,
        uniform_sample_prob=uniform_sample_prob, generator=generator,
    )


def _duct_combine_player_targets(env, parent_targets, team_ids=None):
    """把两个玩家的 slot target 合成为 MINCO 环境需要的 full target。

    Shape:
        parent_targets: [B, 2, A, 2]，stage1 导出的两个玩家 target。
        team_ids: [A]，每个 agent 属于哪个队伍/玩家。
        return: [B, A, 2]，按 agent 所属队伍选择 player0 或 player1 target。

    关键路径：
        从 env.team_ids 或显式 team_ids 取两队标签 -> 构造 player0 mask ->
        torch.where(mask, parent_targets[:,0], parent_targets[:,1])。
    """
    import torch

    if team_ids is None:
        team_ids_t = env.team_ids.to(device=parent_targets.device)
    else:
        team_ids_t = torch.as_tensor(team_ids, dtype=torch.long, device=parent_targets.device)
    team_values = torch.unique(team_ids_t, sorted=True)
    if team_ids_t.shape != (parent_targets.shape[-2],):
        raise ValueError(
            f"team_ids must have shape ({parent_targets.shape[-2]},), "
            f"got {tuple(team_ids_t.shape)}"
        )
    if team_values.numel() != DUCT_PLAYERS:
        raise ValueError(f"DUCT expand expects exactly 2 teams, got {int(team_values.numel())}")

    mask0 = (team_ids_t == team_values[0]).view(1, -1, 1)
    return torch.where(mask0, parent_targets[:, 0], parent_targets[:, 1])


def _duct_step_env_with_projected_targets(env, parent_states, projected_targets):
    """用已投影 target 推进 MINCO 环境一步，避免重复 obstacle projection。

    Shape:
        parent_states: [B, state_dim]，env.pack_state 的扁平 state。
        projected_targets: [B, A, 2]，project_target_away_from_obstacles 输出。
        return next_flat: [B, state_dim]。
        return done/valid: [B]。

    关键路径：
        unpack_state -> robot_transition.transition(projected_targets) ->
        碰撞/越界/active 更新 -> pack_state。
    """
    import torch

    coeff, active, time = env.unpack_state(parent_states)
    active_bool = active > 0.5
    stepped = env.robot_transition.transition(coeff, projected_targets)
    inactive_frozen = env.zero_motion(coeff)
    next_coeff = torch.where(active_bool[..., :, None, None], stepped, inactive_frozen)

    next_time = time + torch.as_tensor(env.dt, dtype=coeff.dtype, device=coeff.device)
    next_positions = env.positions(next_coeff)
    point_colliding = env.collision_mask(next_positions, active_bool)
    obstacle_colliding = env.obstacle_collision_mask(next_positions, active_bool)
    if env.deactivate_on_collision:
        next_active_bool = active_bool & ~point_colliding
    else:
        next_active_bool = active_bool
    if env.deactivate_on_obstacle_collision:
        next_active_bool = next_active_bool & ~obstacle_colliding

    frozen_after_collision = env.zero_motion(next_coeff)
    next_coeff = torch.where(
        next_active_bool[..., :, None, None],
        next_coeff,
        frozen_after_collision,
    )
    valid, _ = env.bounds_validity(env.positions(next_coeff), next_active_bool)
    done = env.done(next_active_bool, next_time, valid, obstacle_colliding)
    next_flat = env.pack_state(next_coeff, next_active_bool, next_time)
    return next_flat, done, valid


def run_duct_expand_stage2_minco(bridge: DuctExpandBridge, env, action_table,
                                 policy0, policy1,
                                 uniform_sample_prob: float = 0.0,
                                 team_ids=None, generator=None):
    """用 MINCO/PyTorch 填充 DUCT expand stage2 buffer。

    输入/输出 shape：
        bridge.expand_valid: [T, W]，只处理 READY job。
        bridge.expand_parent_states: [T, W, state_dim]。
        bridge.expand_parent_targets: [T, W, 2, A, 2]。
        bridge.expand_next_states: [T, W, state_dim]，本函数写入。
        bridge.expand_done: [T, W]，本函数写入。
        bridge.expand_child_action_ids/probs: [T, W, 2, 16]，本函数写入。
        bridge.expand_child_action_targets: [T, W, 2, 16, A, 2]，本函数写入。

    关键算法路径：
        1) gather READY jobs。
        2) 按 env.team_ids 合成 full target [B,A,2]。
        3) 调 project_target_away_from_obstacles 修正障碍约束。
        4) 用已投影 target 推进环境，得到 next_state/done/valid。
        5) 调 policy0/policy1 生成 child 的 2x16 slot 排列。
        6) valid 且 finite 的 job 保持 READY；失败 job 标记 FAILED，交给 commit rollback。
    """
    import torch

    with torch.no_grad():
        job_mask = bridge.expand_valid == int(DUCT_EXPAND_JOB_READY)
        if not bool(job_mask.any().item()):
            return {
                "num_jobs": 0,
                "num_valid": 0,
                "full_targets": None,
                "projected_targets": None,
            }

        parent_states = bridge.expand_parent_states[job_mask]
        parent_targets = bridge.expand_parent_targets[job_mask]
        full_targets = _duct_combine_player_targets(env, parent_targets, team_ids=team_ids)
        coeff, active, _ = env.unpack_state(parent_states)
        projected_targets, _, _ = env.project_target_away_from_obstacles(
            coeff,
            full_targets,
            active > 0.5,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
        next_states, done, valid = _duct_step_env_with_projected_targets(
            env, parent_states, projected_targets,
        )
        finite = torch.isfinite(next_states).all(dim=-1) & torch.isfinite(projected_targets).all(dim=(-1, -2))
        ok = valid.bool() & finite

        bridge.expand_next_states[job_mask] = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        bridge.expand_done[job_mask] = done.to(torch.int32)

        ids, targets, probs = _sample_policy_action_table(
            bridge.expand_next_states[job_mask],
            action_table,
            policy0,
            policy1,
            uniform_sample_prob=uniform_sample_prob,
            generator=generator,
        )
        bridge.expand_child_action_ids[job_mask] = ids
        bridge.expand_child_action_targets[job_mask] = targets
        bridge.expand_child_action_probs[job_mask] = probs

        status = torch.where(
            ok,
            torch.full_like(bridge.expand_valid[job_mask], int(DUCT_EXPAND_JOB_READY)),
            torch.full_like(bridge.expand_valid[job_mask], int(DUCT_EXPAND_JOB_FAILED)),
        )
        bridge.expand_valid[job_mask] = status
        return {
            "num_jobs": int(job_mask.sum().item()),
            "num_valid": int(ok.sum().item()),
            "full_targets": full_targets.detach(),
            "projected_targets": projected_targets.detach(),
        }
