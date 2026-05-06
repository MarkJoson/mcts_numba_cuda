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
DUCT_EDGE_UNEXPANDED = -3
DUCT_EDGE_EXPANDING = -2
DUCT_PLAYER0_MASK = 0x0000FFFF
DUCT_PLAYER1_MASK = 0xFFFF0000
# 手动切到 "categorical" 可恢复 logits + candidate_targets 的离散策略路径。
DUCT_POLICY_OUTPUT = "gaussian"
DUCT_GAUSSIAN_LOG_STD_MIN = -5.0
DUCT_GAUSSIAN_LOG_STD_MAX = 2.0
DUCT_EXPAND_TOPK = 4
DUCT_EXPAND_TOPK_BITS = 2
DUCT_EXPAND_TOPK_MASK = DUCT_EXPAND_TOPK - 1
DUCT_EXPAND_CANDIDATES = DUCT_EXPAND_TOPK * DUCT_EXPAND_TOPK
DUCT_PATH_SLOT_BITS = 16
DUCT_PATH_SLOT_MASK = (1 << DUCT_PATH_SLOT_BITS) - 1
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
#   -3: no rollout/state transition has been created for this joint action.
#   -2: another warp has claimed this edge for Expand.
#   -1: terminal transition (NODE_EXPANDED_TERMINAL).
#  >=0: child node id.
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
#& DUCT新增: path 只保存 (parent_node << 16) | joint_slot。
#& 理由: joint_slot 的高/低 4bit 就是两个玩家动作，backup/rollback 可直接解码，
#&      不需要额外 out_path_actions[T,W,D,2] 缓冲。


# ============================================================
# DUCT device helpers
# ============================================================

@cuda.jit(device=True, inline=True)
def _valid_duct_select_shape(max_edge_steps, node_capacity, node_cnt,
                             max_edges, action_capacity,
                             action_players, action_count_players,
                             path_action_players):
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
def _duct_pack_terminal_selection(node, reason):
    #& DUCT: terminal child 直接存 NODE_EXPANDED_TERMINAL(-1)，packed node 字段不能写负数。
    #& 因此 terminal 输出统一把 node 编成 PACKED_NODE_MASK，kind/reason 表示真实语义。
    return _pack_selection(
        int32(SELECT_TERMINAL),
        int32(PACKED_NODE_MASK),
        int32(0),
        reason,
    )


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
def _duct_claim_joint_expand(edge_tgt_node, node_expand_inflight,
                             tree, node, joint_slot, lane):
    #& DUCT重构: joint action 固定映射到 0..255 的边槽，缺 child 时直接抢占这条边。
    #& 理由: 不再用节点扩展计数分配线性 slot，避免线性扫描和重复联合动作。
    #& Shape:
    #&   edge_tgt_node: [T, N, 256]，UNEXPANDED -> EXPANDING 的边状态表。
    #&   node_expand_inflight: [T, N]，父节点上正在扩展的 edge 数。
    #& 关键路径:
    #&   lane0 对目标 joint edge 做一次 CAS；成功后增加 node_expand_inflight。
    #&   两个玩家动作直接由 joint_slot 解码，不再存冗余 edge_actions 表。
    held = int32(0)
    if lane == int32(0):
        prev = cuda.atomic.cas(
            edge_tgt_node,
            (tree, node, joint_slot),
            int32(DUCT_EDGE_UNEXPANDED),
            int32(DUCT_EDGE_EXPANDING),
        )
        if prev == int32(DUCT_EDGE_UNEXPANDED):
            cuda.atomic.add(node_expand_inflight, (tree, node), int32(1))
            held = int32(1)
    held = cuda.shfl_sync(FULL_MASK, held, 0)
    return held


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
def _duct_claim_soft_expand_fallback(edge_tgt_node, node_expand_inflight,
                                     tree, node, avoid_slot, lane, tie_offset,
                                     sorted_action, sorted_score):
    #& DUCT soft-expand claim。
    #& Shape:
    #&   edge_tgt_node: [T, N, 256]，只在 top4xtop4 候选中的 UNEXPANDED 边上 CAS。
    #& 关键路径:
    #&   16 个 lane 并行评分候选；lane0 逐轮 CAS 最优候选。容量检查由调用方
    #&   tree_full 分支控制，本函数只负责抢边；动作由 candidate slot 解码。
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
def _write_duct_path_step(out_path_eids, tree, wid, depth, lane, node, edge_slot):
    #* Path 编码格式为 (parent_node << 16) | edge_slot。低 16bit 支持最多 65536 个 action slot。
    if lane == int32(0):
        if edge_slot >= int32(0):
            out_path_eids[tree, wid, depth] = (node << int32(DUCT_PATH_SLOT_BITS)) | edge_slot
        else:
            out_path_eids[tree, wid, depth] = int32(-1)


@cuda.jit(device=True, inline=True)
def _rollback_duct_vloss_path_nodes(tree, wid, action_depth, lane,
                                    out_path_eids, action_inflight, virtual_loss,
                                    n_nodes, action_capacity):
    #& DUCT新增: path rollback 按玩家动作维度回滚 virtual loss。
    #& 理由: 单人 PUCT 回滚 edge_inflight；DUCT 没有 joint edge 级别的选择占用。
    #&      动作由 path_eid 低 8bit 的 joint_slot 解码，避免冗余 path action 表。
    player = int32(0)
    half_lane = lane
    if lane >= int32(DUCT_MARGINAL_ACTIONS):
        player = int32(1)
        half_lane = lane - int32(DUCT_MARGINAL_ACTIONS)

    d = half_lane
    while d < action_depth:
        encoded = out_path_eids[tree, wid, d]
        parent = encoded >> int32(DUCT_PATH_SLOT_BITS)
        joint_slot = encoded & int32(DUCT_PATH_SLOT_MASK)
        if encoded < int32(0):
            parent = int32(0)
            joint_slot = int32(0)
        action = joint_slot & int32(DUCT_ACTION_MASK)
        if player == int32(0):
            action = joint_slot >> int32(DUCT_ACTION_BITS)
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


# ============================================================
# DUCT selection kernels
# ============================================================
@cuda.jit(void(
    float32, float32, float32,
    int32[:, :, :],
    float32[:, :, :, :], int32[:, :, :, :], int32[:, :, :, :],
    int32[:, :, :], int32[:, :], int32[:, :], int32[:],
    int32[:, :], int32[:, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_duct_winner_recalc(c_uct, c_pw, alpha_pw,
                                      edge_tgt_node, action_W, action_N, action_inflight,
                                      action_counts, node_N, node_expand_inflight,
                                      tree_nodes,
                                      out_selected_node, out_path_eids, out_path_len):
    """
    双玩家同时决策的 DUCT select kernel。

    一个 CUDA block 负责一棵树，一个 warp 执行一次独立遍历。在每个非终止节点，
    两个玩家分别在自己的 PW 窗口内，用边缘 DUCT 统计选择动作：

        Q_i(s,a) + c_uct * sqrt(log(N_i(s)) / N_i(s,a))

    两个边缘动作直接组成固定的 joint edge slot：

        joint_slot = (action0 << 4) | action1        # 共 256 个 joint action

    Select 只处理这条确定的 joint edge：
        - edge_tgt_node == -3：该 joint edge 还没有 child，用 CAS(-3, -2)
          抢占 Expand 权限，并返回 SELECT_EXPAND(node, joint_slot)。
        - edge_tgt_node == -2：已有其他 warp 正在扩展这条边，回滚本次
          virtual loss 并返回 SELECT_BUSY。
        - edge_tgt_node == -1：该边到达 terminal，记录路径后返回 SELECT_TERMINAL。
        - edge_tgt_node >= 0：该边已有 child，记录路径后继续向 child 遍历。

    PW 是每个玩家独立维护的边缘动作可见窗口，不再按 joint edge 计数。
    terminal 不再使用独立节点状态数组；terminal child 直接以
    NODE_EXPANDED_TERMINAL(-1) 写入 edge_tgt_node[parent, slot]。

    DUCT_SOFT_WINNER / DUCT_SOFT_EXPAND 是导入模块时确定的编译期常量。
    前者允许 winner CAS 失败后保留重复选择；后者在选中边正在扩展时，
    用两个玩家各自的 top-4 动作并行组成 16 个候选，改抢其中最优的
    未扩展 joint edge。

    关键输入：
        edge_tgt_node: [tree, node, 256]，-3 未扩展，-2 正在扩展，-1 terminal，>=0 child。
        action_W/N/inflight: [tree, node, player, action]，每个玩家的边缘 DUCT 统计。
        action_counts: [tree, node, player]，每个玩家的合法边缘动作数，最大 16。
        node_N: 节点访问次数，用于 UCB/PW。
        node_expand_inflight: 当前节点已抢占但尚未完成的 edge Expand 数。
        tree_nodes: [tree]，当前已分配节点上界；select 只读它判断树是否已满。

    输出：
        out_selected_node: 打包后的 SELECT_* 结果，以及被选中的 node/joint slot。
        out_path_eids: 路径边，编码为 (parent_node << 16) | joint_slot。
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
    node_capacity = edge_tgt_node.shape[1]
    action_capacity = action_W.shape[3]
    action_players = action_W.shape[2]
    action_count_players = action_counts.shape[2]

    node_cnt = tree_nodes[tree]

    if not _valid_duct_select_shape(
        max_edge_steps, node_capacity, node_cnt, max_edges,
        action_capacity, action_players, action_count_players,
        int32(DUCT_PLAYERS),
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
    node_limit = node_capacity
    if node_limit > int32(PACKED_NODE_LIMIT):
        node_limit = int32(PACKED_NODE_LIMIT)

    while True:
        #* node id 的有效性完全由 tree_nodes 上界和 terminal sentinel 判定。
        if node < int32(0) or node >= node_cnt:
            _rollback_duct_vloss_path_nodes(
                tree, wid, depth, lane,
                out_path_eids, action_inflight,
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
                out_path_eids, action_inflight,
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
                out_path_eids, action_inflight,
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
        #& 理由: Select 已经选出两个玩家动作，定位边不需要扫描节点扩展表。
        joint_slot = int32(0)
        action1 = cuda.shfl_sync(FULL_MASK, local_best_action, DUCT_MARGINAL_ACTIONS)
        if lane == int32(0):
            joint_slot = _duct_joint_slot(local_best_action, action1)
        joint_slot = cuda.shfl_sync(FULL_MASK, joint_slot, 0)
        joint_child = edge_tgt_node[tree, node, joint_slot]
        joint_child = cuda.shfl_sync(FULL_MASK, joint_child, 0)

        tree_full = int32(0)
        if node_cnt >= node_limit:
            tree_full = int32(1)
        expand_busy = int32(0)
        if joint_child == int32(DUCT_EDGE_UNEXPANDED):
            if tree_full != int32(0):
                expand_busy = int32(0)
            else:
                expand_held = _duct_claim_joint_expand(
                    edge_tgt_node, node_expand_inflight,
                    tree, node, joint_slot, lane,
                )
                if expand_held == int32(1):
                    _write_duct_path_step(
                        out_path_eids, tree, wid, depth, lane, node, joint_slot,
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

        if joint_child == int32(NODE_EXPANDED_TERMINAL):
            _write_duct_path_step(
                out_path_eids, tree, wid, depth, lane, node, joint_slot,
            )
            cuda.syncwarp(FULL_MASK)
            final_packed = _duct_pack_terminal_selection(node, int32(REASON_OK_TERMINAL))
            final_len = depth + int32(2)
            break

        if (
            expand_busy != int32(0) or
            joint_child < int32(0) or joint_child >= node_cnt
        ):
            if (
                int32(DUCT_SOFT_EXPAND) != int32(0) and
                expand_busy != int32(0) and tree_full == int32(0)
            ):
                sorted_action, sorted_score = _duct_sort_actions_halfwarp(
                    c_uct, action_W, action_N, action_inflight,
                    tree, node, allowed, lane, log_parent_n_eff, tie_offset,
                )
                fallback_slot, fallback_held = _duct_claim_soft_expand_fallback(
                    edge_tgt_node, node_expand_inflight,
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
                        out_path_eids, tree, wid, depth, lane, node, fallback_slot,
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
                out_path_eids, action_inflight,
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

        #* Path 编码格式为 (parent_node << 16) | edge_slot。backup 时直接解码。
        _write_duct_path_step(
            out_path_eids, tree, wid, depth, lane, node, joint_slot,
        )
        cuda.syncwarp(FULL_MASK)

        node = joint_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_packed, final_len,
                         out_selected_node, out_path_len)

# ============================================================
# DUCT PyTorch bridge helpers
# ============================================================

class DuctExpandBridge:
    """DUCT expand 的 PyTorch/Numba 共享缓冲区。

    树内持久 storage shape：
        node_states: [T, N, state_dim]。
        node_action_targets: [T, N, 2, 16, A, 2]。
        node_action_probs: [T, N, 2, 16]。

    job staging shape：
        expand_count: [1]。
        expand_tree/parent/slot/child: [T*W] compact jobs。
        expand_next_states: [T*W, state_dim]。
        expand_done: [T*W]。

    关键路径：
        PyTorch stage1 连续写 tree/parent/slot/child；
        Python 调 MINCO projection/transition 和 policy sampling，并直接 scatter child slot；
        PyTorch stage2 批量发布 parent edge。

    注意：
        PyTorch tensor 是 owner；cuda.as_cuda_array 只是零拷贝 view，
        因此 bridge 对象必须在 kernel 使用期间保持存活。
    """

    def __init__(self, n_trees: int, node_capacity: int, state_dim: int,
                 num_agents: int, max_warps: int, device: str = "cuda",
                 target_dtype=None, prob_dtype=None):
        import torch

        self.n_trees = int(n_trees)
        self.node_capacity = int(node_capacity)
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.max_warps = int(max_warps)
        self.max_jobs = self.n_trees * self.max_warps
        self.device = torch.device(device)
        self.target_dtype = target_dtype or torch.float16
        self.prob_dtype = prob_dtype or torch.float16

        #& 树内持久 storage：child 发布后 select/backup 都从这些 tensor 读取。
        self.node_states = torch.zeros(
            (self.n_trees, self.node_capacity, self.state_dim),
            dtype=torch.float32, device=self.device,
        )
        self.node_action_targets = torch.zeros(
            (self.n_trees, self.node_capacity, DUCT_PLAYERS,
             DUCT_MARGINAL_ACTIONS, self.num_agents, 2),
            dtype=self.target_dtype, device=self.device,
        )
        self.node_action_probs = torch.zeros(
            (self.n_trees, self.node_capacity, DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS),
            dtype=self.prob_dtype, device=self.device,
        )

        #& stage1 -> Python stage2 的连续 job staging。
        self.expand_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.expand_tree = torch.full((self.max_jobs,), -1, dtype=torch.int32, device=self.device)
        self.expand_parent = torch.full_like(self.expand_tree, -1)
        self.expand_slot = torch.full_like(self.expand_tree, -1)
        self.expand_child = torch.full_like(self.expand_tree, -1)
        self.expand_next_states = torch.zeros(
            (self.max_jobs, self.state_dim),
            dtype=torch.float32, device=self.device,
        )
        self.expand_done = torch.zeros(
            (self.max_jobs,), dtype=torch.int32, device=self.device,
        )

    def reset_jobs(self) -> None:
        """清空 staging job，不改树内 node_* storage。"""
        #& 用于一轮 expand 前/后清理。tree storage 保留，避免误删已提交 child。
        self.expand_count.zero_()
        self.expand_tree.fill_(-1)
        self.expand_parent.fill_(-1)
        self.expand_slot.fill_(-1)
        self.expand_child.fill_(-1)
        self.expand_next_states.zero_()
        self.expand_done.zero_()


def prepare_duct_expand_stage1_torch(bridge: DuctExpandBridge,
                                     edge_tgt_node,
                                     tree_nodes,
                                     out_selected_node,
                                     ):
    """纯 PyTorch DUCT 扩展阶段1：批量解码 select 输出并分配 child index。

    Shape:
        bridge.expand_*: [T*W]，本函数连续写入 tree/parent/slot/child。
        bridge.expand_count: [1]，本轮 READY job 数。
        edge_tgt_node: int32 [T, N, 256]，select 已经把待扩展 edge 置为 EXPANDING。
        tree_nodes: int32 [T]，本函数按 READY job 数批量递增。
        out_selected_node: int32 [T, W]，packed SELECT_EXPAND(parent, joint_slot)。

    关键算法路径：
        1) 向量化 decode `SELECT_EXPAND(parent, joint_slot)`。
        2) 用 edge_tgt_node == EXPANDING 过滤非扩展 job。
        3) 每棵树用 cumsum 做局部 child id 分配；容量外 job 不进入 READY，不做回滚。
        4) 将 READY job compact 到连续 staging，parent state/target 交给 stage2 gather。

    返回:
        dict(num_jobs, num_ready, num_dropped)，用于调试/benchmark 统计。
    """
    import torch

    with torch.no_grad():
        bridge.reset_jobs()

        raw = out_selected_node.to(torch.int64)
        kind = (raw >> int(PACKED_KIND_SHIFT)) & int(PACKED_KIND_MASK)
        parent = raw & int(PACKED_NODE_MASK)
        slot = (raw >> int(PACKED_SLOT_SHIFT)) & int(PACKED_EDGE_MASK)

        n_trees, max_warps = out_selected_node.shape
        tree_grid = torch.arange(n_trees, dtype=torch.long, device=bridge.device).view(-1, 1)
        tree_grid = tree_grid.expand(n_trees, max_warps)

        node_cap = int(edge_tgt_node.shape[1])
        alloc_limit = min(node_cap, int(PACKED_NODE_LIMIT))
        node_counts = tree_nodes.to(torch.long).view(-1, 1)
        range_ok = (
            (kind == int(SELECT_EXPAND)) &
            (parent >= 0) &
            (parent < node_counts) &
            (parent < node_cap) &
            (slot >= 0) &
            (slot < int(DUCT_JOINT_ACTIONS))
        )

        edge_ok = torch.zeros_like(range_ok, dtype=torch.bool)
        if bool(range_ok.any().item()):
            edge_ok[range_ok] = (
                edge_tgt_node[
                    tree_grid[range_ok],
                    parent[range_ok].to(torch.long),
                    slot[range_ok].to(torch.long),
                ] == int(DUCT_EDGE_EXPANDING)
        )
        claim_ok = range_ok & edge_ok

        local_offset_claim = torch.cumsum(claim_ok.to(torch.long), dim=1) - 1
        available = torch.clamp(
            torch.full_like(node_counts, int(alloc_limit)) - node_counts,
            min=0,
        )
        ready_mask = claim_ok & (local_offset_claim < available)
        child = node_counts + local_offset_claim

        ready_counts = ready_mask.to(torch.int32).sum(dim=1)
        tree_nodes.add_(ready_counts.to(dtype=tree_nodes.dtype))

        ready_jobs = int(ready_mask.sum().item())
        if ready_jobs > 0:
            if ready_jobs > bridge.max_jobs:
                raise RuntimeError(f"ready jobs {ready_jobs} exceed bridge capacity {bridge.max_jobs}")
            bridge.expand_tree[:ready_jobs] = tree_grid[ready_mask].to(torch.int32)
            bridge.expand_parent[:ready_jobs] = parent[ready_mask].to(torch.int32)
            bridge.expand_slot[:ready_jobs] = slot[ready_mask].to(torch.int32)
            bridge.expand_child[:ready_jobs] = child[ready_mask].to(torch.int32)
            bridge.expand_count[0] = ready_jobs

        return {
            "num_jobs": int(claim_ok.sum().item()),
            "num_ready": ready_jobs,
            "num_dropped": int((claim_ok & ~ready_mask).sum().item()),
        }


def _as_cuda_float_tensor(value, *, device):
    """把输入规范化为 CUDA float32 contiguous tensor。"""
    import torch

    return torch.as_tensor(value, dtype=torch.float32, device=device).contiguous()


class DuctActionPolicy:
    """DUCT action policy.

    这个类把 action target 表、双玩家 policy、采样 generator 和排序规则绑在一起；
    只负责产出按概率降序排列的 action targets/probs，不接触 tree storage。
    """

    def __init__(self, candidate_targets, policy0, policy1,
                 uniform_sample_prob: float = 0.0, generator=None):
        self.candidate_targets = candidate_targets
        self.policy0 = policy0
        self.policy1 = policy1
        self.uniform_sample_prob = float(uniform_sample_prob)
        self.generator = generator
        if self.uniform_sample_prob < 0.0 or self.uniform_sample_prob > 1.0:
            raise ValueError(
                "uniform_sample_prob must be in [0, 1], "
                f"got {self.uniform_sample_prob}"
            )

    def rank_targets(self, states):
        """返回 [B,2,16,A,2] target 和 [B,2,16] prob，slot 按概率降序。"""
        import torch

        with torch.no_grad():
            states_t = _as_cuda_float_tensor(states, device=states.device)
            table = self._candidate_target_table(device=states_t.device)
            if DUCT_POLICY_OUTPUT == "gaussian":
                return self._rank_gaussian(states_t, table)
            if DUCT_POLICY_OUTPUT == "categorical":
                return self._rank_categorical(states_t, table)
            raise ValueError(f"unknown DUCT_POLICY_OUTPUT {DUCT_POLICY_OUTPUT!r}")

    def _candidate_target_table(self, *, device):
        table = _as_cuda_float_tensor(self.candidate_targets, device=device)
        if table.ndim != 4:
            raise ValueError(
                "candidate_targets must have shape "
                f"({DUCT_PLAYERS}, {DUCT_MARGINAL_ACTIONS}, num_agents, 2), "
                f"got {tuple(table.shape)}"
            )
        expected = (DUCT_PLAYERS, DUCT_MARGINAL_ACTIONS, table.shape[-2], 2)
        if tuple(table.shape) != expected:
            raise ValueError(
                "candidate_targets must have shape "
                f"({DUCT_PLAYERS}, {DUCT_MARGINAL_ACTIONS}, num_agents, 2), "
                f"got {tuple(table.shape)}"
            )
        return table

    def _mix_logits_with_uniform(self, logits):
        import torch

        if logits.ndim != 2 or logits.shape[1] != DUCT_MARGINAL_ACTIONS:
            raise ValueError(
                f"policy logits must have shape (batch, {DUCT_MARGINAL_ACTIONS}), "
                f"got {tuple(logits.shape)}"
            )
        probs = torch.softmax(logits.float(), dim=-1)
        finite = torch.isfinite(probs).all(dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        sums = probs.sum(dim=-1, keepdim=True)
        good = finite & torch.isfinite(sums.squeeze(-1)) & (sums.squeeze(-1) > 0.0)
        uniform = torch.full_like(probs, 1.0 / DUCT_MARGINAL_ACTIONS)
        probs = torch.where(good.unsqueeze(-1), probs / torch.clamp(sums, min=1e-30), uniform)
        mixed = self.uniform_sample_prob * uniform + (1.0 - self.uniform_sample_prob) * probs
        mixed = torch.clamp(mixed, min=torch.finfo(mixed.dtype).tiny)
        return mixed / mixed.sum(dim=-1, keepdim=True)

    def _gaussian_policy_params(self, output, *, num_agents: int, device):
        import torch

        if isinstance(output, (tuple, list)):
            if len(output) != 2:
                raise ValueError("gaussian policy tuple output must be (mean, log_std)")
            mean = _as_cuda_float_tensor(output[0], device=device)
            log_std = _as_cuda_float_tensor(output[1], device=device)
        else:
            raw = _as_cuda_float_tensor(output, device=device)
            if raw.ndim == 2 and raw.shape[1] == num_agents * 4:
                raw = raw.view(raw.shape[0], num_agents, 4)
            if raw.ndim == 4 and tuple(raw.shape[1:]) == (2, num_agents, 2):
                mean = raw[:, 0]
                log_std = raw[:, 1]
            elif raw.ndim == 3 and tuple(raw.shape[1:]) == (num_agents, 4):
                mean = raw[..., :2]
                log_std = raw[..., 2:]
            elif raw.ndim == 3 and tuple(raw.shape[1:]) == (num_agents, 2):
                mean = raw
                log_std = torch.zeros_like(mean)
            else:
                raise ValueError(
                    "gaussian policy output must be (mean, log_std), "
                    f"(batch, {num_agents}, 4), (batch, {num_agents * 4}), "
                    f"or (batch, 2, {num_agents}, 2); got {tuple(raw.shape)}"
                )

        if mean.ndim != 3 or tuple(mean.shape[1:]) != (num_agents, 2):
            raise ValueError(
                f"gaussian policy mean must have shape (batch, {num_agents}, 2), "
                f"got {tuple(mean.shape)}"
            )
        if log_std.ndim == 0:
            log_std = torch.full_like(mean, float(log_std.item()))
        elif log_std.ndim == 1 and log_std.numel() == 2:
            log_std = log_std.view(1, 1, 2).expand_as(mean)
        elif log_std.ndim == 2 and tuple(log_std.shape) == (num_agents, 2):
            log_std = log_std.unsqueeze(0).expand_as(mean)
        elif log_std.shape != mean.shape:
            log_std = torch.broadcast_to(log_std, mean.shape)

        return mean.contiguous(), torch.clamp(
            log_std.contiguous(),
            DUCT_GAUSSIAN_LOG_STD_MIN,
            DUCT_GAUSSIAN_LOG_STD_MAX,
        )

    def _rank_gaussian(self, states, table):
        import torch

        batch = states.shape[0]
        num_agents = int(table.shape[-2])
        mean0, log_std0 = self._gaussian_policy_params(
            self.policy0(states), num_agents=num_agents, device=states.device,
        )
        mean1, log_std1 = self._gaussian_policy_params(
            self.policy1(states), num_agents=num_agents, device=states.device,
        )

        shape = (batch, DUCT_MARGINAL_ACTIONS, num_agents, 2)
        noise0 = torch.randn(shape, dtype=torch.float32, device=states.device,
                             generator=self.generator)
        noise1 = torch.randn(shape, dtype=torch.float32, device=states.device,
                             generator=self.generator)
        samples0 = mean0.unsqueeze(1) + noise0 * torch.exp(log_std0).unsqueeze(1)
        samples1 = mean1.unsqueeze(1) + noise1 * torch.exp(log_std1).unsqueeze(1)
        scores0 = -0.5 * noise0.square().sum(dim=(-1, -2))
        scores1 = -0.5 * noise1.square().sum(dim=(-1, -2))
        rank0 = torch.argsort(scores0, dim=1, descending=True)
        rank1 = torch.argsort(scores1, dim=1, descending=True)
        gather_rank0 = rank0.view(batch, DUCT_MARGINAL_ACTIONS, 1, 1).expand(
            -1, -1, num_agents, 2,
        )
        gather_rank1 = rank1.view(batch, DUCT_MARGINAL_ACTIONS, 1, 1).expand(
            -1, -1, num_agents, 2,
        )
        targets = torch.stack(
            (
                torch.gather(samples0, 1, gather_rank0),
                torch.gather(samples1, 1, gather_rank1),
            ),
            dim=1,
        )
        probs = torch.stack(
            (
                torch.gather(torch.softmax(scores0, dim=1), 1, rank0),
                torch.gather(torch.softmax(scores1, dim=1), 1, rank1),
            ),
            dim=1,
        )
        return targets, probs

    def _rank_categorical(self, states, table):
        import torch

        logits0 = self.policy0(states)
        logits1 = self.policy1(states)
        probs0 = self._mix_logits_with_uniform(logits0)
        probs1 = self._mix_logits_with_uniform(logits1)
        ids0 = torch.argsort(probs0, dim=1, descending=True)
        ids1 = torch.argsort(probs1, dim=1, descending=True)

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
        return targets, ordered_probs


def rank_action_targets(states, candidate_targets, policy0, policy1,
                        uniform_sample_prob: float = 0.0,
                        generator=None):
    return DuctActionPolicy(
        candidate_targets,
        policy0,
        policy1,
        uniform_sample_prob=uniform_sample_prob,
        generator=generator,
    ).rank_targets(states)


def fill_duct_node_action_slots(bridge: DuctExpandBridge, tree_ids, node_ids,
                                states, candidate_targets, policy0, policy1,
                                uniform_sample_prob: float = 0.0,
                                generator=None):
    """给已有节点填充策略排序的 DUCT marginal action slots。

    Shape:
        tree_ids/node_ids: [B]，指定要填的树和节点。
        states: [B, state_dim]，会同步写入 bridge.node_states。
        写入:
            node_action_targets[tree, node]: [2, 16, A, 2]。
            node_action_probs[tree, node]: [2, 16]。

    关键路径：
        用当前 state 调 policy0/policy1 -> 采样 target slot ->
        只写采样后的 target/prob。root 初始化和 child 发布前的 slot 生成共用这条路径。
    """
    action_policy = DuctActionPolicy(
        candidate_targets,
        policy0,
        policy1,
        uniform_sample_prob=uniform_sample_prob,
        generator=generator,
    )
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

        targets, probs = action_policy.rank_targets(states_t)
        bridge.node_states[tree_t, node_t] = states_t
        bridge.node_action_targets[tree_t, node_t] = targets.to(dtype=bridge.node_action_targets.dtype)
        bridge.node_action_probs[tree_t, node_t] = probs.to(dtype=bridge.node_action_probs.dtype)
        return targets, probs


def initialize_duct_root_slots(bridge: DuctExpandBridge, root_states,
                               candidate_targets, policy0, policy1,
                               uniform_sample_prob: float = 0.0,
                               generator=None):
    """初始化每棵树 root(node=0) 的 DUCT action slots。

    Shape:
        root_states: [T, state_dim]，T 必须等于 bridge.n_trees。
        candidate_targets/policy 输出 shape 与 fill_duct_node_action_slots 相同。

    关键路径：
        tree_ids = arange(T)，node_ids = 0，然后复用统一的策略填槽逻辑。
    """
    import torch

    tree_ids = torch.arange(bridge.n_trees, dtype=torch.long, device=bridge.device)
    node_ids = torch.zeros_like(tree_ids)
    return fill_duct_node_action_slots(
        bridge, tree_ids, node_ids, root_states, candidate_targets, policy0, policy1,
        uniform_sample_prob=uniform_sample_prob, generator=generator,
    )


def combine_team_targets(env, parent_targets, team_ids=None):
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


def step_env_with_projected_targets(env, parent_states, projected_targets):
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


def publish_expand_edges(bridge: DuctExpandBridge, edge_tgt_node):
    """批量发布 compact expand jobs 到 parent edge。

    child 的 state/action storage 必须已经写完；本函数只做最后可见发布点：
    parent edge 从 EXPANDING 变成 child 或 terminal sentinel。
    """
    import torch

    with torch.no_grad():
        edge_tgt_node = torch.as_tensor(edge_tgt_node, device=bridge.device)
        job_count = int(bridge.expand_count[0].item())
        if job_count <= 0:
            return {"num_published": 0}

        tree_ids = bridge.expand_tree[:job_count].to(torch.long)
        parent_nodes = bridge.expand_parent[:job_count].to(torch.long)
        slot_ids = bridge.expand_slot[:job_count].to(torch.long)
        child_nodes = bridge.expand_child[:job_count].to(torch.long)
        done = bridge.expand_done[:job_count].bool()

        publish = torch.where(
            done,
            torch.full_like(child_nodes, int(NODE_EXPANDED_TERMINAL)),
            child_nodes,
        ).to(dtype=edge_tgt_node.dtype)
        edge_tgt_node[tree_ids, parent_nodes, slot_ids] = publish
        return {"num_published": job_count}


def run_duct_expand_stage2_minco(bridge: DuctExpandBridge, env, candidate_targets,
                                 edge_tgt_node, action_W, action_N, action_inflight,
                                 action_counts, node_N, tree_nodes,
                                 policy0, policy1,
                                 uniform_sample_prob: float = 0.0,
                                 team_ids=None, generator=None):
    """用 MINCO/PyTorch 填充 DUCT expand stage2 buffer。

    输入/输出 shape：
        bridge.expand_count: [1]，本轮连续 job 数。
        bridge.expand_tree/parent/slot/child: [Jmax]，stage1 输出的索引。
        bridge.expand_next_states: [Jmax, state_dim]，本函数写入。
        bridge.expand_done: [Jmax]，本函数写入。
        edge_tgt_node/action_W/action_N/action_inflight/action_counts/node_N:
            本函数按 child id 直接批量初始化并发布 parent edge。
        bridge.node_action_probs[tree, child]: [2, 16]，本函数直接 scatter 写入。
        bridge.node_action_targets[tree, child]: [2, 16, A, 2]，本函数直接 scatter 写入。

    关键算法路径：
        1) gather READY jobs。
        2) 按 env.team_ids 合成 full target [B,A,2]。
        3) 调 project_target_away_from_obstacles 修正障碍约束。
        4) 用已投影 target 推进环境，得到 next_state/done/valid。
        5) 调 policy0/policy1 生成 child 的 2x16 target slot。
        6) PyTorch 批量初始化 child storage。
        7) 环境 invalid 作为 terminal 正常提交；最后批量发布 parent edge。
    """
    import torch

    with torch.no_grad():
        edge_tgt_node = torch.as_tensor(edge_tgt_node, device=bridge.device)
        action_W = torch.as_tensor(action_W, device=bridge.device)
        action_N = torch.as_tensor(action_N, device=bridge.device)
        action_inflight = torch.as_tensor(action_inflight, device=bridge.device)
        action_counts = torch.as_tensor(action_counts, device=bridge.device)
        node_N = torch.as_tensor(node_N, device=bridge.device)
        tree_nodes = torch.as_tensor(tree_nodes, device=bridge.device)
        action_policy = DuctActionPolicy(
            candidate_targets,
            policy0,
            policy1,
            uniform_sample_prob=uniform_sample_prob,
            generator=generator,
        )

        job_count = int(bridge.expand_count[0].item())
        if job_count <= 0:
            return {
                "num_jobs": 0,
                "num_valid": 0,
                "full_targets": None,
                "projected_targets": None,
            }

        tree_ids = bridge.expand_tree[:job_count].to(torch.long)
        parent_nodes = bridge.expand_parent[:job_count].to(torch.long)
        slot_ids = bridge.expand_slot[:job_count].to(torch.long)
        parent_states = bridge.node_states[tree_ids, parent_nodes]
        action0 = slot_ids >> DUCT_ACTION_BITS
        action1 = slot_ids & DUCT_ACTION_MASK
        parent_targets = torch.stack(
            (
                bridge.node_action_targets[tree_ids, parent_nodes, 0, action0],
                bridge.node_action_targets[tree_ids, parent_nodes, 1, action1],
            ),
            dim=1,
        ).to(dtype=torch.float32)
        full_targets = combine_team_targets(env, parent_targets, team_ids=team_ids)
        coeff, active, _ = env.unpack_state(parent_states)
        projected_targets, _, _ = env.project_target_away_from_obstacles(
            coeff,
            full_targets,
            active > 0.5,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
        next_states, done, valid = step_env_with_projected_targets(
            env, parent_states, projected_targets,
        )
        finite = torch.isfinite(next_states).all(dim=-1) & torch.isfinite(projected_targets).all(dim=(-1, -2))
        done = done.bool() | ~valid.bool() | ~finite

        clean_next_states = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        bridge.expand_next_states[:job_count] = clean_next_states
        bridge.expand_done[:job_count] = done.to(torch.int32)

        targets, probs = action_policy.rank_targets(clean_next_states)
        child_nodes = bridge.expand_child[:job_count].to(torch.long)
        bridge.node_states[tree_ids, child_nodes] = clean_next_states
        edge_tgt_node[tree_ids, child_nodes] = int(DUCT_EDGE_UNEXPANDED)
        action_W[tree_ids, child_nodes] = 0.0
        action_N[tree_ids, child_nodes] = 0
        action_inflight[tree_ids, child_nodes] = 0
        node_N[tree_ids, child_nodes] = 0
        action_counts[tree_ids, child_nodes, :] = int(DUCT_MARGINAL_ACTIONS)
        terminal = done.bool()
        if bool(terminal.any().item()):
            action_counts[
                tree_ids[terminal],
                child_nodes[terminal],
                :,
            ] = 1
        bridge.node_action_targets[tree_ids, child_nodes] = targets.to(
            dtype=bridge.node_action_targets.dtype,
        )
        bridge.node_action_probs[tree_ids, child_nodes] = probs.to(
            dtype=bridge.node_action_probs.dtype,
        )

        publish_info = publish_expand_edges(bridge, edge_tgt_node)

        return {
            "num_jobs": job_count,
            "num_valid": int(valid.bool().sum().item()),
            "num_published": publish_info["num_published"],
            "full_targets": full_targets.detach(),
            "projected_targets": projected_targets.detach(),
        }
