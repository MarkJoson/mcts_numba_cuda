import math

import numba.cuda as cuda
from numba import void, float32, int32

from puct.puct_gpu_v3 import (
    FULL_MASK,
    INT32_MAX,
    NEG_INF_F32,
    NODE_EXPANDED_TERMINAL,
    PACKED_EDGE_MASK,
    PACKED_NODE_LIMIT,
    PACKED_NODE_MASK,
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
    WARP_SIZE,
    MAX_RECALC_RETRY,
    _allowed_children,
    _claim_expand_slot,
    _init_select_output,
    _pack_selection,
    _score_better_block_local,
    _warp_reduce_best_eid,
    _warp_reduce_sum,
    _write_select_output,
)


POS_INF_F32 = 3.4028234663852886e38
DUCT_PLAYERS = 2


# ============================================================
# DUCT data structure documentation
# ============================================================

#& DUCT新增: DUCT SELECT data layout (two-player simultaneous move)
# -----------------------------------------------------
#& 理由: 单人 PUCT 的 edge_* 统计绑定到一条边；DUCT 需要先让每个玩家
#&      独立选择自己的动作，再把动作组合成 joint move，所以需要记录
#&      “edge_slot 对应哪个 joint action”。
#
# edge_actions[tree, node, edge_slot, player] stores the joint action that
# created an expanded edge. edge_child_id keeps the matching child node.
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
class DuctAction:
    W: float
    N: int
    inflight: int


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
        max_edges > int32(0) and max_edges <= int32(PACKED_EDGE_MASK + 1) and
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
        count1 > int32(0) and count1 <= action_capacity
    )


@cuda.jit(device=True, inline=True)
def _duct_parent_n_eff(node_N, action_inflight,
                       tree, node, action_count0, lane):
    """Node visits for DUCT UCB/PW, including pending player-0 virtual visits."""
    #& DUCT新增: 父节点访问数使用全局 node_N，并加上当前并发中的动作虚拟访问。
    #& 理由: 单人 PUCT 的 parent_n_eff 来自 edge_N/edge_inflight；
    #&      DUCT 的虚拟访问存在 action_inflight，渐宽和 UCB 仍需要并发感知的 n_s。
    inflight_total = int32(0)
    action = lane
    while action < action_count0:
        inflight_total += action_inflight[tree, node, int32(0), action]
        action += int32(WARP_SIZE)

    total = int32(0)
    if lane == int32(0):
        total = node_N[tree, node]
    total = cuda.shfl_sync(FULL_MASK, total, 0)
    total += _warp_reduce_sum(inflight_total)
    return max(total, int32(1))


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
def _duct_reduce_best_action(c_uct, action_W, action_N, action_inflight,
                             tree, node, player, action_count, lane,
                             log_parent_n_eff, tie_offset):
    #& DUCT新增: 单个玩家在自己的动作集合 A_i(s) 上做 reduce。
    #& 理由: joint move 由两个玩家独立 argmax 后组合，不能直接在 edge 上选。
    best_score = float32(NEG_INF_F32)
    best_action = int32(INT32_MAX)
    best_inflight = int32(0)

    action = lane
    while action < action_count:
        inflight = action_inflight[tree, node, player, action]
        score = _duct_ucb_score(
            c_uct,
            action_W[tree, node, player, action],
            action_N[tree, node, player, action],
            inflight,
            log_parent_n_eff,
        )
        if _score_better_block_local(score, int32(action), best_score, best_action,
                                     tie_offset, action_count):
            best_score = score
            best_action = int32(action)
            best_inflight = inflight
        action += int32(WARP_SIZE)

    best_score, best_action, best_inflight = _warp_reduce_best_eid(
        best_score, best_action, best_inflight, tie_offset, action_count,
    )
    return best_action, best_inflight


@cuda.jit(device=True, inline=True)
def _duct_release_two_actions(action_inflight, tree, node, action0, action1, lane):
    #& DUCT新增: 回滚/释放两个玩家各自动作上的 virtual loss。
    #& 理由: DUCT 的并发占用在 action_inflight[player, action]，不是 edge_inflight。
    if lane == int32(0):
        if action0 >= int32(0) and action0 != int32(INT32_MAX):
            cuda.atomic.sub(action_inflight, (tree, node, int32(0), action0), int32(1))
        if action1 >= int32(0) and action1 != int32(INT32_MAX):
            cuda.atomic.sub(action_inflight, (tree, node, int32(1), action1), int32(1))


@cuda.jit(device=True, inline=True)
def _duct_claim_two_actions(action_inflight, tree, node,
                            action0, inflight0, action1, inflight1,
                            lane):
    #& DUCT新增: 两个玩家动作必须同时 claim 成功；第二个失败时回滚第一个。
    #& 理由: 一个 select warp 持有的是 joint move，不能留下半个玩家动作的虚拟访问。
    held = int32(0)
    if lane == int32(0):
        prev0 = cuda.atomic.cas(
            action_inflight, (tree, node, int32(0), action0),
            inflight0, inflight0 + int32(1),
        )
        if prev0 == inflight0:
            prev1 = cuda.atomic.cas(
                action_inflight, (tree, node, int32(1), action1),
                inflight1, inflight1 + int32(1),
            )
            if prev1 == inflight1:
                held = int32(1)
            else:
                cuda.atomic.sub(action_inflight, (tree, node, int32(0), action0), int32(1))
    return cuda.shfl_sync(FULL_MASK, held, 0)


@cuda.jit(device=True, inline=True)
def _duct_best_actions_winner_recalc(c_uct, action_W, action_N, action_inflight,
                                     tree, node, action_count0, action_count1,
                                     lane, log_parent_n_eff, tie_offset,
                                     soft_winner):
    #* 选择最优动作，得到对应的 action 与 inflight
    #& DUCT新增: 分别为 player0/player1 选择动作，然后一起做 winner recalc。
    #& 理由: 保留单人 PUCT 的 CAS 重算模式，避免多个 warp 长时间挤在同一动作上。
    best0 = int32(INT32_MAX)
    best1 = int32(INT32_MAX)
    held = int32(0)
    retry = int32(0)

    tie_offset0 = tie_offset
    if tie_offset0 >= action_count0:
        tie_offset0 = tie_offset0 % action_count0
    tie_offset1 = tie_offset + int32(17)
    if tie_offset1 >= action_count1:
        tie_offset1 = tie_offset1 % action_count1

    while retry < int32(MAX_RECALC_RETRY):
        # TODO. 有些时候32线程
        best0, inflight0 = _duct_reduce_best_action(
            c_uct, action_W, action_N, action_inflight,
            tree, node, int32(0), action_count0, lane,
            log_parent_n_eff, tie_offset0,
        )
        best1, inflight1 = _duct_reduce_best_action(
            c_uct, action_W, action_N, action_inflight,
            tree, node, int32(1), action_count1, lane,
            log_parent_n_eff, tie_offset1,
        )

        if best0 == int32(INT32_MAX) or best1 == int32(INT32_MAX):
            break

        #* 对 inflight 进行 CAS，并检查是否成功 Hold
        held = _duct_claim_two_actions(
            action_inflight, tree, node,
            best0, inflight0, best1, inflight1, lane,
        )
        #* 成功，或超过最大重试次数时退出
        if held == int32(1) or retry + int32(1) >= int32(MAX_RECALC_RETRY):
            break

        retry += int32(1)

    if (
        soft_winner != int32(0) and held == int32(0) and
        best0 != int32(INT32_MAX) and best1 != int32(INT32_MAX)
    ):
        if lane == int32(0):
            cuda.atomic.add(action_inflight, (tree, node, int32(0), best0), int32(1))
            cuda.atomic.add(action_inflight, (tree, node, int32(1), best1), int32(1))
        held = int32(1)

    held = cuda.shfl_sync(FULL_MASK, held, 0)
    return best0, best1, held


@cuda.jit(device=True, inline=True)
def _duct_find_joint_edge(edge_child_id, edge_actions,
                          tree, node, cur_expanded, action0, action1, lane,
                          tie_offset):
    #& DUCT新增: DUCT 先得到 (a0, a1)，再在线性 expanded edge 中查找 joint edge。
    #& 理由: 已展开结构仍复用 edge_child_id/node_expanded，避免为 joint action 建二维表。
    best_score = float32(NEG_INF_F32)
    best_slot = int32(INT32_MAX)
    best_inflight = int32(0)

    slot = lane
    while slot < cur_expanded:
        score = float32(NEG_INF_F32)
        if (
            edge_actions[tree, node, slot, int32(0)] == action0 and
            edge_actions[tree, node, slot, int32(1)] == action1
        ):
            score = float32(1.0)
        if _score_better_block_local(score, int32(slot), best_score, best_slot,
                                     tie_offset, cur_expanded):
            best_score = score
            best_slot = int32(slot)
        slot += int32(WARP_SIZE)

    best_score, best_slot, best_inflight = _warp_reduce_best_eid(
        best_score, best_slot, best_inflight, tie_offset, cur_expanded,
    )

    if best_score < float32(0.0):
        best_slot = int32(INT32_MAX)

    child = int32(-1)
    if best_slot != int32(INT32_MAX):
        child = edge_child_id[tree, node, best_slot]
    child = cuda.shfl_sync(FULL_MASK, child, 0)
    return best_slot, child


@cuda.jit(device=True, inline=True)
def _duct_reduce_best_existing_edge(c_uct, edge_child_id, edge_actions,
                                    action_W, action_N, action_inflight,
                                    tree, node, cur_expanded, action_capacity,
                                    lane,
                                    log_parent_n_eff, tie_offset):
    #& DUCT新增: 当选出的 joint move 还不能扩展时，在已有 joint edge 中做 fallback。
    #& 理由: 渐宽饱和时不能创建新边，但 select 仍应沿已展开子树继续推进。
    best_score = float32(NEG_INF_F32)
    best_slot = int32(INT32_MAX)
    best_inflight = int32(0)

    slot = lane
    while slot < cur_expanded:
        child = edge_child_id[tree, node, slot]
        score = float32(NEG_INF_F32)
        if child >= int32(0):
            action0 = edge_actions[tree, node, slot, int32(0)]
            action1 = edge_actions[tree, node, slot, int32(1)]
            if (
                action0 >= int32(0) and action0 < action_capacity and
                action1 >= int32(0) and action1 < action_capacity
            ):
                n0 = action_N[tree, node, int32(0), action0]
                n1 = action_N[tree, node, int32(1), action1]
                inflight0 = action_inflight[tree, node, int32(0), action0]
                inflight1 = action_inflight[tree, node, int32(1), action1]
                score0 = _duct_ucb_score(
                    c_uct, action_W[tree, node, int32(0), action0],
                    n0, inflight0, log_parent_n_eff,
                )
                score1 = _duct_ucb_score(
                    c_uct, action_W[tree, node, int32(1), action1],
                    n1, inflight1, log_parent_n_eff,
                )
                score = score0 + score1

        if _score_better_block_local(score, int32(slot), best_score, best_slot,
                                     tie_offset, cur_expanded):
            best_score = score
            best_slot = int32(slot)
        slot += int32(WARP_SIZE)

    best_score, best_slot, best_inflight = _warp_reduce_best_eid(
        best_score, best_slot, best_inflight, tie_offset, cur_expanded,
    )

    if best_score == float32(NEG_INF_F32):
        best_slot = int32(INT32_MAX)
    return best_slot


@cuda.jit(device=True, inline=True)
def _duct_best_existing_edge_winner_recalc(c_uct, edge_child_id, edge_actions,
                                           action_W, action_N, action_inflight,
                                           tree, node, cur_expanded,
                                           action_capacity, lane,
                                           log_parent_n_eff, tie_offset,
                                           soft_winner):
    #* 在已完全展开的孩子边中选择 DUCT 最优边。
    #* 只在 winner 对应的两个玩家动作上保留 virtual loss。使用 compare and set 原子操作进行处理
    #& DUCT新增: 已有边的 score 是两个玩家边际 UCB 的组合分数。
    #& 理由: fallback 必须选择一条真实存在的 joint edge，但价值仍来自玩家独立动作统计。
    best_slot = int32(INT32_MAX)
    best0 = int32(INT32_MAX)
    best1 = int32(INT32_MAX)
    held = int32(0)
    retry = int32(0)

    while retry < int32(MAX_RECALC_RETRY):
        best_slot = _duct_reduce_best_existing_edge(
            c_uct, edge_child_id, edge_actions,
            action_W, action_N, action_inflight,
            tree, node, cur_expanded, action_capacity, lane,
            log_parent_n_eff, tie_offset,
        )
        if best_slot == int32(INT32_MAX):
            break

        best0 = edge_actions[tree, node, best_slot, int32(0)]
        best1 = edge_actions[tree, node, best_slot, int32(1)]
        inflight0 = action_inflight[tree, node, int32(0), best0]
        inflight1 = action_inflight[tree, node, int32(1), best1]
        best0 = cuda.shfl_sync(FULL_MASK, best0, 0)
        best1 = cuda.shfl_sync(FULL_MASK, best1, 0)
        inflight0 = cuda.shfl_sync(FULL_MASK, inflight0, 0)
        inflight1 = cuda.shfl_sync(FULL_MASK, inflight1, 0)

        held = _duct_claim_two_actions(
            action_inflight, tree, node,
            best0, inflight0, best1, inflight1, lane,
        )
        if held == int32(1) or retry + int32(1) >= int32(MAX_RECALC_RETRY):
            break
        retry += int32(1)

    if (
        soft_winner != int32(0) and held == int32(0) and
        best_slot != int32(INT32_MAX) and
        best0 != int32(INT32_MAX) and best1 != int32(INT32_MAX)
    ):
        if lane == int32(0):
            cuda.atomic.add(action_inflight, (tree, node, int32(0), best0), int32(1))
            cuda.atomic.add(action_inflight, (tree, node, int32(1), best1), int32(1))
        held = int32(1)

    held = cuda.shfl_sync(FULL_MASK, held, 0)
    return best_slot, best0, best1, held


@cuda.jit(device=True, inline=True)
def _write_duct_path_step(out_path_eids, out_path_actions,
                          tree, wid, depth, lane, node, edge_slot,
                          action0, action1):
    #* Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
    #& DUCT新增: 同步写 out_path_actions，保存每个玩家本次实际选择的动作。
    #& 理由: DUCT backup 只更新被选中的 a_i，因此 path 里必须有 action0/action1。
    if lane == int32(0):
        if edge_slot >= int32(0):
            out_path_eids[tree, wid, depth] = (node << int32(8)) | edge_slot
        else:
            out_path_eids[tree, wid, depth] = int32(-1)
        out_path_actions[tree, wid, depth, int32(0)] = action0
        out_path_actions[tree, wid, depth, int32(1)] = action1


@cuda.jit(device=True, inline=True)
def _rollback_duct_vloss_path_nodes(tree, wid, action_depth, lane,
                                    out_path_eids, out_path_actions,
                                    action_inflight, virtual_loss,
                                    n_nodes, action_capacity):
    #& DUCT新增: path rollback 按玩家动作维度回滚 virtual loss。
    #& 理由: 单人 PUCT 回滚 edge_inflight；DUCT 没有 joint edge 级别的选择占用。
    d = lane
    while d < action_depth:
        encoded = out_path_eids[tree, wid, d]
        parent = encoded >> int32(8)
        if encoded < int32(0):
            parent = int32(0)
        action0 = out_path_actions[tree, wid, d, int32(0)]
        action1 = out_path_actions[tree, wid, d, int32(1)]
        assert parent >= int32(0) and parent < n_nodes
        assert action0 >= int32(0) and action0 < action_capacity
        assert action1 >= int32(0) and action1 < action_capacity
        cuda.atomic.sub(action_inflight, (tree, parent, int32(0), action0), virtual_loss)
        cuda.atomic.sub(action_inflight, (tree, parent, int32(1), action1), virtual_loss)
        d += int32(WARP_SIZE)




# ============================================================
# DUCT selection kernels
# ============================================================
@cuda.jit(void(
    float32, float32, float32, int32,
    int32[:, :, :], int32[:, :, :, :],
    float32[:, :, :, :], int32[:, :, :, :], int32[:, :, :, :],
    int32[:, :, :], int32[:, :], int32[:, :], int32[:, :], int32[:],
    int32[:, :], int32[:, :, :], int32[:, :, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_duct_winner_recalc(c_uct, c_pw, alpha_pw, soft_winner,
                                      edge_tgt_node, edge_actions,
                                      action_W, action_N, action_inflight,
                                      action_counts, node_N, node_expand_inflight,
                                      node_expanded, tree_nodes,
                                      out_selected_node, out_path_eids,
                                      out_path_actions, out_path_len):
    """
    DUCT 是为多人玩家同时决策设计的算法。Turn-based 可以简化为单人决策，但是同时决策的场景却有所不同。
    和单人 MCTS 一样, 每一个 warp 独立遍历一棵树。对于每一个节点 玩家0/1 同时根据各自的 边际决策数据 选择动作。
        W_i(s,a) / N_i(s,a) + c * sqrt(log(N(s)) / N_i(s,a))
    
    由于涉及到了两层扩展, 分别是边扩展和节点扩展. 两种扩展会互相耦合, 因此算法变得更加复杂。
    - 边扩展，对节点的边进行扩展。扩展边后，由于产生了一系列全新的组合，对应的环境状态是缺失的。
    - 节点扩展，计算对应的边产生的结果；需要调用环境模型。
    
    两个玩家形成的联合决策，对应了实际的一条转移边，与目标节点。存在两种情况：
    - 边不存在, 需要先采样动作。这种情况实际不存在, 因为每次的 Expand 阶段都会执行两步: 计算环境+扩展动作
    - 边存在，但节点不存在。需要先执行环境模拟，得到下一步的节点。
    - 边存在，节点存在，走到下一步。
    
    单人决策时边的概念被弱化, MCTS以节点为中心, Select 的输出是一个节点。
    边扩展通过调用策略网络或使用随机策略得到动作, 主要计算发生在神经网络, 代价与动作数不成正比. 在每个节点只能包含有限少数个边的情况下，一般事先扩展出全部的边.
    
    ---
     
    多人同时决策时，因为在 Select 阶段选择的联合动作可能并没有被模拟, 未在节点池中，所以 Select 的输出可能是 (StateNode + JointAction)。
    如果决策某一方决定新增一条动作, 可能的做法是: 
    - 计算出所有新 Joint Action 的组合对应的环境/价值, 那么比较浪费资源
    - 只计算出一种 Joint Action 的组合对应的环境/价值, 那么对于任意一种其他选择, 都将再次触发 Expand.
    
    不过考虑到 MCTS 是由四个阶段组成的: Select + Expand + Rollout + Backward. 无论是否边存在, Select 都会输出两部分结果:
    - Step1, 需要计算环境的节点
    - Step2, 需要扩展动作的节点，包含上一步需要扩展环境的节点，以及叶子节点
    事实上, Select 只会输出 (S,A), 因为需要注意到的是 Select 阶段使用树是有边但尚未结果，需要扩展动作的节点.
    
    在 Rollout 阶段 + Backward 阶段，常规做法是并发数与 Select 保持一致，因此不太会因为在 Select 阶段计算出所有环境而使得后续的并发动态调整。
    然而也有反例, 在PUCT情况下, 为每个新开的节点首先扩展4~8个动作(需要额外处理多人情况, 建议是每玩家预先采集4个, 共评估16个),
    同时评估其价值(Rollout + ValueNetwork)，会辅助 MCTS 算法更好的计算. 在Rollout第一步计算的状态转移, 会被直接丢弃或进入缓存区?...
    
    # TODO. edge_tgt_node 有了新的含义
    
    其次的一个问题是，如何根据两个独立的动作定位到联合动作对应的边？
    
    输入：
        edge_child_id: (S,A)->S 的转移
        edge_actions: 有效转移数量
        action_W: 边缘价值
        action_N: 边缘访问次数
        action_inflight: 边缘 inflight, 用于决策
        action_counts: 边缘 动作数量
        node_N: 当前节点的访问次数
        node_expand_inflight: 当前节点用于扩展的边数量
        ---
        node_expanded: 已经扩展的边（联合动作）数量
        tree_nodes: 当前树的节点数量
        
    # TODO. 为联合扩展池设计渐进扩展，以限制总节点数量。
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
        node_info = node_expanded[tree, node]               #& 当前扩展节点的动作个数，范围 [0, max_edges)
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_packed = _pack_selection(
                int32(SELECT_TERMINAL),
                node,
                int32(0),
                int32(REASON_OK_TERMINAL),
            )
            final_len = depth + int32(1)
            break

        #* 非 terminal 的负数或超过动作数的 expanded 计数都是非法状态。
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


        #* 超过最大rollout长度，直接截断；终止态的节点。node_expanded 使用负数 sentinel 表示 terminal。需要考虑用 Value 网络重新估计节点价值
        if depth >= max_edge_steps:
            final_packed = _pack_selection(
                int32(SELECT_DEPTH_LIMIT),
                node,
                int32(0),
                int32(REASON_OK_DEPTH_LIMIT),
            )
            final_len = depth + int32(1)
            break


        #* 计算当前允许的节点数量
        cur_expanded = node_info
        parent_n_eff = _duct_parent_n_eff(
            node_N, action_inflight, tree, node, action_count0, lane,
        )
        allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)
        log_parent_n_eff = math.log(float32(parent_n_eff))
        
        #& DUCT新增: 每个玩家独立用 DUCT UCB 选择动作，再组合成 joint move。
        #& 理由: 这是 DUCT 和单人 PUCT select 的核心差异；所有的边已经扩展完毕（采样得到），而
        tie_offset = wid
        best_action0, best_action1, claim_held = _duct_best_actions_winner_recalc(
            c_uct, action_W, action_N, action_inflight,
            tree, node, action_count0, action_count1, lane,
            log_parent_n_eff, tie_offset, soft_winner,
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
            if best_action0 == int32(INT32_MAX) or best_action1 == int32(INT32_MAX):
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

        #& DUCT新增: joint move 可能已经有边，也可能是一个尚未展开的新组合。
        #& 理由: DUCT 的动作选择空间是 A_0(s) x A_1(s)，但树结构仍按 expanded edge 存孩子。
        edge_tie_offset = wid
        if cur_expanded > int32(0) and edge_tie_offset >= cur_expanded:
            edge_tie_offset = edge_tie_offset % cur_expanded
        joint_slot = int32(INT32_MAX)
        joint_child = int32(-1)
        if cur_expanded > int32(0):
            joint_slot, joint_child = _duct_find_joint_edge(
                edge_tgt_node, edge_actions,
                tree, node, cur_expanded,
                best_action0, best_action1, lane,
                edge_tie_offset,
            )

        if joint_slot == int32(INT32_MAX):
            #* fast-path，先看看这个节点有多少 inflight 在执行创建任务
            expand_inflight = int32(0)
            if lane == int32(0):
                expand_inflight = node_expand_inflight[tree, node]
            expand_inflight = cuda.shfl_sync(FULL_MASK, expand_inflight, 0)

            expand_slot = int32(-1)
            if allowed > cur_expanded and cur_expanded + expand_inflight < allowed:
                #* 用 node_expand_inflight ticket 给同一父节点分配不同 expand slot。
                expand_slot = _claim_expand_slot(
                    node_expand_inflight, tree, node, cur_expanded, allowed, lane,
                )

            if expand_slot >= int32(0):
                #& DUCT新增: 扩展 slot 中写入当前 joint move 的两个玩家动作。
                #& 理由: 后续 expand 阶段需要把 edge_actions[slot] 填成同一个 joint move。
                _write_duct_path_step(
                    out_path_eids, out_path_actions,
                    tree, wid, depth, lane, node, expand_slot,
                    best_action0, best_action1,
                )
                final_packed = _pack_selection(
                    int32(SELECT_EXPAND),
                    node,
                    expand_slot,
                    int32(REASON_OK_EXPAND),
                )
                final_len = depth + int32(1)
                break

            _duct_release_two_actions(
                action_inflight, tree, node, best_action0, best_action1, lane,
            )
            cuda.syncwarp(FULL_MASK)

            #* 当前节点是叶子节点，且已经有其他warp做扩展时，无法继续往下探索了，直接退出
            if cur_expanded == int32(0):
                _rollback_duct_vloss_path_nodes(
                    tree, wid, depth, lane,
                    out_path_eids, out_path_actions, action_inflight,
                    int32(1), node_cnt, action_capacity,
                )
                cuda.syncwarp(FULL_MASK)
                final_packed = _pack_selection(
                    int32(SELECT_BUSY),
                    node,
                    int32(0),
                    int32(REASON_BUSY_EXPAND_INFLIGHT),
                )
                final_len = int32(0)
                break

            #~~~~~~~~ allowed <= cur_expanded ~~~~~~~~

            #& DUCT新增: 如果独立选出的 joint move 不能扩展，就释放它并改选已有 edge。
            #& 理由: 渐宽饱和时不能强行创建新 joint edge，否则会破坏 node_expanded/slot 约束。
            fallback_slot, fallback_action0, fallback_action1, fallback_held = (
                _duct_best_existing_edge_winner_recalc(
                    c_uct, edge_tgt_node, edge_actions,
                    action_W, action_N, action_inflight,
                    tree, node, cur_expanded, action_capacity, lane,
                    log_parent_n_eff, edge_tie_offset, soft_winner,
                )
            )
            if fallback_held == int32(0):
                _rollback_duct_vloss_path_nodes(
                    tree, wid, depth, lane,
                    out_path_eids, out_path_actions, action_inflight,
                    int32(1), node_cnt, action_capacity,
                )
                cuda.syncwarp(FULL_MASK)
                if fallback_slot == int32(INT32_MAX):
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

            joint_slot = fallback_slot
            best_action0 = fallback_action0
            best_action1 = fallback_action1
            joint_child = edge_tgt_node[tree, node, joint_slot]

        #* 当未取得 held 机会时直接回滚；拿到 held 后再校验 child 合法性。
        if joint_child < int32(0) or joint_child >= node_cnt:
            _duct_release_two_actions(
                action_inflight, tree, node, best_action0, best_action1, lane,
            )
            cuda.syncwarp(FULL_MASK)
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
                int32(REASON_INVALID_CHILD_OOB),
            )
            break

        #* Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
        #& DUCT新增: 这里额外记录 action0/action1，供 DUCT backup 更新每个玩家的动作统计。
        _write_duct_path_step(
            out_path_eids, out_path_actions,
            tree, wid, depth, lane, node, joint_slot,
            best_action0, best_action1,
        )
        cuda.syncwarp(FULL_MASK)

        node = joint_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_packed, final_len,
                         out_selected_node, out_path_len)