import math

import numba.cuda as cuda
from numba import void, float32, int32

from puct.puct_gpu_v3 import (
    FULL_MASK,
    INT32_MAX,
    NEG_INF_F32,
    NODE_EXPANDED_TERMINAL,
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
    _init_select_output,
    _pack_selection,
    _score_better_block_local,
    _warp_reduce_best_eid,
    _warp_reduce_sum,
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


@cuda.jit(int32(int32), device=True, inline=True)
def _duct_slot_action0(slot):
    return slot >> int32(DUCT_ACTION_BITS)


@cuda.jit(int32(int32), device=True, inline=True)
def _duct_slot_action1(slot):
    return slot & int32(DUCT_ACTION_MASK)


@cuda.jit(device=True, inline=True)
def _duct_parent_n_eff(node_N, action_inflight,
                       tree, node, player, action_count, lane):
    """Node visits for DUCT UCB/PW, including one player's pending visits."""
    #& DUCT新增: 父节点访问数使用全局 node_N，并加上当前玩家并发中的动作虚拟访问。
    #& 理由: 单人 PUCT 的 parent_n_eff 来自 edge_N/edge_inflight；
    #&      DUCT 的 PW 现在是逐玩家边缘动作窗口，不能再依赖 joint edge 数量。
    inflight_total = int32(0)
    action = lane
    while action < action_count:
        inflight_total += action_inflight[tree, node, player, action]
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
                                     lane, log_parent_n_eff0,
                                     log_parent_n_eff1, tie_offset,
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
            log_parent_n_eff0, tie_offset0,
        )
        best1, inflight1 = _duct_reduce_best_action(
            c_uct, action_W, action_N, action_inflight,
            tree, node, int32(1), action_count1, lane,
            log_parent_n_eff1, tie_offset1,
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
def _duct_claim_joint_expand(edge_tgt_node, edge_actions, node_expand_inflight,
                             tree, node, joint_slot, action0, action1, lane):
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
            edge_actions[tree, node, joint_slot, int32(0)] = action0
            edge_actions[tree, node, joint_slot, int32(1)] = action1
            cuda.atomic.add(node_expand_inflight, (tree, node), int32(1))
            held = int32(1)
    return cuda.shfl_sync(FULL_MASK, held, 0)


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
        parent_n_eff0 = _duct_parent_n_eff(
            node_N, action_inflight, tree, node, int32(0), action_count0, lane,
        )
        parent_n_eff1 = _duct_parent_n_eff(
            node_N, action_inflight, tree, node, int32(1), action_count1, lane,
        )
        allowed0 = _allowed_children(c_pw, alpha_pw, parent_n_eff0, action_count0)
        allowed1 = _allowed_children(c_pw, alpha_pw, parent_n_eff1, action_count1)
        log_parent_n_eff0 = math.log(float32(parent_n_eff0))
        log_parent_n_eff1 = math.log(float32(parent_n_eff1))

        #& DUCT新增: 每个玩家独立用 DUCT UCB 选择动作，再组合成 joint move。这是 DUCT 和单人 PUCT select 的核心差异；
        tie_offset = wid
        best_action0, best_action1, claim_held = _duct_best_actions_winner_recalc(
            c_uct, action_W, action_N, action_inflight,
            tree, node, allowed0, allowed1, lane,
            log_parent_n_eff0, log_parent_n_eff1, tie_offset, soft_winner,
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

        #& DUCT重构: joint action 数量固定为 256，两份 4-bit 边缘动作直接组成 slot。
        #& 理由: Select 已经选出两个玩家动作，定位边不需要扫描 node_expanded。
        joint_slot = _duct_joint_slot(best_action0, best_action1)
        joint_child = edge_tgt_node[tree, node, joint_slot]
        joint_child = cuda.shfl_sync(FULL_MASK, joint_child, 0)

        expand_busy = int32(0)
        if joint_child == int32(DUCT_EDGE_UNEXPANDED):
            expand_held = _duct_claim_joint_expand(
                edge_tgt_node, edge_actions, node_expand_inflight,
                tree, node, joint_slot, best_action0, best_action1, lane,
            )
            if expand_held == int32(1):
                _write_duct_path_step(
                    out_path_eids, out_path_actions,
                    tree, wid, depth, lane, node, joint_slot,
                    best_action0, best_action1,
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
            best_action0, best_action1,
        )
        cuda.syncwarp(FULL_MASK)

        node = joint_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_packed, final_len,
                         out_selected_node, out_path_len)
