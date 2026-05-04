import math
import numpy as np
import numba as nb
import numba.cuda as cuda
from numba import void, float32, int32

# ============================================================
# Constants
# ============================================================

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF
INT32_MAX = 2147483647
NEG_INF_F32 = -3.4028234663852886e38

NODE_EXPANDED_TERMINAL = -1  # node_expanded sentinel: terminal node
PACKED_NODE_LIMIT = 16384    # final_node stores node_id in 14 bits
PACKED_NODE_MASK = 0x3FFF
PACKED_EDGE_MASK = 0xFF
PACKED_KIND_SHIFT = 22
PACKED_SLOT_SHIFT = 14
PACKED_KIND_MASK = 0x7
PACKED_REASON_SHIFT = 25
PACKED_REASON_MASK = 0x7F

# Selection result kinds
SELECT_INVALID = 0      # No valid selection; virtual loss already rolled back
SELECT_EXPAND = 1       # Node needs expansion (PW not saturated); CAS lock held
SELECT_TERMINAL = 2     # Traversal hit a terminal node
SELECT_DEPTH_LIMIT = 3  # Hit path capacity limit; treat as evaluation leaf
SELECT_BUSY = 4         # Temporary contention; no virtual loss is held

# Per-warp exit reason codes (debug)
REASON_UNSET = 0
REASON_OK_EXPAND = 1
REASON_OK_TERMINAL = 2
REASON_OK_DEPTH_LIMIT = 3
REASON_BUSY_EXPAND_INFLIGHT = 4
REASON_BUSY_WINNER_RECALC = 5
REASON_INVALID_SHAPE = 10
REASON_INVALID_NODE_INFO = 11
REASON_INVALID_EXPAND_TICKET = 12
REASON_INVALID_WINNER_CAS = 13
REASON_INVALID_NO_VALID_EDGE = 14
REASON_INVALID_CHILD_OOB = 15
REASON_INVALID_UNKNOWN = 16

PACKED_INVALID = (SELECT_INVALID << PACKED_KIND_SHIFT) | PACKED_NODE_MASK

MAX_SELECT_RETRY = 8
MAX_RECALC_RETRY = 2

# ============================================================
# Data structure documentation
# ============================================================

class Tree:
    nodes: int     # 当前树中有效 node id 的上界，合法 node: [0, nodes)

class Traverse:
    path: list          # 记录遍历的路径

# 由[tree_id, node_id]索引
class Node:
    expanded: int       # >=0: 已经展开的 edge 个数; -1: terminal
    prepare_expand: int # 预计展开的节点数量
    state: int          # 对应的状态空间，指向状态池
    value: float        # 对应当前状态的状态价值
    

# 由[tree_id, node_id, edge_id]索引
class Edge:
    child_id: int       # -1: Invalid, 否则是正常的node id
    prior: float        # 先验概率 (用于PUCT公式), 对应的NN采样概率
    N: int              # 访问次数 (真实，只在 backup 时增加)
    inflight: int       # 指向该节点的虚拟访问计数
    W: float            # 累积动作价值 (只在 backup 时增加真实 value)


@cuda.jit(device=True, inline=True)
def _score_better(score, eid, best_score, best_eid):
    '''检查当前 score 与 best_score 哪个更优
    输入:
        score: (-inf, inf)
        eid: [0, ...)
        best_score: (-inf, inf)
        best_eid: [0, ...), MAX_INT
    返回:
        True: score 更优 
        False: best 更优
    异常处理:
        score == best_score, eid 小者更优(包含 best_eid=MAX_INT 的情况)
    '''
    return (score>best_score) or (score==best_score and eid<best_eid)


@cuda.jit(device=True, inline=True)
def _tie_key_block_local(eid, tie_offset, span):
    key = eid - tie_offset
    if key < int32(0):
        key += span
    return key


@cuda.jit(device=True, inline=True)
def _score_better_block_local(score, eid, best_score, best_eid, tie_offset, span):
    if score > best_score:
        return True
    if score == best_score:
        if best_eid == int32(INT32_MAX):
            return True
        return (
            _tie_key_block_local(eid, tie_offset, span) <
            _tie_key_block_local(best_eid, tie_offset, span)
        )
    return False


@cuda.jit(device=True, inline=True)
def _warp_reduce_best_eid(score, eid, inflight, tie_offset, span):
    other_score = cuda.shfl_down_sync(FULL_MASK, score, 16)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 16)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 16)
    if _score_better_block_local(other_score, other_eid, score, eid, tie_offset, span):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 8)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 8)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 8)
    if _score_better_block_local(other_score, other_eid, score, eid, tie_offset, span):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 4)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 4)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 4)
    if _score_better_block_local(other_score, other_eid, score, eid, tie_offset, span):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 2)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 2)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 2)
    if _score_better_block_local(other_score, other_eid, score, eid, tie_offset, span):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 1)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 1)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 1)
    if _score_better_block_local(other_score, other_eid, score, eid, tie_offset, span):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    score = cuda.shfl_sync(FULL_MASK, score, 0)
    eid = cuda.shfl_sync(FULL_MASK, eid, 0)
    inflight = cuda.shfl_sync(FULL_MASK, inflight, 0)
    return score, eid, inflight


@cuda.jit(int32(int32), device=True, inline=True)
def _warp_reduce_sum(val):
    other = cuda.shfl_down_sync(FULL_MASK, val, 16)
    val += other
    other = cuda.shfl_down_sync(FULL_MASK, val, 8)
    val += other
    other = cuda.shfl_down_sync(FULL_MASK, val, 4)
    val += other
    other = cuda.shfl_down_sync(FULL_MASK, val, 2)
    val += other
    other = cuda.shfl_down_sync(FULL_MASK, val, 1)
    val += other
    val = cuda.shfl_sync(FULL_MASK, val, 0)
    return val


@cuda.jit(int32(float32, float32, int32, int32), device=True, inline=True)
def _allowed_children(c_pw, alpha_pw, n_node, max_edges):
    """渐宽扩展公式: allowed = ceil(c_pw * N(s)^alpha_pw)
        n_node: 取值 [1, ...)
        max_edges: {4,8,16,32,...}
    返回值: [1, ...)
    """
    n = n_node

    if alpha_pw == float32(0.5):
        val = c_pw * math.sqrt(float32(n))
    else:
        val = c_pw * math.pow(float32(n), alpha_pw)

    if val >= float32(max_edges):
        return max_edges

    return max(int32(1), int32(math.ceil(val)))


@cuda.jit(float32(float32, float32, float32, int32, int32, float32), device=True, inline=True)
def _puct_score(cpuct, prior, w, n_edge, n_edge_inflight, sqrt_parent_n_eff):
    """PUCT with virtual-loss-aware effective counts:
        Q   = W(s,a) / N_real(s,a)       (0 when N_real=0)
        U   = cpuct * P(s,a) * sqrt(N_parent_eff) / (1 + N_edge_eff)
        N_edge_eff = N_real + N_inflight

    前提:
        n_edge > 0: 由外部循环保证。
    Virtual visits inflate N_edge_eff to reduce the exploration bonus
    for sibling warps, discouraging duplicate selection of the same edge.
    """
    n_edge_eff = n_edge + n_edge_inflight
    q = float32(0.0)
    if n_edge > int32(0):
        q = w / float32(n_edge)
    u = cpuct * prior * sqrt_parent_n_eff / float32(n_edge_eff + int32(1))
    return q + u


@cuda.jit(device=True, inline=True)
def _rollback_vloss_path(tree, wid, depth, lane,
                          out_path_eids, edge_inflight, virtual_loss,
                          n_nodes, max_edges):
    """回退 path 上的 inflight 变量.
    前提:
        out_path_eids 中的所有 eid 均有效, 由循环内保存逻辑保证，这里仅保留 assert 供调试时的断言检查
    """
    d = lane
    while d < depth:
        encoded = out_path_eids[tree, wid, d]
        parent = encoded >> int32(8)
        slot = encoded & int32(0xFF)
        assert parent >= int32(0) and parent < n_nodes          # TODO. 检查边界条件
        assert slot >= int32(0) and slot < max_edges
        cuda.atomic.sub(edge_inflight, (tree, parent, slot), virtual_loss)
        
        d += int32(WARP_SIZE)


@cuda.jit(device=True, inline=True)
def _pack_selection(kind, node, expand_slot, reason):
    return (
        (reason << int32(PACKED_REASON_SHIFT))
        | (kind << int32(PACKED_KIND_SHIFT))
        | (expand_slot << int32(PACKED_SLOT_SHIFT))
        | node
    )


@cuda.jit(device=True, inline=True)
def _valid_select_shape(max_edge_steps, node_capacity, node_cnt, max_edges):
    return (
        max_edge_steps >= int32(1) and
        node_capacity > int32(0) and
        node_cnt > int32(0) and
        node_cnt <= node_capacity and
        node_cnt <= int32(PACKED_NODE_LIMIT) and
        max_edges > int32(0) and max_edges <= int32(PACKED_EDGE_MASK + 1)
    )


@cuda.jit(device=True, inline=True)
def _init_select_output(tree, wid, lane, out_selected_node, out_path_len):
    if lane == int32(0):
        out_selected_node[tree, wid] = _pack_selection(
            int32(SELECT_INVALID),
            int32(PACKED_NODE_MASK),
            int32(0),
            int32(REASON_UNSET),
        )
        out_path_len[tree, wid] = int32(0)


@cuda.jit(device=True, inline=True)
def _write_select_output(tree, wid, lane, packed_selected, final_len,
                         out_selected_node, out_path_len):
    if lane == int32(0):
        out_selected_node[tree, wid] = packed_selected
        out_path_len[tree, wid] = final_len


@cuda.jit(device=True, inline=True)
def _claim_expand_slot(node_expand_inflight, tree, node, start_slot, limit_slot, lane):
    """多线程扩展的ticket检查, 当前线程是否拿到了原子操作的权限。仅限 lane=0 线程参与.
    输入:
        start_slot: 当前已经扩展的slot数量
        limit_slot: 当前允许扩展的slot数量
    输出:
        -1:         没有拿到ticket
        other>=0:   拿到ticket, 指示下一个slot的存放位置
    异常情况:
        start_slot > limit_slot: 已经在外部检测，不会进入该函数
    """
    
    expand_slot = int32(-1)
    if lane == int32(0):
        ticket = cuda.atomic.add(node_expand_inflight, (tree, node), int32(1))
        slot = start_slot + ticket
        if slot < limit_slot:
            expand_slot = slot
        else:
            cuda.atomic.sub(node_expand_inflight, (tree, node), int32(1))
    return cuda.shfl_sync(FULL_MASK, expand_slot, 0)


@cuda.jit(device=True, inline=True)
def _parent_n_eff(edge_N, edge_inflight,
                  tree, node, cur_expanded, lane):
    '''计算节点的遍历次数, 范围: [1,...)
    异常情况: cur_expanded == 0, 返回值 total==1.
    '''
    total = int32(0)
    eid = lane
    while eid < cur_expanded:
        total += edge_N[tree, node, eid] + edge_inflight[tree, node, eid]
        eid += int32(WARP_SIZE)
    return max(_warp_reduce_sum(total), int32(1))


@cuda.jit(device=True, inline=True)
def _reduce_best_edge(cpuct, edge_prior, edge_W, edge_N, edge_inflight,
                        tree, node, cur_expanded, lane,
                        sqrt_parent_n_eff, tie_offset):
    ''' 并行读取并Reduce, 得到最优的动作。
    输入:
        cur_expanded: [1, ...)
        n_nodes: 当前树的节点数
    返回值:
        best_eid: INT32_MAX(所有节点都异常), >=0 正常返回
        best_inflight. 即计算 Value 时使用的 inflight. 用于后续的 cas 操作。
    异常:
    - 当前节点扩展边指向的节点是无效节点, 跳过该条边的计算, 不更新 best_score 和 best_eid. 因此不会指向无效边
    - 当前节点扩展边所有指向的节点是无效节点时, score=-inf,因此 返回 INT32_MAX;
    '''
    best_score = float32(NEG_INF_F32)
    best_eid = int32(INT32_MAX)
    best_inflight = int32(0)

    eid = lane
    while eid < cur_expanded:
        inflight = edge_inflight[tree, node, eid]
        score = _puct_score(
            cpuct,
            edge_prior[tree, node, eid],
            edge_W[tree, node, eid],
            edge_N[tree, node, eid],
            inflight,
            sqrt_parent_n_eff,
        )
        if _score_better_block_local(score, int32(eid), best_score, best_eid,
                                     tie_offset, cur_expanded):
            best_score = score
            best_eid = int32(eid)
            best_inflight = inflight
        eid += int32(WARP_SIZE)

    best_score, best_eid, best_inflight = _warp_reduce_best_eid(
        best_score, best_eid, best_inflight, tie_offset, cur_expanded,
    )
    return best_eid, best_inflight


@cuda.jit(device=True, inline=True)
def _best_edge_winner_recalc(cpuct, edge_prior, edge_W, edge_N,
                             edge_inflight, tree, node, cur_expanded,
                             lane, sqrt_parent_n_eff, tie_offset, soft_winner):
    '''多次重试选择下一个最优的孩子节点
    返回:
        best_eid>=0, INT32_MAX
        held: True/False
    异常: 
    - 当前节点扩展边所有指向的节点是无效节点, 返回 best_eid=INT_MAX, held=False
    - 超过最大重试次数, 返回 best_eid>=0, held=False
    '''
    
    best_eid = int32(INT32_MAX)
    held = int32(0)
    retry = int32(0)

    while retry < int32(MAX_RECALC_RETRY):
        #* 选择最优动作，得到对应的 eid 与 inflight
        best_eid, best_inflight = _reduce_best_edge(                    #& 当前节点扩展边所有指向的节点是无效节点时, 返回INT32_MAX; 否则一定有效，且不为 INT_MAX
            cpuct, edge_prior, edge_W, edge_N, edge_inflight,
            tree, node, cur_expanded, lane, sqrt_parent_n_eff, tie_offset,
        )
        
        #* 上一步出现异常，当前节点扩展边所有指向的节点是无效节点
        if best_eid == INT32_MAX:
            break

        #* 对 inflight 进行 CAS，并检查是否成功 Hold
        if lane == int32(0):
            prev_inflight = cuda.atomic.cas(edge_inflight, (tree, node, best_eid), best_inflight, best_inflight + int32(1))
            held = int32(1) if prev_inflight == best_inflight else int32(0) # 说明成功持有
        held = cuda.shfl_sync(FULL_MASK, held, 0)   # 同步给所有线程
        
        #* 成功，或超过最大重试次数时退出
        if held == int32(1) or retry + int32(1) >= int32(MAX_RECALC_RETRY): 
            break

        retry += int32(1)

    if soft_winner != int32(0) and held == int32(0) and best_eid != int32(INT32_MAX):
        if lane == int32(0):
            cuda.atomic.add(edge_inflight, (tree, node, best_eid), int32(1))
        held = int32(1)

    held = cuda.shfl_sync(FULL_MASK, held, 0)

    return best_eid, held


@cuda.jit(device=True, inline=True)
def _write_path_edge(out_path_eids, tree, wid, depth, lane, node, eid):
    if lane == int32(0):
        out_path_eids[tree, wid, depth] = (node << int32(8)) | eid


# ============================================================
# Selection kernel
# ============================================================

@cuda.jit(void(
    float32, float32, float32, int32,
    int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :],
    int32[:, :, :], int32[:, :], int32[:, :], int32[:],
    int32[:, :], int32[:, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_winner_recalc(cpuct, c_pw, alpha_pw, soft_winner,
                                 edge_child_id, edge_prior, edge_W, edge_N,
                                 edge_inflight, node_expand_inflight, node_expanded, tree_nodes,
                                 out_selected_node, out_path_eids, out_path_len):
    """
    每个 warp 独立遍历树，选出待扩展节点。1 block = 1 tree; 1 warp = 1 条遍历路径。

    # TODO. 这个soft_winner不是这样作为标志位的...，是类似宏定义的存在；或者如果要作为参数，可以考虑做自适应的根据某些信息进行开关，这个作为自适应算法的超参数存在
    # TODO. 考虑到节点扩展阶段是调用神经网络获得分布后进行反复采样的过程，因此应该一次性填充所有的有效动作。在 expand 阶段，按需调用环境的状态转移接口、与价值评估接口。
    输出：
      out_selected_node[tree, wid]  — SELECT_EXPAND / SELECT_TERMINAL / SELECT_DEPTH_LIMIT / SELECT_INVALID + 被选中的节点 id
      out_path_eids[tree, wid, :]   — 路径上的边 id
      out_path_len[tree, wid]       — 路径上的节点数
    """
    tree = cuda.blockIdx.x
    lane = cuda.threadIdx.x & int32(31)
    wid = cuda.threadIdx.x >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    if wid >= min(cuda.blockDim.x >> int32(5), out_selected_node.shape[1]):
        return

    _init_select_output(tree, wid, lane, out_selected_node, out_path_len)

    max_edges = edge_child_id.shape[2]
    max_edge_steps = out_path_eids.shape[2]
    node_capacity = node_expanded.shape[1]
    node_cnt = int32(0)
    if lane == int32(0):
        node_cnt = tree_nodes[tree]
    node_cnt = cuda.shfl_sync(FULL_MASK, node_cnt, 0)

    if not _valid_select_shape(max_edge_steps, node_capacity, node_cnt, max_edges):
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
        node_info = node_expanded[tree, node]               # &当前扩展节点的动作个数，范围 [0, max_edges) 
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_packed = _pack_selection(
                int32(SELECT_TERMINAL),
                node,
                int32(0),
                int32(REASON_OK_TERMINAL),
            )
            final_len = depth + int32(1)
            break

        #* 检查node_expanded数组的数据，边个数是否超标。非 terminal 的负数或超过动作数的 expanded 计数都是非法状态。
        if node_info < int32(0) or node_info > max_edges:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1), node_cnt, max_edges)
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
        parent_n_eff = int32(1)
        allowed = int32(1)
        if cur_expanded != int32(1) or c_pw != float32(0.0):        #* fast-path 在 cur_expanded=1 时 bypass 计算
            parent_n_eff = _parent_n_eff(                                       #& 父节点的遍历次数，返回值范围 [1, ...)
                edge_N, edge_inflight, tree, node, cur_expanded, lane,
            )
            allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)#& 允许的最大节点数，返回值范围 [1, ...)
        if allowed > cur_expanded:
            #* fast-path，先看看这个节点有多少 inflight 在执行创建任务
            expand_inflight = int32(0)
            if lane == int32(0):
                expand_inflight = node_expand_inflight[tree, node]
            expand_inflight = cuda.shfl_sync(FULL_MASK, expand_inflight, 0)

            expand_slot = int32(-1)
            if cur_expanded + expand_inflight < allowed:
                #* 用 node_expand_inflight ticket 给同一父节点分配不同 expand slot。
                expand_slot = _claim_expand_slot(                               #& 输出: -1(失败), 其他(成功, 指示slot存放位置)
                    node_expand_inflight, tree, node, cur_expanded, allowed, lane,
                )
            if expand_slot >= int32(0):
                final_packed = _pack_selection(int32(SELECT_EXPAND), node, expand_slot, int32(REASON_OK_EXPAND))
                final_len = depth + int32(1)
                break

            #* 当前节点是叶子节点，且已经有其他warp做扩展时，无法继续往下探索了，直接退出
            if cur_expanded == int32(0):
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1), node_cnt, max_edges)
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

        #* 在已完全展开的孩子边中选择 PUCT 最优边。
        #* 只在 winner 边上保留 virtual loss。使用 compare and set 原子操作进行处理
        tie_offset = wid
        if tie_offset >= cur_expanded:
            tie_offset = tie_offset % cur_expanded
        best_eid = int32(0)
        claim_held = int32(0)
        if cur_expanded == int32(1):        #* fast-path 只有一个节点扩展就别争了，继续往下吧，没必要卡一边瓶颈
            if lane == int32(0):
                cuda.atomic.add(edge_inflight, (tree, node, int32(0)), int32(1))
                claim_held = int32(1)
            claim_held = cuda.shfl_sync(FULL_MASK, claim_held, 0)
        else:
            sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))
            best_eid, claim_held = _best_edge_winner_recalc(
                cpuct, edge_prior, edge_W, edge_N, edge_inflight,
                tree, node, cur_expanded, lane, sqrt_parent_n_eff, tie_offset, soft_winner,
            )

        #* 当未取得 held 机会时直接回滚；拿到 held 后再校验 child 合法性。
        if claim_held == int32(0):
            cuda.syncwarp(FULL_MASK)
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1), node_cnt, max_edges)
            cuda.syncwarp(FULL_MASK)
            if best_eid == int32(INT32_MAX):
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

        best_child = edge_child_id[tree, node, best_eid]
        if best_child < int32(0) or best_child >= node_cnt:
            if lane == int32(0):
                cuda.atomic.sub(edge_inflight, (tree, node, best_eid), int32(1))
            cuda.syncwarp(FULL_MASK)
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1), node_cnt, max_edges)
            cuda.syncwarp(FULL_MASK)
            final_packed = _pack_selection(
                int32(SELECT_INVALID),
                int32(PACKED_NODE_MASK),
                int32(0),
                int32(REASON_INVALID_CHILD_OOB),
            )
            break
        
        #* Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
        _write_path_edge(out_path_eids, tree, wid, depth, lane, node, best_eid)
        cuda.syncwarp(FULL_MASK)

        node = best_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_packed, final_len,
                         out_selected_node, out_path_len)
