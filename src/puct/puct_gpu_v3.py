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

# Selection result kinds
SELECT_INVALID = 0      # No valid selection; virtual loss already rolled back
SELECT_EXPAND = 1       # Node needs expansion (PW not saturated); CAS lock held
SELECT_TERMINAL = 2     # Traversal hit a terminal node
SELECT_DEPTH_LIMIT = 3  # Hit path capacity limit; treat as evaluation leaf

SEL_IN_PROGRESS = -2    # final_node: selection still in progress (replaces done flag)
SEL_INVALID_NODE = -1   # final_node / output: no valid node selected
PACKED_INVALID = (SELECT_INVALID << PACKED_KIND_SHIFT) | PACKED_NODE_MASK

MAX_SELECT_RETRY = 8
MAX_RECALC_RETRY = 2

# ============================================================
# Data structure documentation
# ============================================================

class Tree:
    node_count: int     # 当前树中有效 node id 的上界，合法 node: [0, node_count)

class Traverse:
    path: list          # 记录遍历的路径

# 由[tree_id, node_id]索引
class Node:
    expanded: int       # >=0: 已经展开的 edge 个数; -1: terminal
    prepare_expand: int # 预计展开的节点数量
    state: int          # 对应的状态空间，指向状态池
    value: float        # 对应当前状态的状态价值
    inflight: int       # 指向该节点的虚拟访问计数
    expand_inflight: int # 该节点上正在进行的 expand claim 数量

# 由[tree_id, edge_id]索引
# edge_id = (node_id << 8) | slot   (8-bit slot for MAX_ACTION=256)
class Edge:
    child_id: int       # -1: Invalid, 否则是正常的node id
    prior: float        # 先验概率 (用于PUCT公式)
    N: int              # 访问次数 (真实，只在 backup 时增加)
    W: float            # 累积动作价值 (只在 backup 时增加真实 value)
    prob: float         # 对应的NN采样概率



@cuda.jit(device=True, inline=True)
def _score_better(score, eid, best_score, best_eid):
    if score > best_score:
        return True
    if score == best_score:
        if best_eid == int32(INT32_MAX):
            return True
        return eid < best_eid
    return False


@cuda.jit(device=True, inline=True)
def _tie_key_block_local(eid, tie_offset, span):
    key = eid - tie_offset
    if key < int32(0):
        key += span
    return key


@cuda.jit(device=True, inline=True)
def _score_better_block_local(score, eid, best_score, best_eid, tie_offset, span):
    """
    Score comparison for warp reduce. Ties are distributed across warps in the
    same block by rotating the edge priority with the warp id. This is purely a
    block-local tie-break; it does not coordinate multiple blocks on one tree.
    """
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
def _warp_reduce_best_eid(score, eid, inflight):
    other_score = cuda.shfl_down_sync(FULL_MASK, score, 16)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 16)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 16)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 8)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 8)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 8)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 4)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 4)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 4)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 2)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 2)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 2)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 1)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 1)
    other_inflight = cuda.shfl_down_sync(FULL_MASK, inflight, 1)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        inflight = other_inflight

    score = cuda.shfl_sync(FULL_MASK, score, 0)
    eid = cuda.shfl_sync(FULL_MASK, eid, 0)
    inflight = cuda.shfl_sync(FULL_MASK, inflight, 0)
    return score, eid, inflight


@cuda.jit(int32(int32), device=True, inline=True)
def _warp_reduce_sum(val):
    """
    Warp-level sum reduction. All lanes contribute their val,
    all lanes receive the total sum.
    """
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
    """
    Progressive widening:
        allowed = ceil(c_pw * N(s)^alpha_pw)
    Clamped to [1, max_edges].
    """
    n = n_node
    if n < 1:
        n = int32(1)

    if alpha_pw == float32(0.5):
        val = c_pw * math.sqrt(float32(n))
    else:
        val = c_pw * math.pow(float32(n), alpha_pw)

    if val >= float32(max_edges):
        return max_edges

    k = int32(math.ceil(val))
    if k < 1:
        k = int32(1)

    return k


@cuda.jit(float32(float32, float32, float32, int32, int32, float32), device=True, inline=True)
def _puct_score(cpuct, prior, w, n_edge, n_edge_inflight, sqrt_parent_n_eff):
    """
    PUCT with virtual-loss-aware effective counts:
        Q   = W(s,a) / N_real(s,a)       (0 when N_real=0)
        U   = cpuct * P(s,a) * sqrt(N_parent_eff) / (1 + N_edge_eff)
        N_edge_eff = N_real + N_inflight

    Virtual visits inflate N_edge_eff to reduce the exploration bonus
    for sibling warps, discouraging duplicate selection of the same edge.
    """
    n_edge_eff = n_edge + n_edge_inflight
    q = float32(0.0)
    if n_edge > 0:
        q = w / float32(n_edge)
    u = cpuct * prior * sqrt_parent_n_eff / float32(n_edge_eff + int32(1))
    return q + u


@cuda.jit(device=True, inline=True)
def _rollback_vloss_path(tree, wid, depth, lane,
                          out_path_eids, edge_child_id, node_inflight,
                          virtual_loss, n_nodes):
    """
    Roll back virtual visits on node_inflight for the current attempt.
    Path entries encode (parent << 8) | slot; load the child node and decrement
    the child-node virtual visit.
    Caller must syncwarp after return.
    """
    d = lane
    while d < depth:
        encoded = out_path_eids[tree, wid, d]
        if encoded >= 0:
            parent = encoded >> int32(8)
            slot = encoded & int32(0xFF)
            if parent >= int32(0) and parent < n_nodes:
                child = edge_child_id[tree, parent, slot]
                if child >= int32(0) and child < n_nodes:
                    cuda.atomic.sub(node_inflight, (tree, child), virtual_loss)
        d += int32(WARP_SIZE)


# @cuda.jit(device=True, inline=True)
# def _claim_vloss_counter_cas(node_inflight, tree, node):
#     """
#     Claim one virtual visit with compare-and-swap on the counter value:
#         old -> old + 1

#     This is deliberately not a 0/1 lock. The returned conflict bit records
#     whether the child node already had inflight visits before this warp's claim,
#     while the counter still supports multiple simultaneous holders.
#     """
#     held = int32(0)
#     conflict = int32(0)
#     can_claim = int32(1)

#     old = node_inflight[tree, node]
#     prev = cuda.atomic.cas(node_inflight, (tree, node), old, old + int32(1))
#     if prev == old:
#         held = int32(1)
#         if old > int32(0):
#             conflict = int32(1)

#     # Bounded CAS avoids spinning forever under heavy contention. If the edge
#     # still looks claimable after the retry budget, fall back to atomic add so
#     # a valid traversal is not converted into SELECT_INVALID by CAS churn alone.
#     if held == int32(0) and can_claim != int32(0):
#         old = cuda.atomic.add(node_inflight, (tree, node), int32(1))
#         held = int32(1)
#         if old > int32(0):
#             conflict = int32(1)

#     return held, conflict


@cuda.jit(device=True, inline=True)
def _pack_selection(kind, node, expand_slot):
    return (kind << int32(PACKED_KIND_SHIFT)) | (expand_slot << int32(PACKED_SLOT_SHIFT)) | node


@cuda.jit(device=True, inline=True)
def _valid_select_shape(max_edge_steps, node_capacity, node_limit, max_edges):
    return (
        max_edge_steps >= int32(1) and
        node_capacity > int32(0) and
        node_limit > int32(0) and
        node_limit <= node_capacity and
        node_limit <= int32(PACKED_NODE_LIMIT) and
        max_edges > int32(0) and max_edges <= int32(PACKED_EDGE_MASK + 1)
    )


@cuda.jit(device=True, inline=True)
def _init_select_output(tree, wid, lane, out_selected_node, out_path_len):
    if lane == int32(0):
        out_selected_node[tree, wid] = int32(PACKED_INVALID)
        out_path_len[tree, wid] = int32(0)


@cuda.jit(device=True, inline=True)
def _write_select_output(tree, wid, lane, final_node, final_len,
                         out_selected_node, out_path_len):
    if lane == int32(0):
        if final_node >= int32(0):
            out_selected_node[tree, wid] = final_node
            out_path_len[tree, wid] = final_len
        else:
            out_selected_node[tree, wid] = int32(PACKED_INVALID)
            out_path_len[tree, wid] = int32(0)


@cuda.jit(device=True, inline=True)
def _claim_expand_slot(node_expand_inflight, tree, node, start_slot, limit_slot, lane):
    """
    Claim one expansion slot with a per-node ticket. Multiple warps may expand
    the same node concurrently, but each successful warp receives a distinct
    slot in [start_slot, limit_slot).
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
def _parent_n_eff(edge_child_id, edge_N, node_inflight,
                  tree, node, cur_expanded, n_nodes, lane):
    total = int32(0)
    eid = lane
    while eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
        if child >= int32(0) and child < n_nodes:
            total += edge_N[tree, node, eid] + node_inflight[tree, child]
        eid += int32(WARP_SIZE)
    return max(_warp_reduce_sum(total), int32(1))


@cuda.jit(device=True, inline=True)
def _reduce_best_edge(cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
                        tree, node, cur_expanded, n_nodes, lane,
                        sqrt_parent_n_eff):
    ''' 并行读取并Reduce, 得到最优的动作。
    返回值: best_eid >=0; 当找到有效的最大值时。除非 cur_expanded == 0, 否则一定有一个 eid 产生.
    best_inflight. 即计算 Value 时使用的 inflight. 用于后续的 cas 操作。
    '''
    best_score = float32(NEG_INF_F32)
    best_eid = int32(-1)
    best_inflight = int32(0)

    eid = lane
    while eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
        if child >= int32(0) and child < n_nodes:
            inflight = node_inflight[tree, child]
            score = _puct_score(
                cpuct,
                edge_prior[tree, node, eid],
                edge_W[tree, node, eid],
                edge_N[tree, node, eid],
                inflight,
                sqrt_parent_n_eff,
            )
            if _score_better(score, int32(eid), best_score, best_eid):
                best_score = score
                best_eid = int32(eid)
                best_inflight = inflight
        eid += int32(WARP_SIZE)

    best_score, best_eid, best_inflight = _warp_reduce_best_eid(best_score, best_eid, best_inflight)
    return best_eid, best_inflight


@cuda.jit(device=True, inline=True)
def _best_edge_winner_recalc(cpuct, edge_child_id, edge_prior, edge_W, edge_N,
                             node_inflight, tree, node, cur_expanded, n_nodes,
                             lane, sqrt_parent_n_eff):
    best_eid = int32(INT32_MAX)
    held = False
    retry = int32(0)

    while retry < int32(MAX_RECALC_RETRY):
        # 选择最优动作，得到对应的 eid 与 inflight
        best_eid, best_inflight = _reduce_best_edge(
            cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
            tree, node, cur_expanded, n_nodes, lane, sqrt_parent_n_eff,
        )

        # 对 inflight 进行 CAS，并检查是否成功 Hold
        if lane == int32(0): # and eid >= int32(0) and eid < cur_expanded: # !太严瑾了，但是目前不需要，就只有 _best_edge_winner_recalc 调用，而且之前的函数一定返回有效值 
            child = edge_child_id[tree, node, best_eid]
            if child >= int32(0) and child < n_nodes:
                prev_inflight = cuda.atomic.cas(node_inflight, (tree, node), best_inflight, best_inflight + int32(1))
                held = (prev_inflight == best_inflight) # 说明成功持有
        held = cuda.shfl_sync(FULL_MASK, held, 0)   # 同步给所有线程
        
        # 成功，或超过最大重试次数时退出
        if held == int32(1) or retry + int32(1) >= int32(MAX_RECALC_RETRY): 
            break

        cuda.syncwarp(FULL_MASK)
        held = False
        retry += int32(1)

    return best_eid, held


@cuda.jit(device=True, inline=True)
def _write_path_edge(out_path_eids, tree, wid, depth, lane, node, eid):
    if lane == int32(0):
        out_path_eids[tree, wid, depth] = (node << int32(8)) | eid


# ============================================================
# Selection kernel
# ============================================================

@cuda.jit(void(
    float32, float32, float32,
    int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :],
    int32[:, :], int32[:, :], int32[:, :], int32[:],
    int32[:, :], int32[:, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_winner_recalc(cpuct, c_pw, alpha_pw,
                                 edge_child_id, edge_prior, edge_W, edge_N,
                                 node_inflight, node_expand_inflight, node_expanded, node_count,
                                 out_selected_node, out_path_eids, out_path_len):
    """
    每个 warp 独立遍历树，选出待扩展节点。1 block = 1 tree; 1 warp = 1 条遍历路径。

    与 _select_kernel_native 的差异：
      - native 在评分前对所有候选边预加 virtual loss，然后 rollback losers。
      - winner_recalc 先读取当前 inflight 评分，再用 counter-CAS 只 claim
        winner 边，即 CAS(old, old + 1)，而不是 0/1 互斥锁。
      - 如果 winner claim 时 old > 0，说明同层其它遍历已经占用这个候选；
        本 warp 会先释放自己的 claim，再排除原 winner 重算一次，并尝试
        claim 新 winner。

    这个版本更接近 ref.py 的 CPU batch 语义：一个 readout 选中路径后留下
    virtual loss，后续 readout 再基于新 inflight 做选择。但它不是评分前全候选
    preclaim，因此并发语义比 native 弱，主要作为可选实验路径。

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
    node_limit = int32(0)
    if lane == int32(0):
        node_limit = node_count[tree]
    node_limit = cuda.shfl_sync(FULL_MASK, node_limit, 0)

    if not _valid_select_shape(max_edge_steps, node_capacity, node_limit, max_edges):
        return

    final_node = int32(SEL_IN_PROGRESS)
    final_len = int32(0)
    node = int32(0)
    depth = int32(0)

    while True:
        #* 防御性编程: 节点的id值是无效值，或者大于最大节点数，认为是无效节点
        if node < 0 or node >= node_limit:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), node_limit)
            cuda.syncwarp(FULL_MASK)
            break

        # Path buffer is full: stop before terminal/PW/selection logic can claim work.
        if depth >= max_edge_steps:
            final_node = _pack_selection(int32(SELECT_DEPTH_LIMIT), node, int32(0))
            final_len = depth + int32(1)
            break

        node_info = node_expanded[tree, node]

        #* 终止态的节点。node_expanded 使用负数 sentinel 表示 terminal。需要考虑用 Value 网络重新估计节点价值
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_node = _pack_selection(int32(SELECT_TERMINAL), node, int32(0))
            final_len = depth + int32(1)
            break

        #* 非 terminal 的负数或超过动作数的 expanded 计数都是非法状态。
        if node_info < int32(0) or node_info > max_edges:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), node_limit)
            cuda.syncwarp(FULL_MASK)
            break


        #* 计算当前允许的节点数量 
        cur_expanded = node_info
        parent_n_eff = _parent_n_eff(
            edge_child_id, edge_N, node_inflight,
            tree, node, cur_expanded, node_limit, lane,
        )

        allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)

        #* 渐进增长树：用 node_expand_inflight ticket 给同一父节点分配不同 expand slot。
        if allowed > cur_expanded:
            expand_slot = _claim_expand_slot(
                node_expand_inflight, tree, node, cur_expanded, allowed, lane,
            )
            if expand_slot >= int32(0):
                final_node = _pack_selection(int32(SELECT_EXPAND), node, expand_slot)
                final_len = depth + int32(1)
            else:
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), node_limit)
                cuda.syncwarp(FULL_MASK)
            break

        #* 在已完全展开的孩子边中选择 PUCT 最优边。
        #* 只在 winner 边上保留 virtual loss。使用 compare and set 原子操作进行处理
        sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))
        best_eid, claim_held = _best_edge_winner_recalc(
            cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
            tree, node, cur_expanded, node_limit, lane, sqrt_parent_n_eff,
        )

        if claim_held == int32(0):
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), node_limit)
            cuda.syncwarp(FULL_MASK)
            break
        
        #* winner edge 必须指向合法 child；否则回滚已经选择的整条路径。
        best_child_valid = False
        if best_eid >= 0 or best_eid < cur_expanded:
            best_child = edge_child_id[tree, node, best_eid]
            if best_child >= 0:     # TODO. 检查小于当前最大的node数
                best_child_valid = True
        
        if best_child_valid:
            cuda.atomic.sub(node_inflight, (tree, best_child), int32(1))
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), node_limit)
            cuda.syncwarp(FULL_MASK)
            break

        #* Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
        _write_path_edge(out_path_eids, tree, wid, depth, lane, node, best_eid)
        cuda.syncwarp(FULL_MASK)

        node = best_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_node, final_len, out_selected_node, out_path_len)