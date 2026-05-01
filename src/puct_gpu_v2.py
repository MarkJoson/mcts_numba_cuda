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

# ============================================================
# Data structure documentation
# ============================================================

class Tree:
    pass

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


# ============================================================
# Device array allocation
# ============================================================

def init_data():
    TREE_CNT = 16
    MAX_NODE = 16384        # 最大16K个节点
    MAX_ACTION = 256        # 每个节点最大256个动作
    MAX_WARPS_PER_TREE = 8
    MAX_PATH_DEPTH = 64

    TREE_SHAPE = (TREE_CNT,)
    trees_selected = cuda.device_array(TREE_SHAPE, np.int32)    # type: ignore

    ROBOT_TURNS = 2

    NODE_SHAPE = (TREE_CNT, MAX_NODE)
    nodes_state = cuda.device_array(NODE_SHAPE, np.int32)       # type: ignore
    nodes_expanded = cuda.device_array(NODE_SHAPE, np.int32)    # type: ignore  # >=0 expanded count, -1 terminal
    nodes_inflight = cuda.device_array(NODE_SHAPE, np.int32)    # type: ignore  # virtual loss count per child node
    nodes_expand_inflight = cuda.device_array(NODE_SHAPE, np.int32)  # type: ignore  # expand tickets per parent node
    nodes_value = cuda.device_array((TREE_CNT, MAX_NODE, ROBOT_TURNS), np.int32)  # type: ignore

    EDGE_SHAPE = (TREE_CNT, MAX_NODE, MAX_ACTION)
    edges_child_id = cuda.device_array(EDGE_SHAPE, np.int32)    # type: ignore
    edges_n = cuda.device_array(EDGE_SHAPE, np.int32)           # type: ignore  # visit count
    edges_W = cuda.device_array(EDGE_SHAPE, np.float32)         # type: ignore  # cumulative value
    edges_prob = cuda.device_array(EDGE_SHAPE, np.float32)      # type: ignore
    edges_prior = cuda.device_array(EDGE_SHAPE, np.float32)     # type: ignore  # PUCT prior

    # Output arrays for selection kernel
    out_shape = (TREE_CNT, MAX_WARPS_PER_TREE)
    out_selected_node = cuda.device_array(out_shape, np.int32)  # type: ignore
    out_path_eids = cuda.device_array((TREE_CNT, MAX_WARPS_PER_TREE, MAX_PATH_DEPTH), np.int32)   # type: ignore
    out_path_len = cuda.device_array(out_shape, np.int32)       # type: ignore

    # Value buffer for backup kernel (per-edge value along path)
    path_edge_values = cuda.device_array((TREE_CNT, MAX_WARPS_PER_TREE, MAX_PATH_DEPTH), np.float32)  # type: ignore

# ============================================================
# Device helpers
# ============================================================

@cuda.jit(device=True, inline=True)
def _score_better_rotated(score, eid, best_score, best_eid, rotation, span):
    """
    Score comparison with a warp-specific rotated tie-break.
    Used only by the experimental preclaim selector to avoid all warps
    resolving perfectly equal scores to edge 0.
    """
    if score > best_score:
        return True
    if score == best_score:
        if best_eid == int32(INT32_MAX):
            return True
        if span <= 0:
            return eid < best_eid
        eid_key = eid - rotation
        if eid_key < 0:
            eid_key += span
        best_key = best_eid - rotation
        if best_key < 0:
            best_key += span
        return eid_key < best_key
    return False


@cuda.jit(device=True, inline=True)
def _warp_reduce_best_eid_rotated(score, eid, rotation, span):
    """
    Warp-level reduce-max over (score, eid) only. The winning child is read
    after reduction, which trims one value from every shuffle on the hot path.
    """
    other_score = cuda.shfl_down_sync(FULL_MASK, score, 16)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 16)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 8)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 8)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 4)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 4)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 2)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 2)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 1)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 1)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid

    score = cuda.shfl_sync(FULL_MASK, score, 0)
    eid = cuda.shfl_sync(FULL_MASK, eid, 0)
    return score, eid


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
            child = edge_child_id[tree, parent, slot]
            if child >= int32(0) and child < n_nodes:
                cuda.atomic.sub(node_inflight, (tree, child), virtual_loss)
        d += int32(WARP_SIZE)


@cuda.jit(device=True, inline=True)
def _claim_vloss_counter_cas(node_inflight, tree, node):
    """
    Claim one virtual visit with compare-and-swap on the counter value:
        old -> old + 1

    This is deliberately not a 0/1 lock. The returned conflict bit records
    whether the child node already had inflight visits before this warp's claim,
    while the counter still supports multiple simultaneous holders.
    """
    held = int32(0)
    conflict = int32(0)
    attempts = int32(0)
    can_claim = int32(1)

    while held == int32(0) and attempts < int32(MAX_SELECT_RETRY):
        old = node_inflight[tree, node]
        if old < int32(0) or old >= int32(INT32_MAX):
            can_claim = int32(0)
            break
        prev = cuda.atomic.cas(node_inflight, (tree, node), old, old + int32(1))
        if prev == old:
            held = int32(1)
            if old > int32(0):
                conflict = int32(1)
        attempts += int32(1)

    # Bounded CAS avoids spinning forever under heavy contention. If the edge
    # still looks claimable after the retry budget, fall back to atomic add so
    # a valid traversal is not converted into SELECT_INVALID by CAS churn alone.
    if held == int32(0) and can_claim != int32(0):
        old = cuda.atomic.add(node_inflight, (tree, node), int32(1))
        held = int32(1)
        if old > int32(0):
            conflict = int32(1)

    return held, conflict


@cuda.jit(device=True, inline=True)
def _pack_selection(kind, node, expand_slot):
    return (kind << int32(PACKED_KIND_SHIFT)) | (expand_slot << int32(PACKED_SLOT_SHIFT)) | node


@cuda.jit(device=True, inline=True)
def _valid_select_shape(max_edge_steps, n_nodes, max_edges):
    return (
        max_edge_steps >= int32(1) and
        n_nodes > int32(0) and n_nodes <= int32(PACKED_NODE_LIMIT) and
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
def _best_edge_readonly(cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
                        tree, node, cur_expanded, n_nodes, lane,
                        sqrt_parent_n_eff, rotation, skip_eid):
    best_score = float32(NEG_INF_F32)
    best_eid = int32(INT32_MAX)

    eid = lane
    while eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
        if int32(eid) != skip_eid and child >= int32(0) and child < n_nodes:
            prior = edge_prior[tree, node, eid]
            w = edge_W[tree, node, eid]
            n_edge = edge_N[tree, node, eid]
            inflight = node_inflight[tree, child]
            score = _puct_score(cpuct, prior, w, n_edge, inflight, sqrt_parent_n_eff)
            if _score_better_rotated(score, int32(eid), best_score, best_eid, rotation, cur_expanded):
                best_score = score
                best_eid = int32(eid)
        eid += int32(WARP_SIZE)

    best_score, best_eid = _warp_reduce_best_eid_rotated(
        best_score, best_eid, rotation, cur_expanded,
    )
    return best_eid


@cuda.jit(device=True, inline=True)
def _best_edge_preclaim(cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
                        tree, node, cur_expanded, n_nodes, lane,
                        sqrt_parent_n_eff, rotation):
    best_score = float32(NEG_INF_F32)
    best_eid = int32(INT32_MAX)

    eid = lane
    while eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
        if child >= int32(0) and child < n_nodes:
            prior = edge_prior[tree, node, eid]
            w = edge_W[tree, node, eid]
            n_edge = edge_N[tree, node, eid]
            old_inflight = cuda.atomic.add(node_inflight, (tree, child), int32(1))
            score = _puct_score(cpuct, prior, w, n_edge, old_inflight, sqrt_parent_n_eff)
            if _score_better_rotated(score, int32(eid), best_score, best_eid, rotation, cur_expanded):
                best_score = score
                best_eid = int32(eid)
        eid += int32(WARP_SIZE)

    best_score, best_eid = _warp_reduce_best_eid_rotated(
        best_score, best_eid, rotation, cur_expanded,
    )
    return best_eid


@cuda.jit(device=True, inline=True)
def _rollback_preclaim_losers(edge_child_id, node_inflight,
                              tree, node, cur_expanded, n_nodes, lane, best_eid):
    eid = lane
    while eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
        if child >= int32(0) and child < n_nodes and int32(eid) != best_eid:
            cuda.atomic.sub(node_inflight, (tree, child), int32(1))
        eid += int32(WARP_SIZE)


@cuda.jit(device=True, inline=True)
def _child_for_edge(edge_child_id, tree, node, eid, cur_expanded, n_nodes):
    child = int32(-1)
    if eid >= int32(0) and eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
    if child < int32(0) or child >= n_nodes:
        return int32(-1)
    return child


@cuda.jit(device=True, inline=True)
def _release_claim_if_held(edge_child_id, node_inflight,
                           tree, node, eid, cur_expanded, n_nodes, lane, claim_held):
    if lane == int32(0) and claim_held != int32(0) and eid >= int32(0) and eid < cur_expanded:
        child = edge_child_id[tree, node, eid]
        if child >= int32(0) and child < n_nodes:
            cuda.atomic.sub(node_inflight, (tree, child), int32(1))


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
    int32[:, :], int32[:, :], int32[:, :],
    int32[:, :], int32[:, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_native(cpuct, c_pw, alpha_pw,
                          edge_child_id, edge_prior, edge_W, edge_N,
                          node_inflight, node_expand_inflight, node_expanded,
                          out_selected_node, out_path_eids, out_path_len):
    """
    每个 warp 独立遍历树，选出待扩展节点。1 block = 1 tree; 1 warp = 1 条遍历路径。

    输出：
      out_selected_node[tree, wid]  — SELECT_EXPAND / SELECT_TERMINAL / SELECT_DEPTH_LIMIT / SELECT_INVALID + 被选中的节点 id
      out_path_eids[tree, wid, :]   — 路径上的边 id
      out_path_len[tree, wid]       — 路径上的节点数
    """
    tree = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    lane = tid & int32(31)
    wid = tid >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    warp_count = cuda.blockDim.x >> int32(5)
    active_warps = min(warp_count, out_selected_node.shape[1])
    if wid >= active_warps:
        return

    _init_select_output(tree, wid, lane, out_selected_node, out_path_len)

    max_edges = edge_child_id.shape[2]
    max_edge_steps = out_path_eids.shape[2]
    n_nodes = node_expanded.shape[1]

    if not _valid_select_shape(max_edge_steps, n_nodes, max_edges):
        return

    final_node = int32(SEL_IN_PROGRESS)
    final_len = int32(0)

    node = int32(0)      # start from root
    depth = int32(0)
    while True:
        # 防御性编程: 节点的id值是无效值，或者大于最大节点数，认为是无效节点
        if node < 0 or node >= n_nodes:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break

        node_info = node_expanded[tree, node]

        #* 终止态的节点。node_expanded 使用负数 sentinel 表示 terminal。需要考虑用 Value 网络重新估计节点价值
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_node = _pack_selection(int32(SELECT_TERMINAL), node, int32(0))
            final_len = depth + int32(1)
            break

        if node_info < int32(0) or node_info > max_edges:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break
        
        #* 计算当前允许的节点数量 
        cur_expanded = node_info
        parent_n_eff = _parent_n_eff(
            edge_child_id, edge_N, node_inflight,
            tree, node, cur_expanded, n_nodes, lane,
        )
        allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges) if cur_expanded != 0 else 1
        need_expand = allowed - cur_expanded

        #* 渐进增长树：用 node_expand_inflight ticket 给同一父节点分配不同 expand slot。
        if need_expand > 0:
            expand_slot = _claim_expand_slot(
                node_expand_inflight, tree, node, cur_expanded, allowed, lane,
            )
            if expand_slot >= int32(0):
                final_node = _pack_selection(int32(SELECT_EXPAND), node, expand_slot)
                final_len = depth + int32(1)
            else:   #! 扩展数量不允许进一步拓展，则退出
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
                cuda.syncwarp(FULL_MASK)
            break

        # 达到最大步长时，当前节点不再扩展，为叶子节点
        if depth >= max_edge_steps:
            final_node = _pack_selection(int32(SELECT_DEPTH_LIMIT), node, int32(0))
            final_len = depth + int32(1)
            break

        #* 当渐宽树在完整扩展的情况下，会在已有的孩子边中选择一个最优动作的进行扩展
        #! 存在一种情况是多个线程同时进入了一个node。因此需要在计算 value 时引入虚拟损失
        sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))
        rotation = wid % cur_expanded
        best_eid = _best_edge_preclaim(
            cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
            tree, node, cur_expanded, n_nodes, lane, sqrt_parent_n_eff, rotation,
        )

        #* 撤销本次未被选中的节点价值
        _rollback_preclaim_losers(
            edge_child_id, node_inflight,
            tree, node, cur_expanded, n_nodes, lane, best_eid,
        )

        best_child = _child_for_edge(edge_child_id, tree, node, best_eid, cur_expanded, n_nodes)

        #* 没有节点被选中的情况比较特殊，在这种情况下一般是孩子边的价值出了问题
        if best_child < int32(0):
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break
        
        #* Path的个格式为: [(父节点id, 边id), (父节点id, 边id), (父节点id, 边id), ...] 因为需要在扩展时同时对 节点数据 和 边数据 进行修改
        _write_path_edge(out_path_eids, tree, wid, depth, lane, node, best_eid)
        cuda.syncwarp(FULL_MASK)

        node = best_child
        depth += int32(1)
    _write_select_output(tree, wid, lane, final_node, final_len, out_selected_node, out_path_len)


@cuda.jit(void(
    float32, float32, float32,
    int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :],
    int32[:, :], int32[:, :], int32[:, :],
    int32[:, :], int32[:, :, :], int32[:, :]),
    fastmath=True)
def _select_kernel_winner_recalc(cpuct, c_pw, alpha_pw,
                                 edge_child_id, edge_prior, edge_W, edge_N,
                                 node_inflight, node_expand_inflight, node_expanded,
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
    tid = cuda.threadIdx.x
    lane = tid & int32(31)
    wid = tid >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    warp_count = cuda.blockDim.x >> int32(5)
    active_warps = min(warp_count, out_selected_node.shape[1])
    if wid >= active_warps:
        return

    _init_select_output(tree, wid, lane, out_selected_node, out_path_len)

    max_edges = edge_child_id.shape[2]
    max_edge_steps = out_path_eids.shape[2]
    n_nodes = node_expanded.shape[1]

    if not _valid_select_shape(max_edge_steps, n_nodes, max_edges):
        return

    final_node = int32(SEL_IN_PROGRESS)
    final_len = int32(0)
    node = int32(0)
    depth = int32(0)

    while True:
        # 防御性编程：路径中出现非法 node id 时，回滚已经保留的 winner virtual loss。
        if node < 0 or node >= n_nodes:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break

        node_info = node_expanded[tree, node]

        # 终止态节点。node_expanded 使用负数 sentinel 表示 terminal。
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_node = _pack_selection(int32(SELECT_TERMINAL), node, int32(0))
            final_len = depth + int32(1)
            break

        # 非 terminal 的负数或超过动作数的 expanded 计数都是非法状态。
        if node_info < int32(0) or node_info > max_edges:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break

        cur_expanded = node_info

        # 尚未展开的非终止叶子节点：尝试 claim slot 0 作为 expand 任务。
        # 只有抢到 CAS 的 warp 会返回 SELECT_EXPAND；失败者回滚路径并返回 invalid。
        if cur_expanded == int32(0):
            expand_slot = _claim_expand_slot(
                node_expand_inflight, tree, node, int32(0), int32(1), lane,
            )
            if expand_slot >= int32(0):
                final_node = _pack_selection(int32(SELECT_EXPAND), node, expand_slot)
                final_len = depth + int32(1)
            else:
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
                cuda.syncwarp(FULL_MASK)
            break

        parent_n_eff = _parent_n_eff(
            edge_child_id, edge_N, node_inflight,
            tree, node, cur_expanded, n_nodes, lane,
        )

        allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)
        need_expand = allowed - cur_expanded

        # Progressive widening: 当前节点允许展开更多 child 时，lane0 顺序抢占
        # [cur_expanded, allowed) 中的第一个空 slot。
        if need_expand > 0:
            expand_slot = _claim_expand_slot(
                node_expand_inflight, tree, node, cur_expanded, allowed, lane,
            )
            if expand_slot >= int32(0):
                final_node = _pack_selection(int32(SELECT_EXPAND), node, expand_slot)
                final_len = depth + int32(1)
            else:
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
                cuda.syncwarp(FULL_MASK)
            break

        # 达到 path buffer 上限时，不再继续向下选择，交给上层按 depth-limit 叶子处理。
        if depth >= max_edge_steps:
            final_node = _pack_selection(int32(SELECT_DEPTH_LIMIT), node, int32(0))
            final_len = depth + int32(1)
            break

        # 在已完全展开的孩子边中选择 PUCT 最优边。
        # 与 native 不同：这里评分前不对所有候选做 atomic add，只读取当前 inflight。
        sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))
        rotation = wid % cur_expanded
        best_eid = _best_edge_readonly(
            cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
            tree, node, cur_expanded, n_nodes, lane,
            sqrt_parent_n_eff, rotation, int32(-1),
        )

        # 只在 winner 边上保留 virtual loss。这里使用 counter-CAS：
        #   node_inflight[child]: old -> old + 1
        # 它不是 0/1 锁，因此多个 warp 仍可合法持有同一边的 virtual loss；
        # 同时 old > 0 给了我们一个轻量冲突信号，可触发局部重算。
        original_eid = best_eid
        claim_conflict = int32(0)
        claim_held = int32(0)
        if lane == 0 and best_eid >= int32(0) and best_eid < cur_expanded:
            best_child_for_claim = edge_child_id[tree, node, best_eid]
            if best_child_for_claim >= int32(0) and best_child_for_claim < n_nodes:
                claim_held, claim_conflict = _claim_vloss_counter_cas(node_inflight, tree, best_child_for_claim)
        claim_conflict = cuda.shfl_sync(FULL_MASK, claim_conflict, 0)
        claim_held = cuda.shfl_sync(FULL_MASK, claim_held, 0)

        # 冲突处理：本 warp 已经持有原 winner 的一份 virtual loss。为了
        # 重算时不把自己的 claim 也算作竞争压力，先释放它，再排除原 winner
        # 重新计算一次 PUCT。
        # 这是 winner_recalc 接近 CPU batch reference 的关键：后续选择会感知
        # 已经 claim 的边，并主动避开本轮冲突 winner。
        if claim_conflict != int32(0) and cur_expanded > int32(1):
            _release_claim_if_held(
                edge_child_id, node_inflight,
                tree, node, original_eid, cur_expanded, n_nodes, lane, claim_held,
            )
            cuda.syncwarp(FULL_MASK)

            # 重算时不再 claim 所有候选，只读取当前 inflight 并跳过 original_eid。
            best_eid = _best_edge_readonly(
                cpuct, edge_child_id, edge_prior, edge_W, edge_N, node_inflight,
                tree, node, cur_expanded, n_nodes, lane,
                sqrt_parent_n_eff, rotation, original_eid,
            )

            # 如果没有其它合法 child，duplicate 原 winner 是唯一可行路径；
            # same-path 不是非法状态，只是不应在宽节点高概率必然发生。
            if best_eid == int32(INT32_MAX):
                best_eid = original_eid

            # lane0 claim 重算后的 winner。counter-CAS 保留计数语义，仍可
            # 合法与其它 warp 共享同一边；只有 bounded CAS loop 没抢到时
            # 才返回 invalid 并回滚此前路径。
            claim_held = int32(0)
            if lane == 0:
                if best_eid >= int32(0) and best_eid < cur_expanded:
                    best_child_for_claim = edge_child_id[tree, node, best_eid]
                    if best_child_for_claim >= int32(0) and best_child_for_claim < n_nodes:
                        claim_held, claim_conflict = _claim_vloss_counter_cas(node_inflight, tree, best_child_for_claim)
            best_eid = cuda.shfl_sync(FULL_MASK, best_eid, 0)
            claim_held = cuda.shfl_sync(FULL_MASK, claim_held, 0)

        if claim_held == int32(0):
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break

        # winner edge 必须指向合法 child；否则回滚已经选择的整条路径。
        best_child = _child_for_edge(edge_child_id, tree, node, best_eid, cur_expanded, n_nodes)

        if best_child < int32(0):
            _release_claim_if_held(
                edge_child_id, node_inflight,
                tree, node, best_eid, cur_expanded, n_nodes, lane, claim_held,
            )
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_child_id, node_inflight, int32(1), n_nodes)
            cuda.syncwarp(FULL_MASK)
            break

        # Path 编码格式为 (parent_node << 8) | edge_slot。backup 时直接解码。
        _write_path_edge(out_path_eids, tree, wid, depth, lane, node, best_eid)
        cuda.syncwarp(FULL_MASK)

        node = best_child
        depth += int32(1)

    _write_select_output(tree, wid, lane, final_node, final_len, out_selected_node, out_path_len)


# ============================================================
# Backup kernel
# ============================================================

@cuda.jit(void(int32,
                   int32[:, :, :], float32[:, :, :], int32[:, :, :],
                   int32[:, :], int32[:, :],
                   int32[:, :],
                   int32[:, :, :], int32[:, :],
                   float32[:, :, :]),
          fastmath=True)
def _backup_kernel_native(virtual_loss,
                          edge_child_id, edge_W, edge_N,
                          node_inflight, node_expand_inflight,
                          out_selected_node,
                          out_path_eids, out_path_len,
                          path_edge_values):
    """
    完成 backup：将 virtual visit 转换为真实统计。

    Selection 阶段：
      node_inflight[child] += virtual_loss       （已存在 child 的虚拟访问）
      node_expand_inflight[parent] += 1          （待扩展 slot 的 ticket）

    Backup 阶段完成转换：
      node_inflight -= virtual_loss       （撤销虚拟）
      edge_N += 1                          （写入真实）
      edge_W += value                      （写入真实累积价值，无需 W 偏移）

    out_selected_node 编码: (kind << 22) | (expand_slot << 14) | node_id
    每个 eid 编码为 (parent << 8) | slot，直接解码即可。
    path_edge_values[tree, wid, d] 必须是已转换好视角的 value。
    """
    tree = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    lane = tid & int32(31)
    wid = tid >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    active_warps = min(cuda.blockDim.x >> int32(5), out_selected_node.shape[1])
    if wid >= active_warps:
        return

    raw_selected = out_selected_node[tree, wid]
    if raw_selected < int32(0):
        return

    kind = raw_selected >> int32(PACKED_KIND_SHIFT)
    if kind == int32(SELECT_INVALID):
        return

    plen = out_path_len[tree, wid]
    if plen <= 0:
        return

    edge_count = plen - int32(1)

    # Each lane handles edges at stride=32
    d = lane
    while d < edge_count:
        encoded = out_path_eids[tree, wid, d]
        if encoded >= 0:
            parent = encoded >> int32(8)
            slot = encoded & int32(0xFF)
            child = edge_child_id[tree, parent, slot]
            v = path_edge_values[tree, wid, d]
            if child >= int32(0) and child < node_inflight.shape[1]:
                cuda.atomic.sub(node_inflight, (tree, child), virtual_loss)
            cuda.atomic.add(edge_N, (tree, parent, slot), int32(1))
            cuda.atomic.add(edge_W, (tree, parent, slot), v)
        d += int32(WARP_SIZE)

    # Release expansion ticket on the selected leaf node.
    if lane == 0 and kind == int32(SELECT_EXPAND):
        leaf = raw_selected & int32(PACKED_NODE_MASK)
        if leaf >= 0:
            cuda.atomic.sub(node_expand_inflight, (tree, leaf), int32(1))
