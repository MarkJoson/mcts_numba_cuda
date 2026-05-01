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

NODE_STATUS_OK = 0
NODE_STATUS_TERMINAL = 1

# Selection result kinds
SELECT_INVALID = 0      # No valid selection; virtual loss already rolled back
SELECT_EXPAND = 1       # Node needs expansion (PW not saturated); CAS lock held
SELECT_TERMINAL = 2     # Traversal hit a terminal node
SELECT_DEPTH_LIMIT = 3  # Hit path capacity limit; treat as evaluation leaf

SEL_IN_PROGRESS = -2    # final_node: selection still in progress (replaces done flag)
SEL_INVALID_NODE = -1   # final_node / output: no valid node selected

MAX_SELECT_RETRY = 3

# ============================================================
# Data structure documentation
# ============================================================

class Tree:
    pass

class Traverse:
    path: list          # 记录遍历的路径

# 由[tree_id, node_id]索引
class Node:
    status: int         # 节点状态: OK / Terminal
    expanded: int       # 已经展开的 edge 个数。如果 =0 说明是叶子节点, 由于是渐宽树，所以会逐步展开
    prepare_expand: int # 预计展开的节点数量
    state: int          # 对应的状态空间，指向状态池
    value: float        # 对应当前状态的状态价值

# 由[tree_id, edge_id]索引
# edge_id = (node_id << 8) | slot   (8-bit slot for MAX_ACTION=256)
class Edge:
    child_id: int       # -1: Invalid, 否则是正常的node id
    prior: float        # 先验概率 (用于PUCT公式)
    N: int              # 访问次数 (真实，只在 backup 时增加)
    inflight: int       # 虚拟访问计数 (selection 时 +1，rollback 或 backup 时 -1)
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
    nodes_status = cuda.device_array(NODE_SHAPE, np.int32)      # type: ignore
    nodes_state = cuda.device_array(NODE_SHAPE, np.int32)       # type: ignore
    nodes_expanded = cuda.device_array(NODE_SHAPE, np.int32)    # type: ignore  # expanded child count
    nodes_value = cuda.device_array((TREE_CNT, MAX_NODE, ROBOT_TURNS), np.int32)  # type: ignore

    EDGE_SHAPE = (TREE_CNT, MAX_NODE, MAX_ACTION)
    edges_child_id = cuda.device_array(EDGE_SHAPE, np.int32)    # type: ignore
    edges_n = cuda.device_array(EDGE_SHAPE, np.int32)           # type: ignore  # visit count
    edges_inflight = cuda.device_array(EDGE_SHAPE, np.int32)    # type: ignore  # virtual loss count
    edges_W = cuda.device_array(EDGE_SHAPE, np.float32)         # type: ignore  # cumulative value
    edges_prob = cuda.device_array(EDGE_SHAPE, np.float32)      # type: ignore
    edges_prior = cuda.device_array(EDGE_SHAPE, np.float32)     # type: ignore  # PUCT prior

    # Output arrays for selection kernel
    out_shape = (TREE_CNT, MAX_WARPS_PER_TREE)
    out_selected_node = cuda.device_array(out_shape, np.int32)  # type: ignore
    out_selected_kind = cuda.device_array(out_shape, np.int32)  # type: ignore
    out_path_eids = cuda.device_array((TREE_CNT, MAX_WARPS_PER_TREE, MAX_PATH_DEPTH), np.int32)   # type: ignore
    out_path_len = cuda.device_array(out_shape, np.int32)       # type: ignore

    # Value buffer for backup kernel (per-edge value along path)
    path_edge_values = cuda.device_array((TREE_CNT, MAX_WARPS_PER_TREE, MAX_PATH_DEPTH), np.float32)  # type: ignore

# ============================================================
# Device helpers
# ============================================================

@cuda.jit(device=True, inline=True)
def _score_better(score, eid, best_score, best_eid):
    """
    Deterministic best-score comparison:
      1. Larger score wins
      2. On tie, smaller eid wins (deterministic)
    """
    if score > best_score:
        return True
    if score == best_score and eid < best_eid:
        return True
    return False


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
def _warp_reduce_best(score, eid, child):
    """
    Warp-level reduce-max over (score, eid, child).
    Tie-breaking: larger score wins; on tie, smaller eid wins.
    Broadcasts lane 0 winner to all lanes so every lane sees the same result.
    """
    other_score = cuda.shfl_down_sync(FULL_MASK, score, 16)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 16)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 16)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 8)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 8)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 8)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 4)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 4)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 4)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 2)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 2)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 2)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 1)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 1)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 1)
    if _score_better(other_score, other_eid, score, eid):
        score = other_score
        eid = other_eid
        child = other_child

    # Broadcast lane 0 winner to all lanes — critical for subsequent node assignment
    score = cuda.shfl_sync(FULL_MASK, score, 0)
    eid = cuda.shfl_sync(FULL_MASK, eid, 0)
    child = cuda.shfl_sync(FULL_MASK, child, 0)

    return score, eid, child


@cuda.jit(device=True, inline=True)
def _warp_reduce_best_rotated(score, eid, child, rotation, span):
    """
    Warp-level reduce-max with a rotated tie-break. This is intentionally
    separate from _warp_reduce_best so the production selector keeps its
    deterministic smaller-eid tie-break.
    """
    other_score = cuda.shfl_down_sync(FULL_MASK, score, 16)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 16)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 16)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 8)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 8)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 8)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 4)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 4)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 4)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 2)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 2)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 2)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid
        child = other_child

    other_score = cuda.shfl_down_sync(FULL_MASK, score, 1)
    other_eid = cuda.shfl_down_sync(FULL_MASK, eid, 1)
    other_child = cuda.shfl_down_sync(FULL_MASK, child, 1)
    if _score_better_rotated(other_score, other_eid, score, eid, rotation, span):
        score = other_score
        eid = other_eid
        child = other_child

    score = cuda.shfl_sync(FULL_MASK, score, 0)
    eid = cuda.shfl_sync(FULL_MASK, eid, 0)
    child = cuda.shfl_sync(FULL_MASK, child, 0)

    return score, eid, child


@cuda.jit(device=True, inline=True)
def _top2_insert(score, eid, inflight,
                 best_score, best_eid, best_inflight,
                 alt_score, alt_eid):
    """
    Insert one candidate into a deterministic top-2 score list.
    Duplicate eids are ignored so reduction merges do not report the same
    edge as both best and runner-up.
    """
    if eid == int32(INT32_MAX):
        return best_score, best_eid, best_inflight, alt_score, alt_eid

    if eid == best_eid:
        if _score_better(score, eid, best_score, best_eid):
            best_score = score
            best_inflight = inflight
        return best_score, best_eid, best_inflight, alt_score, alt_eid

    if eid == alt_eid:
        if _score_better(score, eid, alt_score, alt_eid):
            alt_score = score
        return best_score, best_eid, best_inflight, alt_score, alt_eid

    if _score_better(score, eid, best_score, best_eid):
        alt_score = best_score
        alt_eid = best_eid
        best_score = score
        best_eid = eid
        best_inflight = inflight
    elif _score_better(score, eid, alt_score, alt_eid):
        alt_score = score
        alt_eid = eid

    return best_score, best_eid, best_inflight, alt_score, alt_eid


@cuda.jit(device=True, inline=True)
def _warp_reduce_top2(best_score, best_eid, best_inflight,
                      alt_score, alt_eid):
    """
    Warp-level top-2 reduce. Broadcasts lane 0's best and runner-up to all
    lanes so lane 0 can claim best while every lane agrees on retry state.
    Child ids are intentionally not shuffled; the selected child is loaded
    after the winning eid is known.
    """
    other_best_score = cuda.shfl_down_sync(FULL_MASK, best_score, 16)
    other_best_eid = cuda.shfl_down_sync(FULL_MASK, best_eid, 16)
    other_best_inflight = cuda.shfl_down_sync(FULL_MASK, best_inflight, 16)
    other_alt_score = cuda.shfl_down_sync(FULL_MASK, alt_score, 16)
    other_alt_eid = cuda.shfl_down_sync(FULL_MASK, alt_eid, 16)
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_best_score, other_best_eid, other_best_inflight,
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_alt_score, other_alt_eid, int32(0),
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )

    other_best_score = cuda.shfl_down_sync(FULL_MASK, best_score, 8)
    other_best_eid = cuda.shfl_down_sync(FULL_MASK, best_eid, 8)
    other_best_inflight = cuda.shfl_down_sync(FULL_MASK, best_inflight, 8)
    other_alt_score = cuda.shfl_down_sync(FULL_MASK, alt_score, 8)
    other_alt_eid = cuda.shfl_down_sync(FULL_MASK, alt_eid, 8)
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_best_score, other_best_eid, other_best_inflight,
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_alt_score, other_alt_eid, int32(0),
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )

    other_best_score = cuda.shfl_down_sync(FULL_MASK, best_score, 4)
    other_best_eid = cuda.shfl_down_sync(FULL_MASK, best_eid, 4)
    other_best_inflight = cuda.shfl_down_sync(FULL_MASK, best_inflight, 4)
    other_alt_score = cuda.shfl_down_sync(FULL_MASK, alt_score, 4)
    other_alt_eid = cuda.shfl_down_sync(FULL_MASK, alt_eid, 4)
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_best_score, other_best_eid, other_best_inflight,
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_alt_score, other_alt_eid, int32(0),
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )

    other_best_score = cuda.shfl_down_sync(FULL_MASK, best_score, 2)
    other_best_eid = cuda.shfl_down_sync(FULL_MASK, best_eid, 2)
    other_best_inflight = cuda.shfl_down_sync(FULL_MASK, best_inflight, 2)
    other_alt_score = cuda.shfl_down_sync(FULL_MASK, alt_score, 2)
    other_alt_eid = cuda.shfl_down_sync(FULL_MASK, alt_eid, 2)
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_best_score, other_best_eid, other_best_inflight,
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_alt_score, other_alt_eid, int32(0),
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )

    other_best_score = cuda.shfl_down_sync(FULL_MASK, best_score, 1)
    other_best_eid = cuda.shfl_down_sync(FULL_MASK, best_eid, 1)
    other_best_inflight = cuda.shfl_down_sync(FULL_MASK, best_inflight, 1)
    other_alt_score = cuda.shfl_down_sync(FULL_MASK, alt_score, 1)
    other_alt_eid = cuda.shfl_down_sync(FULL_MASK, alt_eid, 1)
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_best_score, other_best_eid, other_best_inflight,
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )
    best_score, best_eid, best_inflight, alt_score, alt_eid = _top2_insert(
        other_alt_score, other_alt_eid, int32(0),
        best_score, best_eid, best_inflight, alt_score, alt_eid,
    )

    best_score = cuda.shfl_sync(FULL_MASK, best_score, 0)
    best_eid = cuda.shfl_sync(FULL_MASK, best_eid, 0)
    best_inflight = cuda.shfl_sync(FULL_MASK, best_inflight, 0)
    alt_score = cuda.shfl_sync(FULL_MASK, alt_score, 0)
    alt_eid = cuda.shfl_sync(FULL_MASK, alt_eid, 0)

    return best_score, best_eid, best_inflight, alt_score, alt_eid


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
    if max_edges <= 0:
        return int32(0)

    n = n_node
    if n < 1:
        n = int32(1)

    val = c_pw * math.pow(float32(n), alpha_pw)

    if val >= float32(max_edges):
        return max_edges

    k = int32(math.ceil(val))
    if k < 1:
        k = int32(1)
    if k > max_edges:
        k = max_edges

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
                          out_path_eids, edge_inflight, virtual_loss):
    """
    Roll back virtual visits on edge_inflight for the current attempt.
    Only int32 atomics on edge_inflight — no W modification during selection.
    Each eid is encoded as (parent << 8) | slot — extract directly.
    Caller must syncwarp after return.
    """
    d = lane
    while d < depth:
        encoded = out_path_eids[tree, wid, d]
        if encoded >= 0:
            parent = encoded >> int32(8)
            slot = encoded & int32(0xFF)
            cuda.atomic.sub(edge_inflight, (tree, parent, slot), virtual_loss)
        d += int32(WARP_SIZE)


# ============================================================
# Selection kernel
# ============================================================

@cuda.jit(void(float32, float32, float32, int32,
                   int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :], int32[:, :, :],
                   int32[:, :], int32[:, :],
                   int32[:, :], int32[:, :], int32[:, :, :], int32[:, :]))
def _select_kernel_native(cpuct, c_pw, alpha_pw,
                          edge_child_id, edge_prior, edge_W, edge_N, edge_inflight,
                          node_status, node_expanded, node_need_expand,
                          out_selected_node, out_selected_kind,
                          out_path_eids, out_path_len):
    """
    每个 warp 独立遍历树，选出待扩展节点。

    1 block = 1 tree; 1 warp = 1 条遍历路径。
    一次扫描拿到 best / runner-up, 只对 best edge 做 atomic claim。
    如果 atomic 返回的 old inflight 与扫描时一致，直接接受；否则
    用 runner-up 做冲突校验，必要时释放并重选。
    CAS-based expansion claim 避免重复扩展。

    node_N / out_path_nodes 已删除：
      - parent_n_eff 由 warp-level sum over edge_N + edge_inflight 实时计算
      - 节点路径可通过 out_path_eids + edge_child_id 从 root 重建

    输出：
      out_selected_node[tree, wid]  — 被选中的节点 id
      out_selected_kind[tree, wid]  — SELECT_EXPAND / SELECT_TERMINAL / SELECT_DEPTH_LIMIT / SELECT_INVALID
      out_path_eids[tree, wid, :]   — 路径上的边 id
      out_path_len[tree, wid]       — 路径上的节点数
    """
    tree = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    lane = tid & int32(31)
    wid = tid >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    # 防止输出数组大小不够
    warp_count = cuda.blockDim.x >> int32(5)
    active_warps = min(warp_count, out_selected_node.shape[1])
    if wid >= active_warps:
        return

    # lane0 负责将输出数组初始化
    if lane == 0:
        out_selected_node[tree, wid] = int32(SEL_INVALID_NODE)
        out_selected_kind[tree, wid] = int32(SELECT_INVALID)
        out_path_len[tree, wid] = int32(0)

    max_edges = edge_child_id.shape[2]
    max_edge_steps = out_path_eids.shape[2]
    n_nodes = node_status.shape[1]
    assert max_edge_steps >= 1
    assert n_nodes > 0

    # 对 final_node 进行了压缩，包含了 当前结果类型(31-22) + expand_slot(21-14) + node_id(0-13)
    #! 当节点状态是待扩展时，expand_slot指示了新的扩展节点的起始位置。
    # TODO: 认为每一个 final_node 是对应一次扩展？还是写明需要扩展节点的数量？
    final_node = int32(SEL_IN_PROGRESS)
    final_len = int32(0)


    # TODO: 因为某些原因，单次尝试可能会失败，如多个节点同时进入扩展，此时再重试一次可能会走不同的路径。但是这样的代价值的商榷。
    node = int32(0)      # start from root
    depth = int32(0)
    while True:
        # 防御性编程: 节点的id值是无效值，或者大于最大节点数，认为是无效节点
        if node < 0 or node >= n_nodes:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1))
            cuda.syncwarp(FULL_MASK)
            break

        # 终止态的节点
        if node_status[tree, node] == NODE_STATUS_TERMINAL:
            final_node = (int32(SELECT_TERMINAL) << int32(22)) | node
            final_len = depth + int32(1)
            break
        
        # TODO: 如果是超出步长的节点，需要考虑用 Value 网络重新估计节点价值

        cur_expanded = node_expanded[tree, node]
        cur_expanded = max(0, min(max_edges, cur_expanded))     # 约束当前节点范围

        # 聚合所有边的访问次数，得到节点的访问次数，用于
        parent_n_eff = int32(0)
        eid = lane
        while eid < cur_expanded:
            child = edge_child_id[tree, node, eid]
            assert child >= 0 and child < n_nodes
            parent_n_eff += edge_N[tree, node, eid] + edge_inflight[tree, node, eid]    # TODO. 这里的 inflight 也存在共同访问时没有自增的问题
            eid += int32(WARP_SIZE)
        parent_n_eff = max(_warp_reduce_sum(parent_n_eff), int32(1))

        allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)
        need_expand = allowed - cur_expanded

        # 渐进增长树，多个 warp 会同时对 edge_inflight 进行原子加法。如果是数量上未满足待扩展需求节点数量，即作为扩展节点
        if need_expand > 0:
            expand_offset = int32(-1)
            if lane == 0:
                expand_offset = cuda.atomic.add(edge_inflight, (tree, node, cur_expanded), int32(1))
                if expand_offset >= need_expand:
                    cuda.atomic.sub(edge_inflight, (tree, node, cur_expanded), int32(1))
            #! 广播当前结果，让所有 lane 进入就绪状态
            expand_offset = cuda.shfl_sync(FULL_MASK, expand_offset, 0)
            if expand_offset >= need_expand:
                final_node = (int32(SELECT_EXPAND) << int32(22)) | (cur_expanded+expand_offset << int32(14)) | node
                final_len = depth + int32(1)
            else:   # 扩展数量不允许进一步拓展，则退出
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1))
                cuda.syncwarp(FULL_MASK)
            break

        # 达到最大步长时，当前节点不再扩展，为叶子节点
        if depth >= max_edge_steps:
            final_node = (int32(SELECT_DEPTH_LIMIT) << int32(22)) | node
            final_len = depth + int32(1)
            break

        # 当渐宽树在完整扩展的情况下，会在已有的孩子边中选择一个最优动作的进行扩展
        #! 存在一种情况是多个线程同时进入了一个node。因此需要在计算 value 时引入虚拟损失。
        claimed = int32(0)
        best_eid = int32(INT32_MAX)
        best_child = int32(-1)
        select_retry = int32(0)

        while select_retry < int32(MAX_SELECT_RETRY) and claimed == int32(0):
            if select_retry != int32(0):
                # First iteration reuses the parent_n_eff already computed
                # for progressive widening. Retries refresh it after
                # releasing a contended claim.
                parent_n_eff = int32(0)
                eid = lane
                while eid < cur_expanded:
                    child = edge_child_id[tree, node, eid]
                    if child >= 0 and child < n_nodes:
                        parent_n_eff += edge_N[tree, node, eid] + edge_inflight[tree, node, eid]
                    eid += int32(WARP_SIZE)
                parent_n_eff = _warp_reduce_sum(parent_n_eff)
                if parent_n_eff < 1:
                    parent_n_eff = int32(1)
            sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))

            best_score = float32(NEG_INF_F32)
            best_eid = int32(INT32_MAX)
            best_child = int32(-1)
            best_seen_inflight = int32(0)
            alt_score = float32(NEG_INF_F32)
            alt_eid = int32(INT32_MAX)

            eid = lane
            while eid < cur_expanded:
                child = edge_child_id[tree, node, eid]
                if child >= 0 and child < n_nodes:
                    prior = edge_prior[tree, node, eid]
                    w = edge_W[tree, node, eid]
                    n_edge = edge_N[tree, node, eid]
                    n_inflight = edge_inflight[tree, node, eid]

                    score = _puct_score(cpuct, prior, w, n_edge, n_inflight, sqrt_parent_n_eff)

                    best_score, best_eid, best_seen_inflight, alt_score, alt_eid = _top2_insert(
                        score, int32(eid), n_inflight,
                        best_score, best_eid, best_seen_inflight,
                        alt_score, alt_eid,
                    )

                eid += int32(WARP_SIZE)

            best_score, best_eid, best_seen_inflight, alt_score, alt_eid = _warp_reduce_top2(
                best_score, best_eid, best_seen_inflight,
                alt_score, alt_eid,
            )

            if best_eid == int32(INT32_MAX):
                break
            best_child = edge_child_id[tree, node, best_eid]
            if best_child < 0 or best_child >= n_nodes:
                break

            claimed_old = int32(0)
            if lane == 0:
                claimed_old = cuda.atomic.add(edge_inflight, (tree, node, best_eid), virtual_loss)
            claimed_old = cuda.shfl_sync(FULL_MASK, claimed_old, 0)

            claimed_score = _puct_score(
                cpuct,
                edge_prior[tree, node, best_eid],
                edge_W[tree, node, best_eid],
                edge_N[tree, node, best_eid],
                claimed_old,
                sqrt_parent_n_eff,
            )

            if claimed_old != best_seen_inflight and _score_better(alt_score, alt_eid, claimed_score, best_eid):
                if lane == 0:
                    cuda.atomic.sub(edge_inflight, (tree, node, best_eid), virtual_loss)
                select_retry += int32(1)
                cuda.syncwarp(FULL_MASK)
            else:
                claimed = int32(1)

        # No valid child found, or contention never settled — rollback and abort this attempt
        if claimed == int32(0) or best_child < 0:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, virtual_loss)
            cuda.syncwarp(FULL_MASK)
            break

        parent = node

        if lane == 0:
            out_path_eids[tree, wid, depth] = (parent << int32(8)) | best_eid

        cuda.syncwarp(FULL_MASK)

        node = best_child
        depth += int32(1)

    if final_node == int32(SEL_IN_PROGRESS):
        attempt += int32(1)

    # Write final result (decode packed final_node)
    # Encoding: (kind << 22) | (expand_slot << 14) | node_id
    # out_selected_kind packs expand_slot in upper bits: kind | (expand_slot << 4)
    if lane == 0:
        if final_node >= 0:
            kind = final_node >> int32(22)
            expand_slot = (final_node >> int32(14)) & int32(0xFF)
            node_id = final_node & int32(0x3FFF)
            out_selected_node[tree, wid] = node_id
            out_selected_kind[tree, wid] = kind | (expand_slot << int32(4))
            out_path_len[tree, wid] = final_len
        else:
            out_selected_node[tree, wid] = int32(SEL_INVALID_NODE)
            out_selected_kind[tree, wid] = int32(SELECT_INVALID)
            out_path_len[tree, wid] = int32(0)


# @cuda.jit(void(float32, float32, float32, int32,
#                    int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :], int32[:, :, :],
#                    int32[:, :], int32[:, :],
#                    int32[:, :], int32[:, :], int32[:, :, :], int32[:, :]))
# def _select_kernel_preclaim(cpuct, c_pw, alpha_pw, virtual_loss,
#                             edge_child_id, edge_prior, edge_W, edge_N, edge_inflight,
#                             node_status, node_expanded,
#                             out_selected_node, out_selected_kind,
#                             out_path_eids, out_path_len):
#     """
#     Experimental selector for A/B comparison.

#     Fully-expanded nodes use candidate pre-claim:
#       1. each lane atomically adds virtual_loss to every candidate it scores
#       2. score uses the atomic old inflight value
#       3. warp keeps only the winning edge's claim and rolls back other claims

#     This matches the "atomic before value/score" idea directly. It is kept as
#     a separate kernel so the top2-retry native selector remains available.
#     """
#     tree = cuda.blockIdx.x
#     tid = cuda.threadIdx.x
#     lane = tid & int32(31)
#     wid = tid >> int32(5)

#     if tree >= out_selected_node.shape[0]:
#         return

#     warp_count = cuda.blockDim.x >> int32(5)
#     active_warps = min(warp_count, out_selected_node.shape[1])
#     if wid >= active_warps:
#         return

#     if lane == 0:
#         out_selected_node[tree, wid] = int32(SEL_INVALID_NODE)
#         out_selected_kind[tree, wid] = int32(SELECT_INVALID)
#         out_path_len[tree, wid] = int32(0)

#     max_edges = edge_child_id.shape[2]
#     max_edge_steps = out_path_eids.shape[2]
#     n_nodes = node_status.shape[1]
#     assert max_edge_steps >= 1
#     assert n_nodes > 0

#     final_node = int32(SEL_IN_PROGRESS)
#     final_len = int32(0)

#     attempt = int32(0)

#     while attempt < int32(MAX_SELECT_RETRY) and final_node == int32(SEL_IN_PROGRESS):
#         node = int32(0)
#         depth = int32(0)

#         cuda.syncwarp(FULL_MASK)

#         while True:
#             if node < 0 or node >= n_nodes:
#                 _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, virtual_loss)
#                 cuda.syncwarp(FULL_MASK)
#                 break

#             if node_status[tree, node] == NODE_STATUS_TERMINAL:
#                 final_node = (int32(SELECT_TERMINAL) << int32(22)) | node
#                 final_len = depth + int32(1)
#                 break

#             cur_expanded = node_expanded[tree, node]
#             if cur_expanded < 0:
#                 cur_expanded = int32(0)
#             if cur_expanded > max_edges:
#                 cur_expanded = max_edges

#             parent_n_eff = int32(0)
#             eid = lane
#             while eid < cur_expanded:
#                 child = edge_child_id[tree, node, eid]
#                 if child >= 0 and child < n_nodes:
#                     parent_n_eff += edge_N[tree, node, eid] + edge_inflight[tree, node, eid]
#                 eid += int32(WARP_SIZE)
#             parent_n_eff = _warp_reduce_sum(parent_n_eff)
#             if parent_n_eff < 1:
#                 parent_n_eff = int32(1)

#             allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)

#             if cur_expanded < allowed:
#                 claim_ok = int32(1)
#                 if lane == 0:
#                     old = cuda.atomic.add(edge_inflight, (tree, node, cur_expanded), virtual_loss)
#                     if old != 0:
#                         claim_ok = int32(0)
#                         cuda.atomic.sub(edge_inflight, (tree, node, cur_expanded), virtual_loss)

#                 claim_ok = cuda.shfl_sync(FULL_MASK, claim_ok, 0)

#                 if claim_ok != 0:
#                     final_node = (int32(SELECT_EXPAND) << int32(22)) | (cur_expanded << int32(14)) | node
#                     final_len = depth + int32(1)
#                 else:
#                     _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, virtual_loss)
#                     cuda.syncwarp(FULL_MASK)
#                 break

#             if depth >= max_edge_steps:
#                 final_node = (int32(SELECT_DEPTH_LIMIT) << int32(22)) | node
#                 final_len = depth + int32(1)
#                 break

#             sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))
#             rotation = wid % cur_expanded
#             best_score = float32(NEG_INF_F32)
#             best_eid = int32(INT32_MAX)
#             best_child = int32(-1)

#             eid = lane
#             while eid < cur_expanded:
#                 child = edge_child_id[tree, node, eid]
#                 if child >= 0 and child < n_nodes:
#                     prior = edge_prior[tree, node, eid]
#                     w = edge_W[tree, node, eid]
#                     n_edge = edge_N[tree, node, eid]
#                     old_inflight = cuda.atomic.add(edge_inflight, (tree, node, eid), virtual_loss)
#                     score = _puct_score(cpuct, prior, w, n_edge, old_inflight, sqrt_parent_n_eff)

#                     if _score_better_rotated(score, int32(eid), best_score, best_eid, rotation, cur_expanded):
#                         best_score = score
#                         best_eid = int32(eid)
#                         best_child = child

#                 eid += int32(WARP_SIZE)

#             best_score, best_eid, best_child = _warp_reduce_best_rotated(
#                 best_score, best_eid, best_child, rotation, cur_expanded,
#             )

#             eid = lane
#             while eid < cur_expanded:
#                 child = edge_child_id[tree, node, eid]
#                 if child >= 0 and child < n_nodes and int32(eid) != best_eid:
#                     cuda.atomic.sub(edge_inflight, (tree, node, eid), virtual_loss)
#                 eid += int32(WARP_SIZE)

#             if best_child < 0:
#                 _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, virtual_loss)
#                 cuda.syncwarp(FULL_MASK)
#                 break

#             parent = node
#             if lane == 0:
#                 out_path_eids[tree, wid, depth] = (parent << int32(8)) | best_eid

#             cuda.syncwarp(FULL_MASK)

#             node = best_child
#             depth += int32(1)

#         if final_node == int32(SEL_IN_PROGRESS):
#             attempt += int32(1)

#     if lane == 0:
#         if final_node >= 0:
#             kind = final_node >> int32(22)
#             expand_slot = (final_node >> int32(14)) & int32(0xFF)
#             node_id = final_node & int32(0x3FFF)
#             out_selected_node[tree, wid] = node_id
#             out_selected_kind[tree, wid] = kind | (expand_slot << int32(4))
#             out_path_len[tree, wid] = final_len
#         else:
#             out_selected_node[tree, wid] = int32(SEL_INVALID_NODE)
#             out_selected_kind[tree, wid] = int32(SELECT_INVALID)
#             out_path_len[tree, wid] = int32(0)


# ============================================================
# Backup kernel
# ============================================================

@cuda.jit(void(int32,
                   float32[:, :, :], int32[:, :, :], int32[:, :, :],
                   int32[:, :], int32[:, :],
                   int32[:, :, :], int32[:, :],
                   float32[:, :, :]))
def _backup_kernel_native(virtual_loss,
                          edge_W, edge_N, edge_inflight,
                          out_selected_node, out_selected_kind,
                          out_path_eids, out_path_len,
                          path_edge_values):
    """
    完成 backup：将 virtual visit 转换为真实统计。

    Selection 阶段只操作 edge_inflight (int32)：
      edge_inflight += virtual_loss       （所有候选边先膨胀，losers 后削减）

    Backup 阶段完成转换：
      edge_inflight -= virtual_loss       （撤销虚拟）
      edge_N += 1                          （写入真实）
      edge_W += value                      （写入真实累积价值，无需 W 偏移）

    out_selected_kind 编码: kind | (expand_slot << 4)
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

    raw_kind = out_selected_kind[tree, wid]
    kind = raw_kind & int32(0xF)
    if kind == SELECT_INVALID:
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
            v = path_edge_values[tree, wid, d]
            cuda.atomic.sub(edge_inflight, (tree, parent, slot), virtual_loss)
            cuda.atomic.add(edge_N, (tree, parent, slot), int32(1))
            cuda.atomic.add(edge_W, (tree, parent, slot), v)
        d += int32(WARP_SIZE)

    # Release expansion claim on edge_inflight (virtual_loss-aware)
    if lane == 0 and kind == SELECT_EXPAND:
        leaf = out_selected_node[tree, wid]
        expand_slot = (raw_kind >> int32(4)) & int32(0xFF)
        if leaf >= 0:
            cuda.atomic.sub(edge_inflight, (tree, leaf, expand_slot), virtual_loss)
