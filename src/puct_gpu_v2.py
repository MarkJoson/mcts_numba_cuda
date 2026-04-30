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
    inflight: int       # CAS expansion lock (0 = free, 1 = claimed)
    state: int          # 对应的状态空间，指向状态池
    expanded: int       # 已经展开的 edge 个数。如果 =0 说明是叶子节点, 由于是渐宽树，所以会逐步展开
    legals: int         # 在 pw_limit 约束下，允许展开的节点个数
    value: float        # 对应当前状态的状态价值

# 由[tree_id, edge_id]索引
# edge_id = (node_id << 6) | slot
class Edge:
    child_id: int       # -1: Invalid, 否则是正常的node id
    n: int              # 访问次数 (真实，只在 backup 时增加)
    inflight: int       # 虚拟访问计数 (selection 时 +1，rollback 或 backup 时 -1)
    W: float            # 累积动作价值 (只在 backup 时增加真实 value)
    prob: float         # 对应的NN采样概率
    prior: float        # 先验概率 (用于PUCT公式)

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
    nodes_inflight = cuda.device_array(NODE_SHAPE, np.int32)    # type: ignore  # CAS expansion lock
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
def _reconstruct_parent(tree, wid, d, out_path_eids, edge_child_id, n_nodes):
    """
    Reconstruct the parent node at depth d by walking the edge path from root.
    Returns -1 if path is broken.
    """
    parent = int32(0)  # root
    i = int32(0)
    while i < d:
        e = out_path_eids[tree, wid, i]
        if parent >= 0 and parent < n_nodes and e >= 0:
            parent = edge_child_id[tree, parent, e]
        else:
            return int32(-1)
        i += int32(1)
    return parent


@cuda.jit(device=True, inline=True)
def _rollback_vloss_path(tree, wid, depth, lane,
                          out_path_eids, edge_inflight,
                          edge_child_id, n_nodes):
    """
    Roll back virtual visits on edges for the current attempt.
    All lanes participate: each lane handles depths at stride=32.
    path_eids[0:depth] are valid.
    Reconstructs parent node from eid path since node_N / out_path_nodes are removed.
    Caller must syncwarp after return.
    """
    d = lane
    while d < depth:
        parent = _reconstruct_parent(tree, wid, d, out_path_eids, edge_child_id, n_nodes)
        eid = out_path_eids[tree, wid, d]
        if parent >= 0 and eid >= 0:
            cuda.atomic.sub(edge_inflight, (tree, parent, eid), int32(1))
        d += int32(WARP_SIZE)


# ============================================================
# Selection kernel
# ============================================================

@cuda.jit(void(float32, float32, float32,
                   int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :], int32[:, :, :],
                   int32[:, :], int32[:, :], int32[:, :],
                   int32[:, :], int32[:, :], int32[:, :, :], int32[:, :]))
def _select_kernel_native(cpuct, c_pw, alpha_pw,
                          edge_child_id, edge_prior, edge_W, edge_N, edge_inflight,
                          node_status, node_expanded, node_in_flight,
                          out_selected_node, out_selected_kind,
                          out_path_eids, out_path_len):
    """
    每个 warp 独立遍历树，选出待扩展节点。

    1 block = 1 tree；1 warp = 1 条遍历路径。
    选择每条边后立刻对其加 virtual visit (edge_inflight += 1)，
    通过 PUCT 中的有效 N 降低后续 warp 选同一条边的概率。
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

    warp_count = cuda.blockDim.x >> int32(5)
    active_warps = min(warp_count, out_selected_node.shape[1])

    if wid >= active_warps:
        return

    # Initialize output to invalid
    if lane == 0:
        out_selected_node[tree, wid] = int32(-1)
        out_selected_kind[tree, wid] = int32(SELECT_INVALID)
        out_path_len[tree, wid] = int32(0)

    max_edge_slots = out_path_eids.shape[2]
    if max_edge_slots < 1:
        return

    max_edge_steps = max_edge_slots

    n_nodes = node_status.shape[1]
    max_edges = edge_child_id.shape[2]
    if n_nodes <= 0:
        return

    final_node = int32(-1)
    final_kind = int32(SELECT_INVALID)
    final_len = int32(0)

    done = int32(0)
    attempt = int32(0)

    while attempt < int32(MAX_SELECT_RETRY) and done == int32(0):
        node = int32(0)      # start from root
        depth = int32(0)

        cuda.syncwarp(FULL_MASK)

        while True:
            # Defensive guard: invalid node
            if node < 0 or node >= n_nodes:
                _rollback_vloss_path(tree, wid, depth, lane,
                                      out_path_eids, edge_inflight,
                                      edge_child_id, n_nodes)
                cuda.syncwarp(FULL_MASK)
                break

            # Terminal node
            if node_status[tree, node] == NODE_STATUS_TERMINAL:
                final_node = node
                final_kind = int32(SELECT_TERMINAL)
                final_len = depth + int32(1)
                done = int32(1)
                break

            cur_expanded = node_expanded[tree, node]
            if cur_expanded < 0:
                cur_expanded = int32(0)
            if cur_expanded > max_edges:
                cur_expanded = max_edges

            # Compute parent_n_eff from edges (sum of edge_N + edge_inflight)
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

            allowed = _allowed_children(c_pw, alpha_pw, parent_n_eff, max_edges)

            # Progressive widening: node can still expand new children
            if cur_expanded < allowed:
                claim_ok = int32(1)
                if lane == 0:
                    old = cuda.atomic.cas(
                        node_in_flight, (tree, node),
                        int32(0), int32(1),
                    )
                    if old != 0:
                        claim_ok = int32(0)

                claim_ok = cuda.shfl_sync(FULL_MASK, claim_ok, 0)

                if claim_ok != 0:
                    final_node = node
                    final_kind = int32(SELECT_EXPAND)
                    final_len = depth + int32(1)
                    done = int32(1)
                else:
                    # Another warp already claimed this node — rollback and retry from root
                    _rollback_vloss_path(tree, wid, depth, lane,
                                          out_path_eids, edge_inflight,
                                          edge_child_id, n_nodes)
                    cuda.syncwarp(FULL_MASK)
                break

            # Path capacity hit — treat as depth-limit leaf
            if depth >= max_edge_steps:
                final_node = node
                final_kind = int32(SELECT_DEPTH_LIMIT)
                final_len = depth + int32(1)
                done = int32(1)
                break

            # Fully expanded under PW: PUCT selection among existing children
            sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))

            best_score = float32(NEG_INF_F32)
            best_eid = int32(INT32_MAX)
            best_child = int32(-1)

            eid = lane
            while eid < cur_expanded:
                child = edge_child_id[tree, node, eid]
                if child >= 0 and child < n_nodes:
                    prior = edge_prior[tree, node, eid]
                    w = edge_W[tree, node, eid]
                    n_edge = edge_N[tree, node, eid]
                    n_inflight = edge_inflight[tree, node, eid]

                    score = _puct_score(cpuct, prior, w, n_edge, n_inflight, sqrt_parent_n_eff)

                    if _score_better(score, int32(eid), best_score, best_eid):
                        best_score = score
                        best_eid = int32(eid)
                        best_child = child

                eid += int32(WARP_SIZE)

            # Warp reduce: find best (score, eid, child) and broadcast to all lanes
            best_score, best_eid, best_child = _warp_reduce_best(
                best_score, best_eid, best_child,
            )

            # No valid child found — data inconsistency, rollback and abort this attempt
            if best_child < 0:
                _rollback_vloss_path(tree, wid, depth, lane,
                                      out_path_eids, edge_inflight,
                                      edge_child_id, n_nodes)
                cuda.syncwarp(FULL_MASK)
                break

            parent = node

            # Apply virtual visit on selected edge
            if lane == 0:
                out_path_eids[tree, wid, depth] = best_eid
                cuda.atomic.add(edge_inflight, (tree, parent, best_eid), int32(1))

            cuda.syncwarp(FULL_MASK)

            node = best_child
            depth += int32(1)

        if done == int32(0):
            attempt += int32(1)

    # Write final result
    if lane == 0:
        if done != 0:
            out_selected_node[tree, wid] = final_node
            out_selected_kind[tree, wid] = final_kind
            out_path_len[tree, wid] = final_len
        else:
            out_selected_node[tree, wid] = int32(-1)
            out_selected_kind[tree, wid] = int32(SELECT_INVALID)
            out_path_len[tree, wid] = int32(0)


# ============================================================
# Backup kernel
# ============================================================

@cuda.jit(void(float32[:, :, :], int32[:, :, :], int32[:, :, :], int32[:, :, :],
                   int32[:, :],
                   int32[:, :], int32[:, :],
                   int32[:, :, :], int32[:, :],
                   float32[:, :, :]))
def _backup_kernel_native(edge_W, edge_N, edge_inflight, edge_child_id,
                          node_in_flight,
                          out_selected_node, out_selected_kind,
                          out_path_eids, out_path_len,
                          path_edge_values):
    """
    完成 backup：将 virtual visit 转换为真实统计。

    Selection 阶段已经做了（virtual）：
      edge_inflight += 1        （边的虚拟访问计数）

    Backup 阶段完成转换：
      edge_inflight -= 1        （撤销虚拟）
      edge_N += 1               （写入真实）
      edge_W += value           （写入真实累积价值）

    node 路径通过 out_path_eids + edge_child_id 从 root 重建。

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

    kind = out_selected_kind[tree, wid]
    if kind == SELECT_INVALID:
        return

    plen = out_path_len[tree, wid]
    if plen <= 0:
        return

    n_nodes = edge_child_id.shape[1]
    edge_count = plen - int32(1)

    # Each lane handles edges at stride=32
    d = lane
    while d < edge_count:
        parent = _reconstruct_parent(tree, wid, d, out_path_eids, edge_child_id, n_nodes)
        eid = out_path_eids[tree, wid, d]
        if parent >= 0 and eid >= 0:
            v = path_edge_values[tree, wid, d]
            cuda.atomic.sub(edge_inflight, (tree, parent, eid), int32(1))
            cuda.atomic.add(edge_N, (tree, parent, eid), int32(1))
            cuda.atomic.add(edge_W, (tree, parent, eid), v)
        d += int32(WARP_SIZE)

    # Release CAS lock for SELECT_EXPAND nodes
    if lane == 0 and kind == SELECT_EXPAND:
        leaf = out_selected_node[tree, wid]
        if leaf >= 0:
            cuda.atomic.exch(node_in_flight, (tree, leaf), int32(0))
