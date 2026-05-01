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
PACKED_EDGE_LIMIT = 256      # path edge encoding stores slot in 8 bits

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
    expanded: int       # >=0: 已经展开的 edge 个数; -1: terminal
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
    nodes_state = cuda.device_array(NODE_SHAPE, np.int32)       # type: ignore
    nodes_expanded = cuda.device_array(NODE_SHAPE, np.int32)    # type: ignore  # >=0 expanded count, -1 terminal
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

@cuda.jit(void(
    float32, float32, float32,
    int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :], int32[:, :, :],
    int32[:, :],
    int32[:, :], int32[:, :, :], int32[:, :]))
def _select_kernel_native(cpuct, c_pw, alpha_pw,
                          edge_child_id, edge_prior, edge_W, edge_N, edge_inflight,
                          node_expanded,
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

    # 防止输出数组大小不够
    warp_count = cuda.blockDim.x >> int32(5)
    active_warps = min(warp_count, out_selected_node.shape[1])
    if wid >= active_warps:
        return

    # lane0 负责将输出数组初始化
    if lane == 0:
        out_selected_node[tree, wid] = int32(SEL_INVALID_NODE)
        out_path_len[tree, wid] = int32(0)

    max_edges = edge_child_id.shape[2]
    max_edge_steps = out_path_eids.shape[2]
    n_nodes = node_expanded.shape[1]

    # Encoding guards:
    #   final_node: (kind << 22) | (expand_slot << 14) | node_id
    #   path eid  : (parent << 8) | slot
    # If these limits are exceeded, returning invalid is safer than silently
    # truncating ids during bit packing.
    if max_edge_steps < 1 or n_nodes <= 0 or n_nodes > int32(PACKED_NODE_LIMIT) or max_edges <= 0 or max_edges > int32(PACKED_EDGE_LIMIT):
        return

    # 对 final_node 进行了压缩，包含了 当前结果类型(31-22) + expand_slot(21-14) + node_id(0-13)
    #! 当节点状态是待扩展时，expand_slot指示了新的扩展节点的起始位置。
    # TODO: 认为每一个 final_node 是对应一次扩展？还是写明需要扩展节点的数量？
    final_node = int32(SEL_IN_PROGRESS)
    final_len = int32(0)


    #! 因为某些原因，单次尝试可能会失败，如多个节点同时进入扩展，此时再重试一次可能会走不同的路径。但是这样的代价值的商榷。
    node = int32(0)      # start from root
    depth = int32(0)
    while True:
        # 防御性编程: 节点的id值是无效值，或者大于最大节点数，认为是无效节点
        if node < 0 or node >= n_nodes:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1))
            cuda.syncwarp(FULL_MASK)
            break

        node_info = node_expanded[tree, node]

        # 终止态的节点。node_expanded 使用负数 sentinel 表示 terminal，
        # 避免 selection 每层同时读取 node_status 和 node_expanded。
        if node_info == int32(NODE_EXPANDED_TERMINAL):
            final_node = (int32(SELECT_TERMINAL) << int32(22)) | node
            final_len = depth + int32(1)
            break

        if node_info < int32(0) or node_info > max_edges:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1))
            cuda.syncwarp(FULL_MASK)
            break
        
        # TODO: 如果是超出步长的节点，需要考虑用 Value 网络重新估计节点价值
        cur_expanded = node_info

        # 显式处理尚未展开的非终止叶子节点。这样不用依赖 progressive
        # widening 在 cur_expanded=0 时一定给出 allowed>=1。
        if cur_expanded == int32(0):
            final_node = (int32(SELECT_EXPAND) << int32(22)) | node
            final_len = depth + int32(1)
            break

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

        #* 渐进增长树，多个 warp 会同时对 edge_inflight 进行原子加法。如果是数量上未满足待扩展需求节点数量，即作为扩展节点
        if need_expand > 0:
            expand_offset = int32(-1)
            if lane == 0:
                expand_offset = cuda.atomic.add(edge_inflight, (tree, node, cur_expanded), int32(1))
                if expand_offset >= need_expand:
                    cuda.atomic.sub(edge_inflight, (tree, node, cur_expanded), int32(1))
            expand_offset = cuda.shfl_sync(FULL_MASK, expand_offset, 0)         #! 广播当前结果，让所有 lane 进入就绪状态
            expand_slot = cur_expanded + expand_offset
            if expand_offset >= int32(0) and expand_offset < need_expand and expand_slot >= int32(0) and expand_slot < max_edges and expand_slot < int32(PACKED_EDGE_LIMIT):
                final_node = (int32(SELECT_EXPAND) << int32(22)) | (expand_slot << int32(14)) | node
                final_len = depth + int32(1)
            else:   #! 扩展数量不允许进一步拓展，则退出
                _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1))
                cuda.syncwarp(FULL_MASK)
            break

        # 达到最大步长时，当前节点不再扩展，为叶子节点
        if depth >= max_edge_steps:
            final_node = (int32(SELECT_DEPTH_LIMIT) << int32(22)) | node
            final_len = depth + int32(1)
            break

        #* 当渐宽树在完整扩展的情况下，会在已有的孩子边中选择一个最优动作的进行扩展
        #! 存在一种情况是多个线程同时进入了一个node。因此需要在计算 value 时引入虚拟损失
        best_score = float32(NEG_INF_F32)
        best_eid = int32(INT32_MAX)
        best_child = int32(-1)

        #* 计算所有动作价值，在计算价值之前，对 edge_inflight 进行自增
        eid = lane
        sqrt_parent_n_eff = math.sqrt(float32(parent_n_eff))
        rotation = wid % cur_expanded
        while eid < cur_expanded:
            child = edge_child_id[tree, node, eid]
            assert child >= 0 and child < n_nodes
            prior = edge_prior[tree, node, eid]
            w = edge_W[tree, node, eid]
            n_edge = edge_N[tree, node, eid]
            old_inflight = cuda.atomic.add(edge_inflight, (tree, node, eid), int32(1))      #! 计算每个动作选项的价值
            score = _puct_score(cpuct, prior, w, n_edge, old_inflight, sqrt_parent_n_eff)

            if _score_better_rotated(score, int32(eid), best_score, best_eid, rotation, cur_expanded):
                best_score = score
                best_eid = int32(eid)
                best_child = child

            eid += int32(WARP_SIZE)

        #* Reduce 操作，得到最优秀的节点
        #! 当出现同一个分数时，尽量选择不同的路径，选择同一个路径也没关系，只是会出现比较多的冲突影响效率
        best_score, best_eid, best_child = _warp_reduce_best_rotated(
            best_score, best_eid, best_child, rotation, cur_expanded,
        )

        #* 撤销本次未被选中的节点价值
        eid = lane
        while eid < cur_expanded:
            child = edge_child_id[tree, node, eid]
            assert child >= 0 and child < n_nodes 
            if int32(eid) != best_eid:
                cuda.atomic.sub(edge_inflight, (tree, node, eid), int32(1))
            eid += int32(WARP_SIZE)

        #* 没有节点被选中的情况比较特殊，在这种情况下一般是孩子边的价值出了问题
        if best_child < 0:
            _rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight, int32(1))
            cuda.syncwarp(FULL_MASK)
            break
        
        #* Path的个格式为: [(父节点id, 边id), (父节点id, 边id), (父节点id, 边id), ...] 因为需要在扩展时同时对 节点数据 和 边数据 进行修改
        if lane == 0:
            out_path_eids[tree, wid, depth] = (node << int32(8)) | best_eid

        cuda.syncwarp(FULL_MASK)

        node = best_child
        depth += int32(1)
    #^---------------------- 遍历结束 得到节点 --------------------------
    if lane == 0:
        if final_node >= 0:
            out_selected_node[tree, wid] = final_node
            out_path_len[tree, wid] = final_len
        else:
            out_selected_node[tree, wid] = int32(SELECT_INVALID<<22) | int32(SEL_INVALID_NODE)
            out_path_len[tree, wid] = int32(0)


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
