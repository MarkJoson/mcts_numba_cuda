import math
from numba import cuda, int32, float32, void


# ============================================================
# Config
# ============================================================

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF
INT32_MAX = 2147483647
NEG_INF_F32 = -3.4028234663852886e38

# TODO: 改成你项目里真实的 terminal status
NODE_STATUS_TERMINAL = 2

# selection result kind
SELECT_INVALID = 0
SELECT_EXPAND = 1
SELECT_TERMINAL = 2
SELECT_DEPTH_LIMIT = 3

# claim 失败后重新从 root selection 的次数
# 过大浪费，过小可能 batch 利用率低；建议 2~4。
MAX_SELECT_RETRY = 3


# ============================================================
# Device helpers
# ============================================================

@cuda.jit(device=True, inline=True)
def _score_better(score, eid, best_score, best_eid):
    """
    比较规则：
      1. score 大者胜
      2. score 相等时 eid 小者胜，保证 deterministic
    """
    if score > best_score:
        return True
    if score == best_score and eid < best_eid:
        return True
    return False


@cuda.jit(device=True, inline=True)
def _warp_reduce_best(score, eid, child):
    """
    warp reduce:
      - 选择最大 score
      - score 相等时选择 eid 最小者
      - 最终把 lane 0 的 winner broadcast 到所有 lane

    返回值在整个 warp 内一致。
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

    # 关键：broadcast lane 0 winner。
    score = cuda.shfl_sync(FULL_MASK, score, 0)
    eid = cuda.shfl_sync(FULL_MASK, eid, 0)
    child = cuda.shfl_sync(FULL_MASK, child, 0)

    return score, eid, child


@cuda.jit(device=True, inline=True)
def _allowed_children(c_pw, alpha_pw, n_node, max_edges):
    """
    Progressive widening:

        allowed = ceil(c_pw * N(s)^alpha_pw)

    并 clamp 到 [1, max_edges]。
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


@cuda.jit(device=True, inline=True)
def _puct_score(cpuct, prior, w, n_edge, sqrt_parent_n):
    """
    标准 PUCT:

        Q + U
        Q = W(s,a) / N(s,a)
        U = cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    注意：
      - W/N 已经包含 virtual loss 的影响。
      - n_edge == 0 时 Q = 0。
    """
    q = float32(0.0)
    if n_edge > 0:
        q = w / float32(n_edge)

    u = cpuct * prior * sqrt_parent_n / float32(n_edge + int32(1))
    return q + u


@cuda.jit(device=True, inline=True)
def _rollback_vloss_path(tree, wid, depth,
                         virtual_loss,
                         path_nodes, path_eids,
                         edge_W, edge_N, node_N):
    """
    只由 lane 0 调用。

    回滚当前 attempt 已经加上的 virtual visit / virtual loss。
    path_nodes[0 : depth + 1] 有效。
    path_eids[0 : depth] 有效。
    """
    d = int32(0)

    while d <= depth:
        node = path_nodes[tree, wid, d]

        if node >= 0:
            cuda.atomic.sub(node_N, (tree, node), int32(1))

        if d < depth:
            eid = path_eids[tree, wid, d]
            if node >= 0 and eid >= 0:
                cuda.atomic.sub(edge_N, (tree, node, eid), int32(1))
                cuda.atomic.add(edge_W, (tree, node, eid), virtual_loss)

        d += int32(1)


# ============================================================
# Main selection kernel
# ============================================================

@cuda.jit(
    void(
        float32, float32, float32, float32,
        int32[:, :, :], float32[:, :, :], float32[:, :, :], int32[:, :, :],
        int32[:, :], int32[:, :], int32[:, :], int32[:, :],
        int32[:, :], int32[:, :], int32[:, :, :], int32[:, :, :], int32[:, :]
    ),
    fastmath=True,
)
def select_puct_pw_vloss_kernel(
    cpuct, c_pw, alpha_pw, virtual_loss,

    # edges
    edge_child_id,     # int32 [tree, node, eid], -1 means invalid
    edge_prior,        # float32 [tree, node, eid], P(s,a)
    edge_W,            # float32 [tree, node, eid], total value, includes in-flight virtual loss
    edge_N,            # int32 [tree, node, eid], visit count, includes in-flight virtual visit

    # nodes
    node_status,       # int32 [tree, node]
    node_expanded,     # int32 [tree, node], number of currently expanded children
    node_N,            # int32 [tree, node], visit count, includes in-flight virtual visit
    node_in_flight,    # int32 [tree, node], 0/1 expansion claim lock

    # outputs
    out_selected_node, # int32 [tree, wid], selected node or -1
    out_selected_kind, # int32 [tree, wid], SELECT_*
    out_path_nodes,    # int32 [tree, wid, max_nodes_in_path]
    out_path_eids,     # int32 [tree, wid, max_edges_in_path]
    out_path_len       # int32 [tree, wid], number of nodes in path
):
    """
    1 block = 1 tree
    1 warp  = 1 selection

    输出语义：
      out_selected_kind == SELECT_EXPAND:
          out_selected_node 是需要 expand / progressive widen 的父节点。
          该节点已经被 node_in_flight CAS claim。
          后续 expansion + backup 完成后必须释放 node_in_flight。

      out_selected_kind == SELECT_TERMINAL:
          out_selected_node 是 terminal node。
          不 claim node_in_flight。
          后续直接 terminal backup。

      out_selected_kind == SELECT_DEPTH_LIMIT:
          达到 out_path_nodes/out_path_eids 容量上限。
          可以当作 evaluation leaf 处理。
          不 claim node_in_flight。

      out_selected_kind == SELECT_INVALID:
          该 warp 没有产出有效 selection。
          kernel 内已经 rollback virtual loss。
    """

    tree = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    lane = tid & int32(31)
    wid = tid >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    warp_count = cuda.blockDim.x >> int32(5)

    active_warps = warp_count
    if active_warps > out_selected_node.shape[1]:
        active_warps = out_selected_node.shape[1]
    if active_warps > out_selected_kind.shape[1]:
        active_warps = out_selected_kind.shape[1]
    if active_warps > out_path_nodes.shape[1]:
        active_warps = out_path_nodes.shape[1]
    if active_warps > out_path_eids.shape[1]:
        active_warps = out_path_eids.shape[1]
    if active_warps > out_path_len.shape[1]:
        active_warps = out_path_len.shape[1]

    if wid >= active_warps:
        return

    if lane == 0:
        out_selected_node[tree, wid] = int32(-1)
        out_selected_kind[tree, wid] = int32(SELECT_INVALID)
        out_path_len[tree, wid] = int32(0)

    max_node_slots = out_path_nodes.shape[2]
    if max_node_slots < 1:
        return

    max_edge_steps = max_node_slots - 1
    if out_path_eids.shape[2] < max_edge_steps:
        max_edge_steps = out_path_eids.shape[2]

    n_nodes = int32(node_status.shape[1])
    max_edges = int32(edge_child_id.shape[2])

    if n_nodes <= 0:
        return

    final_node = int32(-1)
    final_kind = int32(SELECT_INVALID)
    final_len = int32(0)

    done = int32(0)
    attempt = int32(0)

    while attempt < int32(MAX_SELECT_RETRY) and done == int32(0):
        node = int32(0)
        depth = int32(0)

        # Root virtual visit.
        if lane == 0:
            out_path_nodes[tree, wid, 0] = int32(0)
            cuda.atomic.add(node_N, (tree, int32(0)), int32(1))

        cuda.syncwarp(FULL_MASK)

        while True:
            # Defensive guard.
            if node < 0 or node >= n_nodes:
                if lane == 0:
                    _rollback_vloss_path(
                        tree, wid, depth,
                        virtual_loss,
                        out_path_nodes, out_path_eids,
                        edge_W, edge_N, node_N,
                    )
                cuda.syncwarp(FULL_MASK)
                break

            # Terminal leaf.
            if node_status[tree, node] == NODE_STATUS_TERMINAL:
                final_node = node
                final_kind = int32(SELECT_TERMINAL)
                final_len = depth + int32(1)
                done = int32(1)
                break

            parent_n = node_N[tree, node]
            if parent_n < 1:
                parent_n = int32(1)

            cur_expanded = node_expanded[tree, node]
            if cur_expanded < 0:
                cur_expanded = int32(0)
            if cur_expanded > max_edges:
                cur_expanded = max_edges

            allowed = _allowed_children(c_pw, alpha_pw, parent_n, max_edges)

            # Progressive widening: 当前 node 还可以扩展新 child。
            if cur_expanded < allowed:
                claim_ok = int32(1)

                if lane == 0:
                    old = cuda.atomic.cas(
                        node_in_flight,
                        (tree, node),
                        int32(0),
                        int32(1),
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
                    # 别的 warp / batch 已经 claim 这个 node。
                    # 回滚本 attempt，然后 retry from root。
                    if lane == 0:
                        _rollback_vloss_path(
                            tree, wid, depth,
                            virtual_loss,
                            out_path_nodes, out_path_eids,
                            edge_W, edge_N, node_N,
                        )
                    cuda.syncwarp(FULL_MASK)

                break

            # 输出路径容量达到上限：作为 depth-limit leaf。
            if depth >= max_edge_steps:
                final_node = node
                final_kind = int32(SELECT_DEPTH_LIMIT)
                final_len = depth + int32(1)
                done = int32(1)
                break

            # Fully expanded under PW: 在已有 child edges 里按 PUCT 选。
            sqrt_parent_n = math.sqrt(float32(parent_n))

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

                    score = _puct_score(
                        cpuct,
                        prior,
                        w,
                        n_edge,
                        sqrt_parent_n,
                    )

                    if _score_better(score, int32(eid), best_score, best_eid):
                        best_score = score
                        best_eid = int32(eid)
                        best_child = child

                eid += int32(WARP_SIZE)

            best_score, best_eid, best_child = _warp_reduce_best(
                best_score,
                best_eid,
                best_child,
            )

            # 数据不一致：node 被认为 fully expanded，但没有合法 child。
            if best_child < 0:
                if lane == 0:
                    _rollback_vloss_path(
                        tree, wid, depth,
                        virtual_loss,
                        out_path_nodes, out_path_eids,
                        edge_W, edge_N, node_N,
                    )
                cuda.syncwarp(FULL_MASK)
                break

            parent = node

            # Apply virtual loss / virtual visit on selected edge and child node.
            if lane == 0:
                out_path_eids[tree, wid, depth] = best_eid

                cuda.atomic.add(edge_N, (tree, parent, best_eid), int32(1))
                cuda.atomic.add(edge_W, (tree, parent, best_eid), -virtual_loss)
                cuda.atomic.add(node_N, (tree, best_child), int32(1))

                out_path_nodes[tree, wid, depth + int32(1)] = best_child

            cuda.syncwarp(FULL_MASK)

            node = best_child
            depth += int32(1)

        if done == int32(0):
            attempt += int32(1)

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
# Backup / finalize virtual loss
# ============================================================

@cuda.jit(
    void(
        float32,
        float32[:, :, :],
        int32[:, :],
        int32[:, :], int32[:, :],
        int32[:, :, :], int32[:, :, :], int32[:, :],
        float32[:, :, :]
    ),
    fastmath=True,
)
def backup_selected_paths_puct_vloss_kernel(
    virtual_loss,

    edge_W,            # float32 [tree, node, eid]
    node_in_flight,    # int32 [tree, node]

    out_selected_node, # int32 [tree, wid]
    out_selected_kind, # int32 [tree, wid]
    out_path_nodes,    # int32 [tree, wid, max_nodes_in_path]
    out_path_eids,     # int32 [tree, wid, max_edges_in_path]
    out_path_len,      # int32 [tree, wid]

    path_edge_values   # float32 [tree, wid, max_edges_in_path]
):
    """
    完成 backup：

    selection 阶段已经做了：
      edge_N += 1
      node_N += 1
      edge_W -= virtual_loss

    所以 backup 阶段不要再加 edge_N/node_N。
    这里只做：
      edge_W += virtual_loss + value_for_this_edge

    path_edge_values[tree, wid, d] 必须已经是 edge_W 所采用视角下的 value。
    例如二人零和游戏里，如果 edge_W 存的是 parent-to-move 视角，
    你需要在外部或另一个 kernel 里把符号翻转处理好。
    """

    tree = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    lane = tid & int32(31)
    wid = tid >> int32(5)

    if tree >= out_selected_node.shape[0]:
        return

    warp_count = cuda.blockDim.x >> int32(5)

    active_warps = warp_count
    if active_warps > out_selected_node.shape[1]:
        active_warps = out_selected_node.shape[1]

    if wid >= active_warps:
        return

    kind = out_selected_kind[tree, wid]
    if kind == SELECT_INVALID:
        return

    plen = out_path_len[tree, wid]
    if plen <= 0:
        return

    edge_count = plen - int32(1)

    d = lane
    while d < edge_count:
        parent = out_path_nodes[tree, wid, d]
        eid = out_path_eids[tree, wid, d]

        if parent >= 0 and eid >= 0:
            v = path_edge_values[tree, wid, d]
            cuda.atomic.add(edge_W, (tree, parent, eid), virtual_loss + v)

        d += int32(WARP_SIZE)

    # 只有 SELECT_EXPAND claim 过 node_in_flight。
    if lane == 0 and kind == SELECT_EXPAND:
        leaf = out_selected_node[tree, wid]
        if leaf >= 0:
            cuda.atomic.exch(node_in_flight, (tree, leaf), int32(0))