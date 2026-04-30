import math
import numpy as np
import numba as nb
import numba.cuda as cuda
from numba import void, float32, int32
# 数据结构设计

class Tree:
    pass

class Traverse:
    path: list          # 记录遍历的；路径

# 由[tree_id, node_id]索引
class Node:
    status: int         # 节点状态: OK / Terminal
    inflight: int       # inflight 的数量
    state: int          # 对应的状态空间，指向状态池
    N: int              # 节点访问次数
    expended: int       # 已经展开的 edge 个数。如果 =0 说明是叶子节点, 由于是渐宽树，所以会逐步展开
    legals: int         # 在 pw_limit 约束下，允许展开的节点个数
    value: float        # 对应当前状态的状态价值
    

selected_vec_list: list # 用于存储当前被选中的边节点列表

# 由[tree_id, edge_id]索引
# edge_id = (node_id << 6) | slot
class Edge:
    child_id: int       # -1: Invalid, 否则是正常的node id 
    n: int              # 访问次数
    inflight: int       # 虚拟代价
    W: float            # 累积动作价值
    prob: float         # 对应的NN采样概率
    prior: float        # 先验价值，来自 Value 计算？

def init_data():

    # 宽松存储，将256个Action一次性分配完毕
    TREE_CNT = 16
    MAX_NODE = 16384        # 最大16K个节点
    MAX_ACTION = 256        # 每个节点最大256个动作

    TREE_SHAPE = (TREE_CNT,)
    trees_selected = cuda.device_array(TREE_SHAPE, np.int32)    # type: ignore

    ROBOT_TURNS = 2
    
    NODE_SHAPE = (TREE_CNT, MAX_NODE)
    nodes_status = cuda.device_array(NODE_SHAPE, np.int32)      # type: ignore      # TODO. 与 expanded 合并
    nodes_inflight = cuda.device_array(NODE_SHAPE, np.int32)    # type: ignore
    nodes_state = cuda.device_array(NODE_SHAPE, np.int32)       # type: ignore
    nodes_value = cuda.device_array((TREE_CNT, MAX_NODE, ROBOT_TURNS), np.int32)       # type: ignore      # TODO. 也许可以和state存到一块


    EDGE_SHAPE = (TREE_CNT, MAX_NODE, MAX_ACTION)
    edges_child_id = cuda.device_array(EDGE_SHAPE, np.int32)    # type: ignore
    edges_n = cuda.device_array(EDGE_SHAPE, np.int32)           # type: ignore      # 访问次数
    edges_inflight = cuda.device_array(EDGE_SHAPE, np.int32)    # type: ignore      # 虚拟损失
    edges_W = cuda.device_array(EDGE_SHAPE, np.float32)         # type: ignore      # 累计价值
    edges_prob = cuda.device_array(EDGE_SHAPE, np.float32)      # type: ignore
    edges_prior = cuda.device_array(EDGE_SHAPE, np.float32)     # type: ignore


NODE_STATUS_OK = 0
NODE_STATUS_TERMINAL = 1

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF
INT32_MAX = 2147483647

@cuda.jit(device=True, inline=True)
def warp_reduce_max_argmax_f32(v, idx):
    """
    warp-level reduce max + argmax.

    输入：
    - v: 当前 lane 的 value, float32
    - idx: 当前 lane 对应的全局 index, int32

    输出：
    - lane 0 上返回 warp 的 max value 和 argmax index
    - 其他 lane 上返回值未必有意义，除非你再 broadcast
    """
    other_v = cuda.shfl_down_sync(FULL_MASK, v, 16)
    other_i = cuda.shfl_down_sync(FULL_MASK, idx, 16)
    if other_v > v or (other_v == v and other_i < idx):
        v = other_v
        idx = other_i

    other_v = cuda.shfl_down_sync(FULL_MASK, v, 8)
    other_i = cuda.shfl_down_sync(FULL_MASK, idx, 8)
    if other_v > v or (other_v == v and other_i < idx):
        v = other_v
        idx = other_i

    other_v = cuda.shfl_down_sync(FULL_MASK, v, 4)
    other_i = cuda.shfl_down_sync(FULL_MASK, idx, 4)
    if other_v > v or (other_v == v and other_i < idx):
        v = other_v
        idx = other_i

    other_v = cuda.shfl_down_sync(FULL_MASK, v, 2)
    other_i = cuda.shfl_down_sync(FULL_MASK, idx, 2)
    if other_v > v or (other_v == v and other_i < idx):
        v = other_v
        idx = other_i

    other_v = cuda.shfl_down_sync(FULL_MASK, v, 1)
    other_i = cuda.shfl_down_sync(FULL_MASK, idx, 1)
    if other_v > v or (other_v == v and other_i < idx):
        v = other_v
        idx = other_i

    return v, idx


@cuda.jit(int32(float32, float32, int32), device=True, inline=True)
def _max_children(Cpw, alpha_pw, n_visit):
    return math.ceil(Cpw*math.pow(n_visit, alpha_pw))

EPS = 1e-5

@cuda.jit(void(float32, float32, float32, float32,
                   int32[:, :, :], float32[:, :, :], int32[:, :, :],
                   int32[:, :], int32[:, :], int32[:, :],
                   int32[:, :], int32[:, :, :]))
def _select_kernel_native(Cexp, alpha_exp, Cpw, alpha_pw,
                          edge_child_id, edges_W, edges_n,
                          node_status, node_expended, node_n,
                          trees_nodes_selected, trees_selected_paths):
    """每个 warp 独立遍历树，选出一个待扩展节点。

    1 block = 1 tree；1 warp = 1 条遍历路径。
    虚拟损失避免 warp 之间重复选择同一条边；跨 warp 去重确保节点不重复扩展。

    输出：
      trees_nodes_selected[tree, warp]   — 选中的待扩展节点 id
      trees_selected_paths[tree, warp, :] — 遍历路径，末尾元素 = 路径长度
    """
    tree = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    wid = tid // 32          # warp id in block
    lid = tid & 31          # lane id in warp
    warp_count = cuda.blockDim.x // 32
    max_path_dim = trees_selected_paths.shape[2]
    max_depth = max_path_dim - 1   # 最后一个位置留给路径长度

    # ── 跨 warp 通信用的 shared memory ──
    s_warp_node = cuda.shared.array(32, dtype=int32)       # type: ignore  # warp
    s_warp_depth = cuda.shared.array(32, dtype=int32)      # type: ignore
    s_warp_valid = cuda.shared.array(32, dtype=int32)      # type: ignore

    node = 0    # 从根节点开始
    depth = 0

    # 根节点写入路径        # TODO. 默认不写入节点id，而是写入边 id； 节点的 N 作废，全部使用边 N 代替
    trees_selected_paths[tree, wid, 0] = int32(0)

    # 每个 warp 独立遍历一次树
    while depth < max_depth:
        if node_status[tree, node] == NODE_STATUS_TERMINAL:
            break

        n_visit = node_n[tree, node]
        max_children_val = max(_max_children(Cpw, alpha_pw, n_visit), 1)    # 需保证未访问节点至少可扩展 1 个子节点
        cur_expended = node_expended[tree, node]

        # 节点尚未完全扩展 -> 选为待扩展节点，停止下钻
        if cur_expended < max_children_val:
            break

        # warp 内的 lane 以 stride=32 遍历边，计算 UCB，使用规约选择算子
        best_ucb = -math.inf
        best_child = int32(-1)
        best_eid = int32(-1)

        eid = lid
        while eid < cur_expended:
            child = edge_child_id[tree, node, eid]
            if child != int32(-1):
                w = edges_W[tree, node, eid]
                n_edge = edges_n[tree, node, eid]
                if n_edge > 0:
                    q = w / float32(n_edge)
                    u_val = Cexp * math.sqrt(math.pow(max(float32(1.0), w), alpha_exp) / float32(n_edge))
                    ucb = q + u_val
                else:
                    ucb = math.inf       # 未访问过的子节点优先级最高       # TODO. 未选择节点会有值函数输出的价值作为初始参考

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
                    best_eid = eid
            eid += 32

        # warp 级 reduce-max / argmax                                   # TODO. 使用CCCL的库代替    
        for offset in (16, 8, 4, 2, 1):
            other_ucb = cuda.shfl_down_sync(0xFFFFFFFF, best_ucb, offset)
            other_child = cuda.shfl_down_sync(0xFFFFFFFF, best_child, offset)
            other_eid = cuda.shfl_down_sync(0xFFFFFFFF, best_eid, offset)
            if other_ucb > best_ucb:
                best_ucb = other_ucb
                best_child = other_child
                best_eid = other_eid                                   # TODO: best_eid 有什么用？ 不如用 warp_reduce_max_argmax_f32 代替

        # 安全兜底：无合法子节点时停止                                      # TODO. 什么时候会出现无合法子节点？
        if best_child == int32(-1):
            break

        # 下钻到最优子节点，保存路径                                        # TODO. 虚拟损失如何添加？
        node = best_child
        depth += 1
        if lid == 0:
            trees_selected_paths[tree, wid, depth] = node

    # 记录每个 warp 的结果
    if lid == 0:
        s_warp_node[wid] = node
        s_warp_depth[wid] = depth
        s_warp_valid[wid] = int32(1)

    cuda.syncthreads()

    # 跨 warp 去重：相同节点只保留 wid 最小的
    # TODO. warp级别的规约，不应该只交给 iwid=0 的线程做
    if lid == 0:
        for i in range(warp_count):
            if s_warp_valid[i]:
                for j in range(i + 1, warp_count):
                    if s_warp_valid[j] and s_warp_node[i] == s_warp_node[j]:
                        s_warp_valid[j] = int32(0)

    cuda.syncthreads()                                                  # TODO: 不必要

    # 写出最终结果
    if lid == 0:
        trees_nodes_selected[tree, wid] = s_warp_node[wid]
        # 末尾元素存路径长度（路径上的节点数）
        trees_selected_paths[tree, wid, max_path_dim - 1] = (s_warp_depth[wid] + int32(1))