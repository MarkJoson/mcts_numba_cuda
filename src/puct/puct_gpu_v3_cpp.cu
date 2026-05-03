#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFFu
#define INT32_MAX_V 2147483647
#define NEG_INF_F32 -3.4028234663852886e38f

#define NODE_EXPANDED_TERMINAL -1
#define PACKED_NODE_LIMIT 16384
#define PACKED_NODE_MASK 0x3FFF
#define PACKED_EDGE_MASK 0xFF
#define PACKED_KIND_SHIFT 22
#define PACKED_SLOT_SHIFT 14
#define PACKED_REASON_SHIFT 25

#define SELECT_INVALID 0
#define SELECT_EXPAND 1
#define SELECT_TERMINAL 2
#define SELECT_DEPTH_LIMIT 3
#define SELECT_BUSY 4

#define REASON_UNSET 0
#define REASON_OK_EXPAND 1
#define REASON_OK_TERMINAL 2
#define REASON_OK_DEPTH_LIMIT 3
#define REASON_BUSY_EXPAND_INFLIGHT 4
#define REASON_BUSY_WINNER_RECALC 5
#define REASON_INVALID_SHAPE 10
#define REASON_INVALID_NODE_INFO 11
#define REASON_INVALID_NO_VALID_EDGE 14
#define REASON_INVALID_CHILD_OOB 15
#define REASON_INVALID_UNKNOWN 16

#define MAX_RECALC_RETRY 2

static __device__ __forceinline__ int pack_selection(int kind, int node, int expand_slot, int reason) {
    return (reason << PACKED_REASON_SHIFT) | (kind << PACKED_KIND_SHIFT) |
           (expand_slot << PACKED_SLOT_SHIFT) | node;
}

static __device__ __forceinline__ bool valid_select_shape(
    int max_edge_steps, int node_capacity, int node_cnt, int max_edges) {
    return max_edge_steps >= 1 && node_capacity > 0 && node_cnt > 0 &&
           node_cnt <= node_capacity && node_cnt <= PACKED_NODE_LIMIT &&
           max_edges > 0 && max_edges <= (PACKED_EDGE_MASK + 1);
}

static __device__ __forceinline__ int tie_key_block_local(int eid, int tie_offset, int span) {
    int key = eid - tie_offset;
    if (key < 0) {
        key += span;
    }
    return key;
}

static __device__ __forceinline__ bool score_better_block_local(
    float score, int eid, float best_score, int best_eid, int tie_offset, int span) {
    if (score > best_score) {
        return true;
    }
    if (score == best_score) {
        if (best_eid == INT32_MAX_V) {
            return true;
        }
        return tie_key_block_local(eid, tie_offset, span) <
               tie_key_block_local(best_eid, tie_offset, span);
    }
    return false;
}

static __device__ __forceinline__ int warp_reduce_sum_int(int val) {
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return __shfl_sync(FULL_MASK, val, 0);
}

static __device__ __forceinline__ void warp_reduce_best(
    float &score, int &eid, int &inflight, int tie_offset, int span) {
    float other_score = __shfl_down_sync(FULL_MASK, score, 16);
    int other_eid = __shfl_down_sync(FULL_MASK, eid, 16);
    int other_inflight = __shfl_down_sync(FULL_MASK, inflight, 16);
    if (score_better_block_local(other_score, other_eid, score, eid, tie_offset, span)) {
        score = other_score;
        eid = other_eid;
        inflight = other_inflight;
    }

    other_score = __shfl_down_sync(FULL_MASK, score, 8);
    other_eid = __shfl_down_sync(FULL_MASK, eid, 8);
    other_inflight = __shfl_down_sync(FULL_MASK, inflight, 8);
    if (score_better_block_local(other_score, other_eid, score, eid, tie_offset, span)) {
        score = other_score;
        eid = other_eid;
        inflight = other_inflight;
    }

    other_score = __shfl_down_sync(FULL_MASK, score, 4);
    other_eid = __shfl_down_sync(FULL_MASK, eid, 4);
    other_inflight = __shfl_down_sync(FULL_MASK, inflight, 4);
    if (score_better_block_local(other_score, other_eid, score, eid, tie_offset, span)) {
        score = other_score;
        eid = other_eid;
        inflight = other_inflight;
    }

    other_score = __shfl_down_sync(FULL_MASK, score, 2);
    other_eid = __shfl_down_sync(FULL_MASK, eid, 2);
    other_inflight = __shfl_down_sync(FULL_MASK, inflight, 2);
    if (score_better_block_local(other_score, other_eid, score, eid, tie_offset, span)) {
        score = other_score;
        eid = other_eid;
        inflight = other_inflight;
    }

    other_score = __shfl_down_sync(FULL_MASK, score, 1);
    other_eid = __shfl_down_sync(FULL_MASK, eid, 1);
    other_inflight = __shfl_down_sync(FULL_MASK, inflight, 1);
    if (score_better_block_local(other_score, other_eid, score, eid, tie_offset, span)) {
        score = other_score;
        eid = other_eid;
        inflight = other_inflight;
    }

    score = __shfl_sync(FULL_MASK, score, 0);
    eid = __shfl_sync(FULL_MASK, eid, 0);
    inflight = __shfl_sync(FULL_MASK, inflight, 0);
}

static __device__ __forceinline__ int idx2(int tree, int node, int nodes) {
    return tree * nodes + node;
}

static __device__ __forceinline__ int idx3(int tree, int node, int edge, int nodes, int edges) {
    return (tree * nodes + node) * edges + edge;
}

static __device__ __forceinline__ int idx_out_path(int tree, int wid, int depth, int warps, int path_depth) {
    return (tree * warps + wid) * path_depth + depth;
}

static __device__ __forceinline__ int idx_out(int tree, int wid, int warps) {
    return tree * warps + wid;
}

static __device__ __forceinline__ int allowed_children(float c_pw, float alpha_pw, int n_node, int max_edges) {
    float val = (alpha_pw == 0.5f) ? c_pw * sqrtf((float)n_node) : c_pw * powf((float)n_node, alpha_pw);
    if (val >= (float)max_edges) {
        return max_edges;
    }
    int allowed = (int)ceilf(val);
    return max(1, allowed);
}

static __device__ __forceinline__ int parent_n_eff(
    const int *edge_N, const int *edge_inflight, int tree, int node, int cur_expanded,
    int lane, int nodes, int edges) {
    int total = 0;
    for (int eid = lane; eid < cur_expanded; eid += WARP_SIZE) {
        int ix = idx3(tree, node, eid, nodes, edges);
        total += edge_N[ix] + edge_inflight[ix];
    }
    total = warp_reduce_sum_int(total);
    return max(total, 1);
}

static __device__ __forceinline__ float puct_score(
    float cpuct, float prior, float w, int n_edge, int n_edge_inflight, float sqrt_parent_n_eff) {
    int n_edge_eff = n_edge + n_edge_inflight;
    float q = 0.0f;
    if (n_edge > 0) {
        q = w / (float)n_edge;
    }
    float u = cpuct * prior * sqrt_parent_n_eff / (float)(n_edge_eff + 1);
    return q + u;
}

static __device__ __forceinline__ void reduce_best_edge(
    float cpuct, const float *edge_prior, const float *edge_W, const int *edge_N,
    const int *edge_inflight, int tree, int node, int cur_expanded, int lane,
    float sqrt_parent_n_eff, int tie_offset, int nodes, int edges,
    int &best_eid, int &best_inflight) {
    float best_score = NEG_INF_F32;
    best_eid = INT32_MAX_V;
    best_inflight = 0;

    for (int eid = lane; eid < cur_expanded; eid += WARP_SIZE) {
        int ix = idx3(tree, node, eid, nodes, edges);
        int inflight = edge_inflight[ix];
        float score = puct_score(
            cpuct, edge_prior[ix], edge_W[ix], edge_N[ix], inflight, sqrt_parent_n_eff);
        if (score_better_block_local(score, eid, best_score, best_eid, tie_offset, cur_expanded)) {
            best_score = score;
            best_eid = eid;
            best_inflight = inflight;
        }
    }

    warp_reduce_best(best_score, best_eid, best_inflight, tie_offset, cur_expanded);
}

static __device__ __forceinline__ void rollback_vloss_path(
    int tree, int wid, int depth, int lane, int *out_path_eids, int *edge_inflight,
    int virtual_loss, int node_cnt, int nodes, int max_edges, int warps, int path_depth) {
    for (int d = lane; d < depth; d += WARP_SIZE) {
        int encoded = out_path_eids[idx_out_path(tree, wid, d, warps, path_depth)];
        int parent = encoded >> 8;
        int slot = encoded & 0xFF;
        if (parent >= 0 && parent < node_cnt && slot >= 0 && slot < max_edges) {
            atomicSub(&edge_inflight[idx3(tree, parent, slot, nodes, max_edges)], virtual_loss);
        }
    }
}

static __device__ __forceinline__ int claim_expand_slot(
    int *node_expand_inflight, int tree, int node, int start_slot, int limit_slot,
    int lane, int nodes) {
    int expand_slot = -1;
    if (lane == 0) {
        int ticket = atomicAdd(&node_expand_inflight[idx2(tree, node, nodes)], 1);
        int slot = start_slot + ticket;
        if (slot < limit_slot) {
            expand_slot = slot;
        } else {
            atomicSub(&node_expand_inflight[idx2(tree, node, nodes)], 1);
        }
    }
    return __shfl_sync(FULL_MASK, expand_slot, 0);
}

static __device__ __forceinline__ void best_edge_winner_recalc(
    float cpuct, const float *edge_prior, const float *edge_W, const int *edge_N,
    int *edge_inflight, int tree, int node, int cur_expanded, int lane,
    float sqrt_parent_n_eff, int tie_offset, int soft_winner, int nodes, int edges,
    int &best_eid, int &held) {
    best_eid = INT32_MAX_V;
    held = 0;
    int retry = 0;

    while (retry < MAX_RECALC_RETRY) {
        int best_inflight = 0;
        reduce_best_edge(
            cpuct, edge_prior, edge_W, edge_N, edge_inflight, tree, node,
            cur_expanded, lane, sqrt_parent_n_eff, tie_offset, nodes, edges,
            best_eid, best_inflight);

        if (best_eid == INT32_MAX_V) {
            break;
        }

        if (lane == 0) {
            int *ptr = &edge_inflight[idx3(tree, node, best_eid, nodes, edges)];
            int prev = atomicCAS(ptr, best_inflight, best_inflight + 1);
            held = (prev == best_inflight) ? 1 : 0;
        }
        held = __shfl_sync(FULL_MASK, held, 0);

        if (held == 1 || retry + 1 >= MAX_RECALC_RETRY) {
            break;
        }
        retry += 1;
    }

    if (soft_winner != 0 && held == 0 && best_eid != INT32_MAX_V) {
        if (lane == 0) {
            atomicAdd(&edge_inflight[idx3(tree, node, best_eid, nodes, edges)], 1);
        }
        held = 1;
    }
    held = __shfl_sync(FULL_MASK, held, 0);
}

extern "C" __global__ void puct_select_winner_recalc_kernel(
    float cpuct,
    float c_pw,
    float alpha_pw,
    int soft_winner,
    const int *__restrict__ edge_child_id,
    const float *__restrict__ edge_prior,
    const float *__restrict__ edge_W,
    const int *__restrict__ edge_N,
    int *__restrict__ edge_inflight,
    int *__restrict__ node_expand_inflight,
    const int *__restrict__ node_expanded,
    const int *__restrict__ tree_nodes,
    int *__restrict__ out_selected_node,
    int *__restrict__ out_path_eids,
    int *__restrict__ out_path_len,
    int trees,
    int nodes,
    int edges,
    int warps,
    int path_depth) {
    int tree = blockIdx.x;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    if (tree >= trees || wid >= min(blockDim.x >> 5, warps)) {
        return;
    }

    if (lane == 0) {
        out_selected_node[idx_out(tree, wid, warps)] =
            pack_selection(SELECT_INVALID, PACKED_NODE_MASK, 0, REASON_UNSET);
        out_path_len[idx_out(tree, wid, warps)] = 0;
    }

    int node_cnt = 0;
    if (lane == 0) {
        node_cnt = tree_nodes[tree];
    }
    node_cnt = __shfl_sync(FULL_MASK, node_cnt, 0);

    if (!valid_select_shape(path_depth, nodes, node_cnt, edges)) {
        if (lane == 0) {
            out_selected_node[idx_out(tree, wid, warps)] =
                pack_selection(SELECT_INVALID, PACKED_NODE_MASK, 0, REASON_INVALID_SHAPE);
        }
        return;
    }

    int final_packed = pack_selection(SELECT_INVALID, PACKED_NODE_MASK, 0, REASON_INVALID_UNKNOWN);
    int final_len = 0;
    int node = 0;
    int depth = 0;

    while (true) {
        int node_info = node_expanded[idx2(tree, node, nodes)];
        if (node_info == NODE_EXPANDED_TERMINAL) {
            final_packed = pack_selection(SELECT_TERMINAL, node, 0, REASON_OK_TERMINAL);
            final_len = depth + 1;
            break;
        }

        if (node_info < 0 || node_info > edges) {
            rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight,
                                1, node_cnt, nodes, edges, warps, path_depth);
            __syncwarp(FULL_MASK);
            final_packed = pack_selection(SELECT_INVALID, PACKED_NODE_MASK, 0, REASON_INVALID_NODE_INFO);
            break;
        }

        if (depth >= path_depth) {
            final_packed = pack_selection(SELECT_DEPTH_LIMIT, node, 0, REASON_OK_DEPTH_LIMIT);
            final_len = depth + 1;
            break;
        }

        int cur_expanded = node_info;
        int parent_eff = 1;
        int allowed = 1;
        if (cur_expanded != 1 || c_pw != 0.0f) {
            parent_eff = parent_n_eff(edge_N, edge_inflight, tree, node, cur_expanded, lane, nodes, edges);
            allowed = allowed_children(c_pw, alpha_pw, parent_eff, edges);
        }

        if (allowed > cur_expanded) {
            int expand_inflight = 0;
            if (lane == 0) {
                expand_inflight = node_expand_inflight[idx2(tree, node, nodes)];
            }
            expand_inflight = __shfl_sync(FULL_MASK, expand_inflight, 0);

            int expand_slot = -1;
            if (cur_expanded + expand_inflight < allowed) {
                expand_slot = claim_expand_slot(
                    node_expand_inflight, tree, node, cur_expanded, allowed, lane, nodes);
            }
            if (expand_slot >= 0) {
                final_packed = pack_selection(SELECT_EXPAND, node, expand_slot, REASON_OK_EXPAND);
                final_len = depth + 1;
                break;
            }

            if (cur_expanded == 0) {
                rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight,
                                    1, node_cnt, nodes, edges, warps, path_depth);
                __syncwarp(FULL_MASK);
                final_packed = pack_selection(SELECT_BUSY, node, 0, REASON_BUSY_EXPAND_INFLIGHT);
                final_len = 0;
                break;
            }
        }

        int tie_offset = wid;
        if (tie_offset >= cur_expanded) {
            tie_offset = tie_offset % cur_expanded;
        }
        int best_eid = 0;
        int claim_held = 0;
        if (cur_expanded == 1) {
            if (lane == 0) {
                atomicAdd(&edge_inflight[idx3(tree, node, 0, nodes, edges)], 1);
                claim_held = 1;
            }
            claim_held = __shfl_sync(FULL_MASK, claim_held, 0);
        } else {
            float sqrt_parent = sqrtf((float)parent_eff);
            best_edge_winner_recalc(
                cpuct, edge_prior, edge_W, edge_N, edge_inflight, tree, node,
                cur_expanded, lane, sqrt_parent, tie_offset, soft_winner,
                nodes, edges, best_eid, claim_held);
        }

        if (claim_held == 0) {
            __syncwarp(FULL_MASK);
            rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight,
                                1, node_cnt, nodes, edges, warps, path_depth);
            __syncwarp(FULL_MASK);
            if (best_eid == INT32_MAX_V) {
                final_packed = pack_selection(SELECT_INVALID, PACKED_NODE_MASK, 0, REASON_INVALID_NO_VALID_EDGE);
            } else {
                final_packed = pack_selection(SELECT_BUSY, node, 0, REASON_BUSY_WINNER_RECALC);
            }
            break;
        }

        int best_child = edge_child_id[idx3(tree, node, best_eid, nodes, edges)];
        if (best_child < 0 || best_child >= node_cnt) {
            if (lane == 0) {
                atomicSub(&edge_inflight[idx3(tree, node, best_eid, nodes, edges)], 1);
            }
            __syncwarp(FULL_MASK);
            rollback_vloss_path(tree, wid, depth, lane, out_path_eids, edge_inflight,
                                1, node_cnt, nodes, edges, warps, path_depth);
            __syncwarp(FULL_MASK);
            final_packed = pack_selection(SELECT_INVALID, PACKED_NODE_MASK, 0, REASON_INVALID_CHILD_OOB);
            break;
        }

        if (lane == 0) {
            out_path_eids[idx_out_path(tree, wid, depth, warps, path_depth)] = (node << 8) | best_eid;
        }
        __syncwarp(FULL_MASK);

        node = best_child;
        depth += 1;
    }

    if (lane == 0) {
        out_selected_node[idx_out(tree, wid, warps)] = final_packed;
        out_path_len[idx_out(tree, wid, warps)] = final_len;
    }
}

extern "C" int puct_select_winner_recalc_launch(
    float cpuct,
    float c_pw,
    float alpha_pw,
    int soft_winner,
    const int *edge_child_id,
    const float *edge_prior,
    const float *edge_W,
    const int *edge_N,
    int *edge_inflight,
    int *node_expand_inflight,
    const int *node_expanded,
    const int *tree_nodes,
    int *out_selected_node,
    int *out_path_eids,
    int *out_path_len,
    int trees,
    int nodes,
    int edges,
    int warps,
    int path_depth,
    void *stream) {
    dim3 grid(trees);
    dim3 block(warps * WARP_SIZE);
    puct_select_winner_recalc_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        cpuct, c_pw, alpha_pw, soft_winner,
        edge_child_id, edge_prior, edge_W, edge_N, edge_inflight,
        node_expand_inflight, node_expanded, tree_nodes,
        out_selected_node, out_path_eids, out_path_len,
        trees, nodes, edges, warps, path_depth);
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char *puct_cpp_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
