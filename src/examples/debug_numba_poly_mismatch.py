from __future__ import annotations

import numpy as np
from numba import cuda

import examples.bench_minco_polygon_collision_numba as bm


def main() -> None:
    batch = 4096
    agents = 4
    obs = 16
    margin = 0.04
    aabb_eps = 1e-5

    polys = bm._complex_obstacles(obs)
    edge_n_h, edge_b_h, obs_start_h, obs_count_h, obs_lo_h, obs_hi_h = bm._build_compact_obstacles(polys)
    pos_h = bm._make_positions(batch, agents)
    active_h = np.ones((batch, agents), dtype=np.bool_)
    ref = bm._cpu_reference(
        pos_h,
        active_h,
        edge_n_h,
        edge_b_h,
        obs_start_h,
        obs_count_h,
        np.float32(margin),
    )

    pos_d = cuda.to_device(pos_h)
    active_d = cuda.to_device(active_h)
    edge_n_d = cuda.to_device(edge_n_h)
    edge_b_d = cuda.to_device(edge_b_h)
    obs_start_d = cuda.to_device(obs_start_h)
    obs_count_d = cuda.to_device(obs_count_h)
    obs_lo_d = cuda.to_device(obs_lo_h)
    obs_hi_d = cuda.to_device(obs_hi_h)
    out_n = cuda.device_array((batch, agents), dtype=np.bool_)
    out_b = cuda.device_array((batch, agents), dtype=np.bool_)

    total = batch * agents
    blocks = (total + bm.THREADS_PER_BLOCK - 1) // bm.THREADS_PER_BLOCK
    bm.collision_narrow_kernel[blocks, bm.THREADS_PER_BLOCK](
        pos_d, active_d, edge_n_d, edge_b_d, obs_start_d, obs_count_d, np.float32(margin), out_n
    )
    bm.collision_broad_narrow_kernel[blocks, bm.THREADS_PER_BLOCK](
        pos_d,
        active_d,
        edge_n_d,
        edge_b_d,
        obs_start_d,
        obs_count_d,
        obs_lo_d,
        obs_hi_d,
        np.float32(margin),
        np.float32(margin + aabb_eps),
        out_b,
    )
    cuda.synchronize()
    n = out_n.copy_to_host()
    b = out_b.copy_to_host()
    print("n mismatch", int(np.count_nonzero(n != ref)))
    print("b mismatch", int(np.count_nonzero(b != ref)))
    mis = np.argwhere(b != ref)
    print("mis count", mis.shape)
    if mis.size == 0:
        return
    i, j = mis[0]
    x, y = pos_h[i, j]
    print("idx", int(i), int(j), "pos", (float(x), float(y)), "ref", bool(ref[i, j]), "b", bool(b[i, j]), "n", bool(n[i, j]))
    coll = []
    for o in range(obs_start_h.shape[0]):
        s = int(obs_start_h[o])
        c = int(obs_count_h[o])
        sd = x * edge_n_h[s:s + c, 0] + y * edge_n_h[s:s + c, 1] - edge_b_h[s:s + c]
        mx = float(np.max(sd))
        if mx <= margin:
            coll.append((o, mx))
    print("coll obs", coll)
    near = []
    am = margin + aabb_eps
    for o in range(obs_start_h.shape[0]):
        if not (
            x < obs_lo_h[o, 0] - am
            or x > obs_hi_h[o, 0] + am
            or y < obs_lo_h[o, 1] - am
            or y > obs_hi_h[o, 1] + am
        ):
            near.append(o)
    print("near obs", near)


if __name__ == "__main__":
    main()
