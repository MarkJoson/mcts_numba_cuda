"""Benchmark polygon collision kernels with Numba CUDA.

This script compares two GPU kernels for point-vs-convex-polygon collision:

* ``narrow``: check every obstacle edge directly
* ``broad_narrow``: AABB broad phase + narrow edge check

The data layout is compact:

* ``edge_n``: flat outward normals, shape ``(total_edges, 2)``
* ``edge_b``: flat half-space offsets, shape ``(total_edges,)``
* ``obs_start``: per-obstacle edge start index, shape ``(num_obs,)``
* ``obs_count``: per-obstacle edge count, shape ``(num_obs,)``
* ``obs_lo`` / ``obs_hi``: per-obstacle AABB, shape ``(num_obs, 2)``

The goal is to see whether a fused broad/narrow CUDA kernel improves
throughput over the current PyTorch obstacle path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from numba import cuda


NEG_INF_F32 = np.float32(-3.4028235e38)
THREADS_PER_BLOCK = 256


@dataclass(frozen=True)
class PolyBenchResult:
    device: str
    batch: int
    agents: int
    obstacles: int
    edges: int
    warmup: int
    repeats: int
    mode: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    cases_per_s: float
    collisions: int
    ref_ok: bool
    max_abs_err: float
    mismatch_count: int


def _pick_device(requested: str) -> None:
    if not cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if requested == "cuda" or requested == "auto":
        return
    raise ValueError(f"unsupported device: {requested}")


def _complex_obstacles(count: int) -> list[list[list[float]]]:
    centers = [
        (-4.7, -2.7), (-3.0, -1.6), (-1.2, -2.7), (0.8, -1.7),
        (2.7, -2.6), (4.6, -1.4), (-4.8, 0.2), (-2.8, 0.8),
        (-0.7, 0.2), (1.5, 0.8), (3.5, 0.4), (4.8, 2.2),
        (-3.8, 2.6), (-1.1, 2.7), (1.8, 2.7), (3.8, 2.8),
    ]
    polys: list[list[list[float]]] = []
    for idx, (cx, cy) in enumerate(centers[:count]):
        sides = 4 + (idx % 4)
        rx = 0.42 + 0.10 * (idx % 3)
        ry = 0.55 + 0.08 * ((idx + 1) % 3)
        theta0 = 0.31 * idx
        pts = []
        for k in range(sides):
            theta = theta0 + math.tau * k / sides
            pts.append([cx + rx * math.cos(theta), cy + ry * math.sin(theta)])
        polys.append(pts)
    return polys


def _build_compact_obstacles(
    polys: list[list[list[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = []
    counts = []
    lo_list = []
    hi_list = []
    n_list = []
    b_list = []
    start = 0
    for poly in polys:
        v = np.asarray(poly, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError("each polygon must have shape (vertices, 2)")
        cnt = int(v.shape[0])
        if cnt < 3:
            raise ValueError("each obstacle must have at least 3 vertices")
        nv = np.roll(v, shift=-1, axis=0)
        ed = nv - v
        el = np.linalg.norm(ed, axis=-1)
        if np.any(el <= 1e-12):
            raise ValueError("zero-length edge detected")
        cr = v[:, 0] * nv[:, 1] - v[:, 1] * nv[:, 0]
        area2 = float(np.sum(cr))
        if abs(area2) <= 1e-12:
            raise ValueError("polygon must have non-zero signed area")
        ori = 1.0 if area2 >= 0.0 else -1.0
        n = np.stack((ed[:, 1], -ed[:, 0]), axis=-1) * ori
        n = n / el[:, None]
        b = np.sum(v * n, axis=-1)
        starts.append(start)
        counts.append(cnt)
        start += cnt
        lo_list.append(np.min(v, axis=0))
        hi_list.append(np.max(v, axis=0))
        n_list.append(n.astype(np.float32))
        b_list.append(b.astype(np.float32))
    edge_n = np.concatenate(n_list, axis=0).astype(np.float32, copy=False)
    edge_b = np.concatenate(b_list, axis=0).astype(np.float32, copy=False)
    obs_start = np.asarray(starts, dtype=np.int32)
    obs_count = np.asarray(counts, dtype=np.int32)
    obs_lo = np.asarray(lo_list, dtype=np.float32)
    obs_hi = np.asarray(hi_list, dtype=np.float32)
    return edge_n, edge_b, obs_start, obs_count, obs_lo, obs_hi


def _expanded_aabb_from_halfspaces(
    edge_n: np.ndarray,
    edge_b: np.ndarray,
    obs_start: np.ndarray,
    obs_count: np.ndarray,
    margin: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-obstacle AABB after shifting all edge constraints by margin."""
    n_obs = int(obs_start.shape[0])
    lo = np.empty((n_obs, 2), dtype=np.float32)
    hi = np.empty((n_obs, 2), dtype=np.float32)
    tol = float(eps)
    for o in range(n_obs):
        s = int(obs_start[o])
        c = int(obs_count[o])
        n = edge_n[s:s + c].astype(np.float64, copy=False)
        b = edge_b[s:s + c].astype(np.float64, copy=False) + float(margin)
        cand: list[tuple[float, float]] = []
        for i in range(c):
            ni0 = float(n[i, 0])
            ni1 = float(n[i, 1])
            bi = float(b[i])
            for j in range(i + 1, c):
                nj0 = float(n[j, 0])
                nj1 = float(n[j, 1])
                bj = float(b[j])
                det = ni0 * nj1 - ni1 * nj0
                if abs(det) <= 1e-12:
                    continue
                inv_det = 1.0 / det
                x = (bi * nj1 - ni1 * bj) * inv_det
                y = (ni0 * bj - bi * nj0) * inv_det
                if np.all(n[:, 0] * x + n[:, 1] * y - b <= tol):
                    cand.append((x, y))
        if len(cand) == 0:
            # Conservative fallback: expand by margin in XY.
            v_lo = np.array([np.inf, np.inf], dtype=np.float64)
            v_hi = np.array([-np.inf, -np.inf], dtype=np.float64)
            for i in range(c):
                for j in range(i + 1, c):
                    ni0 = float(n[i, 0])
                    ni1 = float(n[i, 1])
                    bi = float(edge_b[s + i])
                    nj0 = float(n[j, 0])
                    nj1 = float(n[j, 1])
                    bj = float(edge_b[s + j])
                    det = ni0 * nj1 - ni1 * nj0
                    if abs(det) <= 1e-12:
                        continue
                    inv_det = 1.0 / det
                    x = (bi * nj1 - ni1 * bj) * inv_det
                    y = (ni0 * bj - bi * nj0) * inv_det
                    v_lo[0] = min(v_lo[0], x)
                    v_lo[1] = min(v_lo[1], y)
                    v_hi[0] = max(v_hi[0], x)
                    v_hi[1] = max(v_hi[1], y)
            lo[o, 0] = np.float32(v_lo[0] - margin - eps)
            lo[o, 1] = np.float32(v_lo[1] - margin - eps)
            hi[o, 0] = np.float32(v_hi[0] + margin + eps)
            hi[o, 1] = np.float32(v_hi[1] + margin + eps)
        else:
            arr = np.asarray(cand, dtype=np.float64)
            lo[o, 0] = np.float32(np.min(arr[:, 0]) - eps)
            lo[o, 1] = np.float32(np.min(arr[:, 1]) - eps)
            hi[o, 0] = np.float32(np.max(arr[:, 0]) + eps)
            hi[o, 1] = np.float32(np.max(arr[:, 1]) + eps)
    return lo, hi


def _make_positions(batch: int, agents: int) -> np.ndarray:
    base = np.linspace(-5.4, 5.4, batch, dtype=np.float32)
    pos = np.empty((batch, agents, 2), dtype=np.float32)
    for agent in range(agents):
        phase = float(agent) * 0.83
        shift = agent * max(1, batch // max(agents, 1))
        x = np.roll(base, shift=shift)
        y = np.float32(2.8) * np.sin(base * np.float32(0.7) + np.float32(phase)).astype(np.float32)
        pos[:, agent, 0] = x
        pos[:, agent, 1] = y
    return pos


@cuda.jit(device=True, inline=True)
def _edge_hit(x, y, start, count, edge_n, edge_b, margin):
    for e in range(start, start + count):
        sd = x * edge_n[e, 0] + y * edge_n[e, 1] - edge_b[e]
        if sd > margin:
            return False
    return True


@cuda.jit
def collision_narrow_kernel(pos, active, edge_n, edge_b, obs_start, obs_count, margin, out):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = pos.shape[0] * pos.shape[1]
    n_agents = pos.shape[1]
    n_obs = obs_start.shape[0]
    while idx < total:
        b = idx // n_agents
        a = idx - b * n_agents
        coll = False
        if active[b, a]:
            x = pos[b, a, 0]
            y = pos[b, a, 1]
            for o in range(n_obs):
                start = obs_start[o]
                count = obs_count[o]
                if _edge_hit(x, y, start, count, edge_n, edge_b, margin):
                    coll = True
                    break
        out[b, a] = coll
        idx += stride


@cuda.jit
def collision_broad_narrow_kernel(
    pos,
    active,
    edge_n,
    edge_b,
    obs_start,
    obs_count,
    obs_lo,
    obs_hi,
    margin,
    out,
):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = pos.shape[0] * pos.shape[1]
    n_agents = pos.shape[1]
    n_obs = obs_start.shape[0]
    while idx < total:
        b = idx // n_agents
        a = idx - b * n_agents
        coll = False
        if active[b, a]:
            x = pos[b, a, 0]
            y = pos[b, a, 1]
            for o in range(n_obs):
                if x < obs_lo[o, 0] or x > obs_hi[o, 0] or y < obs_lo[o, 1] or y > obs_hi[o, 1]:
                    continue
                start = obs_start[o]
                count = obs_count[o]
                if _edge_hit(x, y, start, count, edge_n, edge_b, margin):
                    coll = True
                    break
        out[b, a] = coll
        idx += stride


def _launch(kernel, blocks: int, threads: int, *args) -> np.ndarray:
    out = args[-1]
    kernel[blocks, threads](*args)
    cuda.synchronize()
    return out.copy_to_host()


def _time_kernel(
    kernel,
    *,
    blocks: int,
    threads: int,
    args: tuple,
    warmup: int,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    out = args[-1]
    for _ in range(warmup):
        kernel[blocks, threads](*args)
    cuda.synchronize()
    samples: list[float] = []
    last = out
    for _ in range(repeats):
        start = time.perf_counter()
        kernel[blocks, threads](*args)
        cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        last = out
    return samples, last.copy_to_host()


def _cpu_reference(
    pos: np.ndarray,
    active: np.ndarray,
    edge_n: np.ndarray,
    edge_b: np.ndarray,
    obs_start: np.ndarray,
    obs_count: np.ndarray,
    margin: np.float32,
) -> np.ndarray:
    batch, agents, _ = pos.shape
    out = np.zeros((batch, agents), dtype=np.bool_)
    for o in range(obs_start.shape[0]):
        start = int(obs_start[o])
        count = int(obs_count[o])
        n = edge_n[start : start + count]
        b = edge_b[start : start + count]
        sd = np.einsum("bad,ed->bae", pos, n) - b.reshape(1, 1, -1)
        hit = np.max(sd, axis=-1) <= margin
        out |= hit
    out &= active
    return out


def _bench_one(
    *,
    batch: int,
    agents: int,
    edge_n_h: np.ndarray,
    edge_b_h: np.ndarray,
    obs_start_h: np.ndarray,
    obs_count_h: np.ndarray,
    obs_lo_h: np.ndarray,
    obs_hi_h: np.ndarray,
    margin: float,
    warmup: int,
    repeats: int,
) -> list[PolyBenchResult]:
    pos_h = _make_positions(batch, agents)
    active_h = np.ones((batch, agents), dtype=np.bool_)
    ref = _cpu_reference(
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
    out_d = cuda.device_array((batch, agents), dtype=np.bool_)

    total = batch * agents
    blocks = (total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    rows: list[PolyBenchResult] = []
    for mode, kernel, args in (
        ("narrow", collision_narrow_kernel, (pos_d, active_d, edge_n_d, edge_b_d, obs_start_d, obs_count_d, np.float32(margin), out_d)),
        (
            "broad_narrow",
            collision_broad_narrow_kernel,
            (
                pos_d,
                active_d,
                edge_n_d,
                edge_b_d,
                obs_start_d,
                obs_count_d,
                obs_lo_d,
                obs_hi_d,
                np.float32(margin),
                out_d,
            ),
        ),
    ):
        samples, out_h = _time_kernel(kernel, blocks=blocks, threads=THREADS_PER_BLOCK, args=args, warmup=warmup, repeats=repeats)
        mismatch = int(np.count_nonzero(out_h != ref))
        ok = mismatch == 0
        max_abs_err = 0.0
        mean_ms = statistics.fmean(samples)
        median_ms = statistics.median(samples)
        p95_ms = sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]
        cases_per_s = batch / max(mean_ms / 1000.0, 1e-12)
        rows.append(
            PolyBenchResult(
                device="cuda",
                batch=batch,
                agents=agents,
                obstacles=obs_start_h.shape[0],
                edges=edge_n_h.shape[0],
                warmup=warmup,
                repeats=repeats,
                mode=mode,
                mean_ms=mean_ms,
                median_ms=median_ms,
                p95_ms=p95_ms,
                cases_per_s=cases_per_s,
                collisions=int(out_h.sum()),
                ref_ok=ok,
                max_abs_err=max_abs_err,
                mismatch_count=mismatch,
            )
        )
    # Attach a quick sanity check in the terminal output.
    if not rows[0].ref_ok or not rows[1].ref_ok:
        raise AssertionError(
            "kernel output mismatch vs CPU reference: "
            f"narrow={rows[0].mismatch_count}, broad_narrow={rows[1].mismatch_count}"
        )
    return rows


def _print_table(rows: list[PolyBenchResult]) -> None:
    print(
        f"{'mode':>12} | {'batch':>7} | {'ms':>8} | {'cases/s':>12} | {'coll':>6} | "
        f"{'ref':>4} | {'mis':>5} | {'p95':>8}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{row.mode:>12} | {row.batch:7d} | {row.mean_ms:8.3f} | {row.cases_per_s:12.0f} | "
            f"{row.collisions:6d} | {str(row.ref_ok):>4} | {row.mismatch_count:5d} | {row.p95_ms:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda"])
    parser.add_argument("--output-dir", default="outputs/minco_polygon_collision_numba")
    parser.add_argument("--batches", default=None, help="Comma-separated batch sizes.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--obstacles", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--aabb-eps", type=float, default=1e-5)
    args = parser.parse_args()

    _pick_device(args.device)
    if args.batches is None:
        batches = [4096, 16384, 65536]
    else:
        batches = [int(x) for x in args.batches.split(",") if x.strip()]

    polys = _complex_obstacles(args.obstacles)
    edge_n_h, edge_b_h, obs_start_h, obs_count_h, _, _ = _build_compact_obstacles(polys)
    obs_lo_h, obs_hi_h = _expanded_aabb_from_halfspaces(
        edge_n_h,
        edge_b_h,
        obs_start_h,
        obs_count_h,
        margin=args.margin,
        eps=args.aabb_eps,
    )

    results: list[PolyBenchResult] = []
    for batch in batches:
        rows = _bench_one(
            batch=batch,
            agents=args.agents,
            edge_n_h=edge_n_h,
            edge_b_h=edge_b_h,
            obs_start_h=obs_start_h,
            obs_count_h=obs_count_h,
            obs_lo_h=obs_lo_h,
            obs_hi_h=obs_hi_h,
            margin=args.margin,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.extend(rows)
        _print_table(rows)
        narrow_ms = rows[0].mean_ms
        broad_ms = rows[1].mean_ms
        print(f"speedup broad_narrow vs narrow: {narrow_ms / max(broad_ms, 1e-12):.2f}x")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "minco_polygon_collision_numba_cuda.json"
    csv_path = out_dir / "minco_polygon_collision_numba_cuda.csv"
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    print("\nSummary")
    _print_table(results)
    for i in range(0, len(results), 2):
        narrow = results[i]
        broad = results[i + 1]
        print(
            f"batch {narrow.batch}: broad_narrow {narrow.mean_ms / max(broad.mean_ms, 1e-12):.2f}x "
            f"vs narrow"
        )
    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
