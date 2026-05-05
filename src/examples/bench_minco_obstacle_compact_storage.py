"""Compare padded obstacle storage vs compact flat-edge storage.

This benchmark isolates the obstacle geometry hot path:

* padded: obstacle-wise padded tensors + mask + ``amax``
* compact: flat edge tensors + ``segment_reduce`` by obstacle lengths

The goal is to test whether compact storage improves throughput for
collision-related queries.
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

import torch

from examples.bench_minco_env_throughput import _complex_obstacles, _sync
from examples.minco_torch.constants import NDIM
from examples.minco_torch.scene import MincoScene


@dataclass(frozen=True)
class StorageBenchResult:
    device: str
    batch: int
    agents: int
    obstacles: int
    max_edges: int
    warmup: int
    repeats: int
    mode: str
    op: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    cases_per_s: float
    peak_mem_mb: float
    max_abs_err: float
    equal_mask: bool


def _make_scene(
    *,
    device: torch.device,
    num_agents: int,
    obstacles: list[list[list[float]]] | None,
) -> MincoScene:
    return MincoScene(
        num_agents=num_agents,
        collision_radius=0.0,
        position_bounds=((-6.0, 6.0), (-4.0, 4.0)),
        team_ids=None,
        collide_same_team=False,
        bounds_check_active_only=False,
        obstacle_vertices=obstacles,
        obstacle_vertex_counts=None,
        obstacle_collision_margin=0.04 if obstacles else 0.0,
        obstacle_projection_check_active_only=False,
        max_obstacle_vertices=8,
        dtype=torch.float32,
        device=device,
    )


def _make_positions(batch: int, agents: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(-5.4, 5.4, batch, dtype=torch.float32, device=device)
    items = []
    for agent in range(agents):
        phase = float(agent) * 0.83
        x = torch.roll(base, shifts=agent * max(1, batch // max(agents, 1)))
        y = 2.8 * torch.sin(base * 0.7 + phase)
        items.append(torch.stack((x, y), dim=-1))
    return torch.stack(items, dim=1)


def _compact_buffers(scene: MincoScene, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact flat-edge buffers once for the benchmark."""
    n, b, em, _, _ = scene.obstacle_buffers(reference)
    cnt = scene.obstacle_vertex_counts
    flat = em.reshape(-1)
    flat_n = n.reshape(-1, NDIM)[flat]
    flat_b = b.reshape(-1)[flat]
    return flat_n, flat_b, cnt


def _compact_clearance(
    positions: torch.Tensor,
    flat_n: torch.Tensor,
    flat_b: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    sd = torch.einsum("ed,...d->e...", flat_n, positions) - flat_b.view(-1, *([1] * (positions.ndim - 1)))
    clr = torch.segment_reduce(sd, "amax", lengths=lengths, axis=0)
    return clr.movedim(0, -1)


def _time_call(
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
):
    out = fn()
    for _ in range(warmup):
        out = fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples: list[float] = []
    last = out
    for _ in range(repeats):
        start = time.perf_counter()
        last = fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return samples, last, peak_mem_mb


def _bench_one(
    *,
    scene: MincoScene,
    flat_n: torch.Tensor,
    flat_b: torch.Tensor,
    lengths: torch.Tensor,
    batch: int,
    warmup: int,
    repeats: int,
) -> list[StorageBenchResult]:
    device = scene.obstacle_edge_normals.device
    pos = _make_positions(batch, scene.num_agents, device)
    act = torch.ones((batch, scene.num_agents), dtype=torch.bool, device=device)
    margin = torch.as_tensor(scene.obstacle_collision_margin, dtype=pos.dtype, device=pos.device)

    def padded_clearance():
        with torch.inference_mode():
            return scene.obstacle_clearance(pos)

    def compact_clearance():
        with torch.inference_mode():
            return _compact_clearance(pos, flat_n, flat_b, lengths)

    def padded_mask():
        with torch.inference_mode():
            return scene.obstacle_collision_mask(pos, act)

    def compact_mask():
        with torch.inference_mode():
            return act & (_compact_clearance(pos, flat_n, flat_b, lengths) <= margin).any(dim=-1)

    pad_clr = padded_clearance()
    cmp_clr = compact_clearance()
    max_abs_err = float((pad_clr - cmp_clr).abs().max().detach().cpu())

    pad_msk = padded_mask()
    cmp_msk = compact_mask()
    equal_mask = bool(torch.equal(pad_msk, cmp_msk))

    rows: list[StorageBenchResult] = []
    for mode, fn in (("padded", padded_clearance), ("compact", compact_clearance), ("padded", padded_mask), ("compact", compact_mask)):
        samples, _, peak_mem_mb = _time_call(fn, device=device, warmup=warmup, repeats=repeats)
        mean_ms = statistics.fmean(samples)
        median_ms = statistics.median(samples)
        p95_ms = sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]
        cases_per_s = batch / max(mean_ms / 1000.0, 1e-12)
        rows.append(
            StorageBenchResult(
                device=str(device),
                batch=batch,
                agents=scene.num_agents,
                obstacles=scene.num_obstacles,
                max_edges=scene.max_obstacle_vertices,
                warmup=warmup,
                repeats=repeats,
                mode=mode,
                op="clearance" if fn in (padded_clearance, compact_clearance) else "mask",
                mean_ms=mean_ms,
                median_ms=median_ms,
                p95_ms=p95_ms,
                cases_per_s=cases_per_s,
                peak_mem_mb=peak_mem_mb,
                max_abs_err=max_abs_err,
                equal_mask=equal_mask,
            )
        )
    return rows


def _print_table(rows: list[StorageBenchResult]) -> None:
    print(
        f"{'op':>10} | {'mode':>7} | {'batch':>7} | {'ms':>8} | {'cases/s':>12} | "
        f"{'p95':>8} | {'err':>10} | {'mask':>5}"
    )
    print("-" * 88)
    for row in rows:
        print(
            f"{row.op:>10} | {row.mode:>7} | {row.batch:7d} | {row.mean_ms:8.3f} | "
            f"{row.cases_per_s:12.0f} | {row.p95_ms:8.3f} | {row.max_abs_err:10.3e} | {str(row.equal_mask):>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs/minco_obstacle_compact_storage")
    parser.add_argument("--batches", default=None, help="Comma-separated batch sizes.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--obstacles", type=int, default=16)
    args = parser.parse_args()

    if args.device == "cuda":
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                device = torch.device("cuda")
            except RuntimeError:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.batches is None:
        batches = [4096, 16384, 65536] if device.type == "cuda" else [1024, 4096, 16384, 65536]
    else:
        batches = [int(x) for x in args.batches.split(",") if x.strip()]

    scene = _make_scene(
        device=device,
        num_agents=args.agents,
        obstacles=_complex_obstacles(args.obstacles),
    )
    ref = torch.zeros((1, args.agents, NDIM), dtype=torch.float32, device=device)
    flat_n, flat_b, lengths = _compact_buffers(scene, ref)

    rows: list[StorageBenchResult] = []
    for batch in batches:
        rows.extend(
            _bench_one(
                scene=scene,
                flat_n=flat_n,
                flat_b=flat_b,
                lengths=lengths,
                batch=batch,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
        _print_table(rows[-4:])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"minco_obstacle_compact_storage_{device.type}.json"
    csv_path = out_dir / f"minco_obstacle_compact_storage_{device.type}.csv"
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print("\nSummary")
    _print_table(rows)
    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
