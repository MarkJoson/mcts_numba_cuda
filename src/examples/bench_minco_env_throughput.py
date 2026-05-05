"""Throughput benchmark for batched MINCO environment rollout/expand.

The benchmark separates the main costs relevant to MCTS simulation:

* ``step_flat`` on contiguous states;
* ``gather_step_flat`` from a synthetic ``(trees, nodes, state_dim)`` pool;
* increasingly expensive environment features: pure MINCO, point collision,
  obstacle collision, and obstacle target projection.

Example:
    PYTHONPATH=src conda run -n py312_numba python \
        src/examples/bench_minco_env_throughput.py --device auto
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
from typing import Callable, Sequence

import torch

from examples.minco_torch_transition import MincoPointEnvTransition


@dataclass(frozen=True)
class BenchResult:
    device: str
    scenario: str
    path: str
    batch: int
    agents: int
    obstacles: int
    max_edges: int
    projection: bool
    return_info: bool
    warmup: int
    repeats: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    states_per_s: float
    agent_steps_per_s: float
    bytes_state_mb: float
    peak_mem_mb: float
    collision_count: int
    residual_count: int
    done_count: int


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda":
            torch.zeros(1, device=device)
        return device
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
        except RuntimeError:
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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
        points = []
        for k in range(sides):
            theta = theta0 + math.tau * k / sides
            points.append([cx + rx * math.cos(theta), cy + ry * math.sin(theta)])
        polys.append(points)
    return polys


def _make_env(
    *,
    device: torch.device,
    scenario: str,
    num_agents: int,
    obstacles: Sequence[Sequence[Sequence[float]]] | None,
    obstacle_target_projection: bool,
    point_collision_radius: float,
    n_checkpoints: int,
) -> MincoPointEnvTransition:
    return MincoPointEnvTransition(
        num_agents=num_agents,
        piece_t=0.1,
        dt=0.1,
        n_checkpoints=n_checkpoints,
        collision_radius=point_collision_radius,
        obstacle_vertices=obstacles,
        obstacle_collision_margin=0.04 if obstacles else 0.0,
        obstacle_target_projection=obstacle_target_projection,
        obstacle_projection_iters=3,
        obstacle_projection_extra_margin=0.02,
        obstacle_projection_topk=8,
        obstacle_projection_fixes_per_iter=2,
        velocity_limit=12.0,
        acceleration_limit=18.0,
        dtype=torch.float32,
        device=device,
        position_bounds=((-6.0, 6.0), (-4.0, 4.0)),
        collide_same_team=False,
        deactivate_on_collision=False,
        deactivate_on_obstacle_collision=False,
        done_on_obstacle_collision=False,
        done_on_out_of_bounds=False,
        done_on_team_eliminated=False,
        tf=1000.0,
    )


def _make_positions(batch: int, agents: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(-5.4, 5.4, batch, dtype=torch.float32, device=device)
    positions = []
    for agent in range(agents):
        phase = float(agent) * 0.83
        x = torch.roll(base, shifts=agent * max(1, batch // max(agents, 1)))
        y = 2.8 * torch.sin(base * 0.7 + phase)
        positions.append(torch.stack((x, y), dim=-1))
    return torch.stack(positions, dim=1)


def _make_targets(positions: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(positions[..., 1], positions[..., 0]) + 0.42
    radius = torch.linalg.norm(positions, dim=-1).clamp_min(0.5)
    target = torch.stack(
        (
            0.82 * radius * torch.cos(angle),
            0.82 * radius * torch.sin(angle),
        ),
        dim=-1,
    )
    return target


def _time_call(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, object | None]],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[list[float], tuple[torch.Tensor, torch.Tensor, object | None], float]:
    out = fn()
    for _ in range(warmup):
        out = fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples = []
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
    env: MincoPointEnvTransition,
    scenario: str,
    path: str,
    batch: int,
    warmup: int,
    repeats: int,
    return_info: bool,
) -> BenchResult:
    device = env.robot_transition.mat_f_stab.device
    positions = _make_positions(batch, env.num_agents, device)
    target = _make_targets(positions)
    flat = env.initial_flat_state(positions)
    pool = None
    tree_ids = None
    node_ids = None
    if path == "gather_step_flat":
        nodes = 8
        trees = max(1, (batch + nodes - 1) // nodes)
        pool_positions = _make_positions(trees * nodes, env.num_agents, device)
        pool = env.initial_flat_state(pool_positions).view(trees, nodes, env.state_dim)
        ids = torch.arange(batch, dtype=torch.long, device=device)
        tree_ids = torch.div(ids, nodes, rounding_mode="floor")
        node_ids = ids - tree_ids * nodes

    def call() -> tuple[torch.Tensor, torch.Tensor, object | None]:
        if path == "step_flat":
            out = env.step_flat(flat, target, return_info=return_info)
        elif path == "gather_step_flat":
            assert pool is not None and tree_ids is not None and node_ids is not None
            out = env.gather_step_flat(pool, tree_ids, node_ids, target, return_info=return_info)
        else:
            raise ValueError(f"unknown path: {path}")
        if return_info:
            next_flat, done, info = out
            return next_flat, done, info
        next_flat, done = out
        return next_flat, done, None

    samples, last, peak_mem_mb = _time_call(call, device=device, warmup=warmup, repeats=repeats)
    next_flat, done, info = last
    _sync(device)
    mean_ms = statistics.fmean(samples)
    median_ms = statistics.median(samples)
    p95_ms = sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]
    states_per_s = batch / max(mean_ms / 1000.0, 1e-12)
    collision_count = 0
    residual_count = 0
    if info is not None:
        collision_count = int(info.obstacle_collision_mask.sum().detach().cpu())
        residual_count = int(info.obstacle_projection_residual_mask.sum().detach().cpu())
    done_count = int(done.sum().detach().cpu())
    bytes_state_mb = next_flat.numel() * next_flat.element_size() / (1024.0 * 1024.0)
    return BenchResult(
        device=str(device),
        scenario=scenario,
        path=path,
        batch=batch,
        agents=env.num_agents,
        obstacles=env.num_obstacles,
        max_edges=env.max_obstacle_vertices,
        projection=env.obstacle_target_projection,
        return_info=return_info,
        warmup=warmup,
        repeats=repeats,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        states_per_s=states_per_s,
        agent_steps_per_s=states_per_s * env.num_agents,
        bytes_state_mb=bytes_state_mb,
        peak_mem_mb=peak_mem_mb,
        collision_count=collision_count,
        residual_count=residual_count,
        done_count=done_count,
    )


def _print_table(results: Sequence[BenchResult]) -> None:
    print(
        f"{'scenario':>20} | {'path':>16} | {'batch':>7} | {'ms':>8} | "
        f"{'states/s':>12} | {'agent/s':>12} | {'p95':>8} | {'peakMB':>8} | {'coll':>6} | {'resid':>6}"
    )
    print("-" * 130)
    for row in results:
        print(
            f"{row.scenario:>20} | {row.path:>16} | {row.batch:7d} | "
            f"{row.mean_ms:8.3f} | {row.states_per_s:12.0f} | {row.agent_steps_per_s:12.0f} | "
            f"{row.p95_ms:8.3f} | {row.peak_mem_mb:8.1f} | {row.collision_count:6d} | {row.residual_count:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs/minco_env_throughput")
    parser.add_argument("--batches", default=None, help="Comma-separated batch sizes.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, default=20)
    parser.add_argument("--return-info", action="store_true")
    args = parser.parse_args()

    device = _pick_device(args.device)
    if args.batches is None:
        batches = [4096, 16384, 65536, 131072] if device.type == "cuda" else [1024, 4096, 16384]
    else:
        batches = [int(x) for x in args.batches.split(",") if x.strip()]

    scenarios = [
        ("minco_only", None, False, 0.0),
        ("point_collision", None, False, 0.35),
        ("obstacle_4", _complex_obstacles(4), False, 0.0),
        ("obstacle_16", _complex_obstacles(16), False, 0.0),
        ("obstacle_16_project", _complex_obstacles(16), True, 0.0),
    ]
    results: list[BenchResult] = []
    for scenario, obstacles, project, point_radius in scenarios:
        env = _make_env(
            device=device,
            scenario=scenario,
            num_agents=args.agents,
            obstacles=obstacles,
            obstacle_target_projection=project,
            point_collision_radius=point_radius,
            n_checkpoints=args.checkpoints,
        )
        for batch in batches:
            for path in ("step_flat", "gather_step_flat"):
                try:
                    result = _bench_one(
                        env=env,
                        scenario=scenario,
                        path=path,
                        batch=batch,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        return_info=args.return_info,
                    )
                except RuntimeError as exc:
                    if device.type == "cuda" and "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                        print(f"SKIP OOM: scenario={scenario} path={path} batch={batch}: {exc}")
                        continue
                    raise
                results.append(result)
                _print_table([result])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"minco_env_throughput_{device.type}.json"
    csv_path = out_dir / f"minco_env_throughput_{device.type}.csv"
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    print("\nSummary")
    _print_table(results)
    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
