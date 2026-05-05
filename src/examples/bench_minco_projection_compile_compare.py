"""Benchmark torch.compile impact on MINCO projection critical paths.

This script compares eager vs compiled performance on:
1) projection_only: dynamic target clamp + obstacle target projection
2) step_flat: end-to-end env step (includes projection path)

All timings use warmup + repeated average (compile overhead excluded from timing).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

import torch

from examples.bench_minco_env_throughput import _complex_obstacles, _make_positions, _make_targets
from examples.minco_torch_transition import MincoPointEnvTransition


@dataclass(frozen=True)
class Row:
    path: str
    upd: str
    compiled: bool
    batch: int
    warmup: int
    iters: int
    avg_ms: float
    states_per_s: float
    agent_steps_per_s: float


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_env(*, device: torch.device, agents: int, checkpoints: int, obstacles: int) -> MincoPointEnvTransition:
    env = MincoPointEnvTransition(
        num_agents=agents,
        piece_t=0.1,
        dt=0.1,
        n_checkpoints=checkpoints,
        collision_radius=0.0,
        obstacle_vertices=_complex_obstacles(obstacles),
        obstacle_collision_margin=0.04,
        obstacle_target_projection=True,
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
    return env


def _bench_projection_only(
    env: MincoPointEnvTransition,
    *,
    batch: int,
    warmup: int,
    iters: int,
    compiled: bool,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> tuple[float, float]:
    pos = _make_positions(batch, env.num_agents, env.robot_transition.mat_f_stab.device)
    tgt = _make_targets(pos)
    coeff = env.initial_coefficients(pos)
    active = torch.ones((batch, env.num_agents), dtype=torch.bool, device=coeff.device)
    env.projection_ops.set_prof_nvtx(False)
    env.projection_ops.set_prof_select_stats(False, reset=True)

    def _proj_call(c: torch.Tensor, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        p = env.robot_transition.project_target(c, t)
        out, _, _ = env.project_target_away_from_obstacles(
            c,
            p,
            a,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
        return out

    if compiled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        run_fn = torch.compile(
            _proj_call,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )
    else:
        run_fn = _proj_call

    for _ in range(warmup):
        run_fn(coeff, tgt, active)
    _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_fn(coeff, tgt, active)
    _sync()
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = total_ms / max(iters, 1)
    states_per_s = batch / max(avg_ms / 1000.0, 1e-12)
    return avg_ms, states_per_s


def _bench_step_flat(
    env: MincoPointEnvTransition,
    *,
    batch: int,
    warmup: int,
    iters: int,
    compiled: bool,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> tuple[float, float]:
    pos = _make_positions(batch, env.num_agents, env.robot_transition.mat_f_stab.device)
    tgt = _make_targets(pos)
    flat = env.initial_flat_state(pos)
    env.projection_ops.set_prof_nvtx(False)
    env.projection_ops.set_prof_select_stats(False, reset=True)

    def _step_call(f: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return env.step_flat(f, t, return_info=False)

    if compiled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        run_fn = torch.compile(
            _step_call,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )
    else:
        run_fn = _step_call

    for _ in range(warmup):
        _ = run_fn(flat, tgt)
    _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = run_fn(flat, tgt)
    _sync()
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = total_ms / max(iters, 1)
    states_per_s = batch / max(avg_ms / 1000.0, 1e-12)
    return avg_ms, states_per_s


def _print_table(rows: list[Row]) -> None:
    print(
        f"{'path':>16} | {'upd':>6} | {'cmp':>5} | {'batch':>8} | {'avg_ms':>10} | "
        f"{'states/s':>12} | {'agent/s':>12}"
    )
    print("-" * 86)
    for r in rows:
        print(
            f"{r.path:>16} | {r.upd:>6} | {str(r.compiled):>5} | {r.batch:8d} | {r.avg_ms:10.3f} | "
            f"{r.states_per_s:12.0f} | {r.agent_steps_per_s:12.0f}"
        )


def _print_speedup(rows: list[Row]) -> None:
    print("\ncompile speedup (eager -> compile):")
    print(f"{'path':>16} | {'upd':>6} | {'batch':>8} | {'ms_speedup':>11} | {'thr_gain':>10}")
    print("-" * 66)
    key_to_eager: dict[tuple[str, str, int], Row] = {}
    key_to_comp: dict[tuple[str, str, int], Row] = {}
    for r in rows:
        key = (r.path, r.upd, r.batch)
        if r.compiled:
            key_to_comp[key] = r
        else:
            key_to_eager[key] = r
    for key in sorted(set(key_to_eager.keys()) & set(key_to_comp.keys())):
        eager = key_to_eager[key]
        comp = key_to_comp[key]
        ms_speedup = eager.avg_ms / max(comp.avg_ms, 1e-12)
        thr_gain = comp.states_per_s / max(eager.states_per_s, 1e-12) - 1.0
        print(
            f"{key[0]:>16} | {key[1]:>6} | {key[2]:8d} | {ms_speedup:11.3f}x | {thr_gain*100:9.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--batches", default="4096,16384,65536,131072,262144")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--obstacles", type=int, default=16)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--output-dir", default="outputs/minco_projection_compile_compare")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    torch.zeros(1, device=device)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows: list[Row] = []
    paths = ("projection_only", "step_flat")
    upds = ("filter", "jacobi")
    cmps = (False, True)

    print(
        "config:",
        f"batches={batches}",
        f"warmup={args.warmup}",
        f"iters={args.iters}",
        f"agents={args.agents}",
        f"checkpoints={args.checkpoints}",
        f"obstacles={args.obstacles}",
        f"compile_mode={args.compile_mode}",
        f"compile_fullgraph={args.compile_fullgraph}",
        f"compile_dynamic={args.compile_dynamic}",
    )

    for path in paths:
        for upd in upds:
            for cmp_on in cmps:
                env = _make_env(
                    device=device,
                    agents=args.agents,
                    checkpoints=args.checkpoints,
                    obstacles=args.obstacles,
                )
                env.projection_ops.set_iter_update_mode(upd)
                env.projection_ops.set_jacobi_relax(0.5)
                for batch in batches:
                    try:
                        if path == "projection_only":
                            avg_ms, states_per_s = _bench_projection_only(
                                env,
                                batch=batch,
                                warmup=args.warmup,
                                iters=args.iters,
                                compiled=cmp_on,
                                compile_mode=args.compile_mode,
                                compile_fullgraph=args.compile_fullgraph,
                                compile_dynamic=args.compile_dynamic,
                            )
                        else:
                            avg_ms, states_per_s = _bench_step_flat(
                                env,
                                batch=batch,
                                warmup=args.warmup,
                                iters=args.iters,
                                compiled=cmp_on,
                                compile_mode=args.compile_mode,
                                compile_fullgraph=args.compile_fullgraph,
                                compile_dynamic=args.compile_dynamic,
                            )
                    except RuntimeError as exc:
                        if device.type == "cuda" and "out of memory" in str(exc).lower():
                            torch.cuda.empty_cache()
                            print(f"SKIP OOM: path={path} upd={upd} cmp={cmp_on} batch={batch}")
                            continue
                        raise
                    row = Row(
                        path=path,
                        upd=upd,
                        compiled=cmp_on,
                        batch=batch,
                        warmup=args.warmup,
                        iters=args.iters,
                        avg_ms=avg_ms,
                        states_per_s=states_per_s,
                        agent_steps_per_s=states_per_s * args.agents,
                    )
                    rows.append(row)
                    _print_table([row])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compile_compare.json"
    csv_path = out_dir / "compile_compare.csv"
    json_path.write_text(json.dumps([asdict(x) for x in rows], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    print("\nSummary")
    _print_table(rows)
    _print_speedup(rows)
    print(f"\njson: {json_path}")
    print(f"csv: {csv_path}")
    if rows:
        mean_ms = statistics.fmean(r.avg_ms for r in rows)
        print(f"rows={len(rows)} avg_of_avg_ms={mean_ms:.3f}")


if __name__ == "__main__":
    main()

