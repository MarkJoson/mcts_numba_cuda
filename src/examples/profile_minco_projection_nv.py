"""Nsight profiling driver for MINCO obstacle projection path.

Run this script under ``nsys``/``ncu`` to capture projection-heavy kernels.
It supports two modes:

* ``step_flat``: full env step with obstacle projection enabled.
* ``projection_only``: isolate target projection path.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from examples.bench_minco_env_throughput import _complex_obstacles, _make_positions, _make_targets
from examples.minco_torch_transition import MincoPointEnvTransition


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_env(*, device: torch.device, agents: int, checkpoints: int, obstacles: int) -> MincoPointEnvTransition:
    return MincoPointEnvTransition(
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


def _run_step_flat(
    env: MincoPointEnvTransition,
    *,
    batch: int,
    warmup: int,
    iters: int,
) -> None:
    pos = _make_positions(batch, env.num_agents, env.robot_transition.mat_f_stab.device)
    tgt = _make_targets(pos)
    flat = env.initial_flat_state(pos)

    for _ in range(warmup):
        flat, _ = env.step_flat(flat, tgt, return_info=False)
    _sync()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile_step_flat")
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.cuda.nvtx.range_push("step_flat_iter")
        flat, _ = env.step_flat(flat, tgt, return_info=False)
        torch.cuda.nvtx.range_pop()
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0
    torch.cuda.nvtx.range_pop()
    cudart.cudaProfilerStop()
    print(f"[step_flat] batch={batch} warmup={warmup} iters={iters} total_ms={ms:.3f} avg_ms={ms/iters:.3f}")


def _run_projection_only(
    env: MincoPointEnvTransition,
    *,
    batch: int,
    warmup: int,
    iters: int,
    inner_nvtx: bool,
    dump_select_stats: str | None,
) -> None:
    pos = _make_positions(batch, env.num_agents, env.robot_transition.mat_f_stab.device)
    tgt = _make_targets(pos)
    coeff = env.initial_coefficients(pos)
    active = torch.ones((batch, env.num_agents), dtype=torch.bool, device=coeff.device)
    env.projection_ops.set_prof_nvtx(inner_nvtx)
    # 关闭 sel_count，避免统计开销干扰性能对比。
    env.projection_ops.set_prof_select_stats(dump_select_stats is not None, reset=True)

    for _ in range(warmup):
        p = env.robot_transition.project_target(coeff, tgt)
        env.project_target_away_from_obstacles(
            coeff,
            p,
            active,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
    _sync()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile_projection_only")
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.cuda.nvtx.range_push("projection_iter")
        torch.cuda.nvtx.range_push("proj.outer.project_target_initial")
        p = env.robot_transition.project_target(coeff, tgt)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("proj.outer.project_away")
        env.project_target_away_from_obstacles(
            coeff,
            p,
            active,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0
    torch.cuda.nvtx.range_pop()
    cudart.cudaProfilerStop()
    print(f"[projection_only] batch={batch} warmup={warmup} iters={iters} total_ms={ms:.3f} avg_ms={ms/iters:.3f}")
    if dump_select_stats is not None:
        stats = env.projection_ops.get_prof_select_stats()
        with open(dump_select_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[select_stats] json: {dump_select_stats}")
        print("name,valid,total,valid_ratio,invalid_ratio")
        for name in sorted(stats):
            item = stats[name]
            print(
                f"{name},{int(item['valid'])},{int(item['total'])},"
                f"{item['valid_ratio']:.6f},{item['invalid_ratio']:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="step_flat", choices=["step_flat", "projection_only"])
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--obstacles", type=int, default=16)
    parser.add_argument("--inner-nvtx", action="store_true")
    parser.add_argument("--dump-select-stats", default=None)
    parser.add_argument("--iter-update-mode", default="filter", choices=["filter", "jacobi"])
    parser.add_argument("--jacobi-relax", type=float, default=0.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler script")
    device = torch.device(args.device)
    torch.zeros(1, device=device)
    env = _make_env(
        device=device,
        agents=args.agents,
        checkpoints=args.checkpoints,
        obstacles=args.obstacles,
    )
    env.projection_ops.set_iter_update_mode(args.iter_update_mode)
    env.projection_ops.set_jacobi_relax(args.jacobi_relax)
    print(
        "profile config:",
        f"mode={args.mode}",
        f"batch={args.batch}",
        f"iters={args.iters}",
        f"agents={args.agents}",
        f"checkpoints={args.checkpoints}",
        f"obstacles={args.obstacles}",
        f"iter_update_mode={args.iter_update_mode}",
        f"jacobi_relax={args.jacobi_relax}",
        "compile=always_on",
    )
    if args.mode == "step_flat":
        _run_step_flat(env, batch=args.batch, warmup=args.warmup, iters=args.iters)
    else:
        _run_projection_only(
            env,
            batch=args.batch,
            warmup=args.warmup,
            iters=args.iters,
            inner_nvtx=args.inner_nvtx,
            dump_select_stats=args.dump_select_stats,
        )


if __name__ == "__main__":
    main()
