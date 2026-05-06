"""Compare projection_only jacobi with/without Triton fused update."""

from __future__ import annotations

import argparse
import time

import torch

from examples.bench_minco_env_throughput import _complex_obstacles, _make_positions, _make_targets
from examples.minco_torch_transition import MincoPointEnvTransition


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_env(device: torch.device, triton_on: bool, checkpoints: int, triton_mode: str) -> MincoPointEnvTransition:
    env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        n_checkpoints=checkpoints,
        collision_radius=0.0,
        obstacle_vertices=_complex_obstacles(16),
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
    env.projection_ops.set_iter_update_mode("jacobi")
    env.projection_ops.set_jacobi_relax(0.5)
    env.projection_ops.set_triton_jacobi(triton_on)
    env.projection_ops.set_triton_jacobi_mode(triton_mode)
    return env


def _bench(env: MincoPointEnvTransition, batch: int, warmup: int, iters: int) -> float:
    pos = _make_positions(batch, env.num_agents, env.robot_transition.mat_f_stab.device)
    tgt = _make_targets(pos)
    coeff = env.initial_coefficients(pos)
    active = torch.ones((batch, env.num_agents), dtype=torch.bool, device=coeff.device)
    p = env.robot_transition.project_target(coeff, tgt)
    for _ in range(warmup):
        env.project_target_away_from_obstacles(
            coeff,
            p,
            active,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        env.project_target_away_from_obstacles(
            coeff,
            p,
            active,
            clamp_to_dynamic_bounds=True,
            return_residual=False,
        )
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--triton-mode", default="mega", choices=["basic", "mega"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    torch.set_grad_enabled(False)

    env_off = _make_env(dev, False, args.checkpoints, args.triton_mode)
    env_on = _make_env(dev, True, args.checkpoints, args.triton_mode)

    ms_off = _bench(env_off, args.batch, args.warmup, args.iters)
    ms_on = _bench(env_on, args.batch, args.warmup, args.iters)
    print(f"triton_off_avg_ms={ms_off:.3f}")
    print(f"triton_on_avg_ms={ms_on:.3f}")
    print(f"triton_mode={args.triton_mode}")
    print(f"speedup_off_over_on={ms_off / max(ms_on, 1e-12):.3f}x")


if __name__ == "__main__":
    main()
