"""Single-robot MINCO target-projection visual diagnostic.

This script writes a GIF and keyframe PNG for a continuous single-robot rollout
through a scene with more than ten convex obstacles.  Each animation frame is
driven by the real ``MincoPointEnvTransition.step_flat`` path:

    target -> velocity/acceleration projection -> obstacle target projection
    -> MINCO transition

The robot is colored orange/red when the projected MINCO checkpoints still
violate obstacle constraints or when the current point is inside an obstacle.

Example:
    PYTHONPATH=src conda run -n py312_numba python \
        src/examples/test_minco_torch_complex_projection_visual.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import torch

from examples.minco_torch_transition import MincoPointEnvTransition


ROBOT_NORMAL = "#2364aa"
ROBOT_PROJECTED_VIOLATION = "#f77f00"
ROBOT_ACTUAL_COLLISION = "#d62828"
PROJECTED_PATH = "#00a896"
RAW_PATH = "#8d99ae"
HISTORY_PATH = "#1d3557"
DESIRED_TARGET = "#ffb703"
DYNAMIC_TARGET = "#6c757d"
PROJECTED_TARGET = "#008000"
OBSTACLE_EDGE = "#2b2d42"
OBSTACLE_FILL = "#edf2f4"
VIOLATION_POINT = "#d62828"
SAFE_POINT = "#2a9d8f"


@dataclass(frozen=True)
class ProjectionFrame:
    step: int
    time: float
    waypoint_index: int
    position: np.ndarray
    global_waypoint: np.ndarray
    local_target: np.ndarray
    dynamic_target: np.ndarray
    projected_target: np.ndarray
    raw_path: np.ndarray
    projected_path: np.ndarray
    raw_checkpoint_collision: np.ndarray
    projected_checkpoint_collision: np.ndarray
    target_shift: float
    raw_min_clearance: float
    projected_min_clearance: float
    projection_residual: bool
    actual_collision: bool
    out_of_bounds: bool
    done: bool
    speed: float


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


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _rotated_box(
    cx: float,
    cy: float,
    width: float,
    height: float,
    degrees: float,
) -> list[list[float]]:
    rad = math.radians(degrees)
    rot = np.array(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]],
        dtype=np.float32,
    )
    half = np.array(
        [
            [-width * 0.5, -height * 0.5],
            [width * 0.5, -height * 0.5],
            [width * 0.5, height * 0.5],
            [-width * 0.5, height * 0.5],
        ],
        dtype=np.float32,
    )
    points = half @ rot.T + np.array([cx, cy], dtype=np.float32)
    return points.tolist()


def _diamond(cx: float, cy: float, radius_x: float, radius_y: float) -> list[list[float]]:
    return [
        [cx, cy - radius_y],
        [cx + radius_x, cy],
        [cx, cy + radius_y],
        [cx - radius_x, cy],
    ]


def _complex_obstacles() -> list[list[list[float]]]:
    return [
        _rotated_box(-4.45, -2.35, 1.15, 1.60, -16.0),
        _rotated_box(-2.85, -3.05, 1.05, 1.70, 9.0),
        _diamond(-1.35, -2.05, 0.75, 0.95),
        _rotated_box(0.35, -2.95, 1.25, 1.15, 22.0),
        _rotated_box(2.15, -2.10, 1.00, 1.80, -20.0),
        _diamond(3.95, -1.25, 0.78, 1.00),
        _rotated_box(-4.85, 0.05, 1.05, 1.80, 12.0),
        _rotated_box(-2.85, 0.35, 1.30, 1.25, -26.0),
        _diamond(-0.65, 0.15, 0.88, 1.10),
        _rotated_box(1.35, 0.45, 1.25, 1.45, 18.0),
        _rotated_box(3.25, 0.75, 1.05, 1.50, -13.0),
        _diamond(4.85, 1.85, 0.72, 1.00),
        _rotated_box(-3.65, 2.55, 1.35, 1.05, 17.0),
        _rotated_box(-1.15, 2.62, 1.40, 1.08, -10.0),
        _diamond(1.65, 2.70, 0.95, 0.78),
    ]


def _make_env(device: torch.device, obstacles: Sequence[Sequence[Sequence[float]]]) -> MincoPointEnvTransition:
    env = MincoPointEnvTransition(
        num_agents=1,
        piece_t=0.12,
        dt=0.12,
        n_checkpoints=28,
        collision_radius=0.0,
        obstacle_vertices=obstacles,
        obstacle_collision_margin=0.10,
        obstacle_target_projection=True,
        obstacle_projection_iters=3,
        obstacle_projection_extra_margin=0.04,
        obstacle_projection_topk=8,
        obstacle_projection_fixes_per_iter=2,
        obstacle_projection_check_active_only=True,
        velocity_limit=8.5,
        acceleration_limit=14.0,
        position_bounds=((-6.0, 6.0), (-4.2, 4.2)),
        done_on_team_eliminated=False,
        done_on_obstacle_collision=False,
        deactivate_on_obstacle_collision=False,
        done_on_out_of_bounds=True,
        tf=60.0,
        dtype=torch.float32,
        device=device,
    )
    return env


def _path_and_collision(
    env: MincoPointEnvTransition,
    coeff: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float]:
    path = env._future_positions_from_target(coeff, target)[0]
    clearance = env.future_obstacle_clearance_from_target(coeff, target)[0]
    checkpoint_collision = (clearance <= env.obstacle_collision_margin).any(dim=-1)
    return _np(path), _np(checkpoint_collision), float(clearance.min().detach().cpu())


def _simulate_rollout(
    env: MincoPointEnvTransition,
    *,
    steps: int,
) -> list[ProjectionFrame]:
    device = env.robot_transition.mat_f_stab.device
    dtype = env.robot_transition.mat_f_stab.dtype
    waypoints = torch.tensor(
        [
            [-5.70, -1.25],
            [-5.70, 1.30],
            [-4.95, 3.55],
            [-2.55, 3.78],
            [-0.05, 3.78],
            [2.60, 3.75],
            [5.35, 3.22],
            [5.45, 1.10],
            [5.35, -0.95],
            [4.25, -2.75],
            [2.55, -3.72],
            [0.20, -3.88],
        ],
        dtype=dtype,
        device=device,
    )
    flat = env.initial_flat_state(
        torch.tensor([[-5.55, -3.65]], dtype=dtype, device=device),
        time=torch.tensor(0.0, dtype=dtype, device=device),
    )
    active_target = 0
    frames: list[ProjectionFrame] = []

    for step in range(steps):
        coeff, active, time_tensor = env.unpack_state(flat)
        pos = env.positions(coeff)[0]
        velocity = coeff[0, 1]
        while active_target < waypoints.shape[0] - 1:
            if torch.linalg.norm(pos - waypoints[active_target]) >= 0.48:
                break
            active_target += 1

        waypoint = waypoints[active_target]
        to_waypoint = waypoint - pos
        distance = torch.linalg.norm(to_waypoint)
        direction = to_waypoint / torch.clamp(distance, min=1e-6)
        lookahead = torch.minimum(distance, torch.as_tensor(0.88, dtype=dtype, device=device))
        local_target = pos + direction * lookahead - 0.08 * velocity
        lower = env.position_lower + torch.as_tensor([0.18, 0.18], dtype=dtype, device=device)
        upper = env.position_upper - torch.as_tensor([0.18, 0.18], dtype=dtype, device=device)
        local_target = torch.minimum(torch.maximum(local_target, lower), upper).view(1, 2)

        dynamic = env.robot_transition.project_target(coeff, local_target)
        projected, _, residual = env.project_target_away_from_obstacles(
            coeff,
            dynamic,
            active.bool(),
            clamp_to_dynamic_bounds=True,
        )
        raw_path, raw_cp_collision, raw_min_clearance = _path_and_collision(env, coeff, dynamic)
        projected_path, projected_cp_collision, projected_min_clearance = _path_and_collision(env, coeff, projected)
        current_collision = env.obstacle_collision_mask(
            pos.view(1, 1, 2),
            active.bool().view(1, 1),
        )[0, 0]
        _, out_of_bounds = env.bounds_validity(pos.view(1, 1, 2), active.bool().view(1, 1))
        current_out_of_bounds = out_of_bounds[0, 0]

        next_flat, done, info = env.step_flat(
            flat,
            local_target,
            clamp_target=True,
            return_info=True,
        )
        frames.append(
            ProjectionFrame(
                step=step,
                time=float(time_tensor.detach().cpu()),
                waypoint_index=active_target,
                position=_np(pos),
                global_waypoint=_np(waypoint),
                local_target=_np(local_target[0]),
                dynamic_target=_np(dynamic[0]),
                projected_target=_np(info.projected_targets[0]),
                raw_path=raw_path,
                projected_path=projected_path,
                raw_checkpoint_collision=raw_cp_collision,
                projected_checkpoint_collision=projected_cp_collision,
                target_shift=float(torch.linalg.norm(projected[0] - dynamic[0]).detach().cpu()),
                raw_min_clearance=raw_min_clearance,
                projected_min_clearance=projected_min_clearance,
                projection_residual=bool(residual[0].detach().cpu()),
                actual_collision=bool(current_collision.detach().cpu()),
                out_of_bounds=bool(current_out_of_bounds.detach().cpu()),
                done=bool(done.detach().cpu()),
                speed=float(torch.linalg.norm(velocity).detach().cpu()),
            )
        )
        flat = next_flat
        if bool(done.detach().cpu()):
            break
    return frames


def _draw_obstacles(ax: plt.Axes, obstacles: Sequence[Sequence[Sequence[float]]]) -> None:
    for idx, poly in enumerate(obstacles):
        points = np.asarray(poly, dtype=np.float32)
        patch = Polygon(
            points,
            closed=True,
            facecolor=OBSTACLE_FILL,
            edgecolor=OBSTACLE_EDGE,
            linewidth=1.2,
            alpha=0.92,
        )
        ax.add_patch(patch)
        center = points.mean(axis=0)
        ax.text(center[0], center[1], str(idx + 1), ha="center", va="center", fontsize=7, color="#343a40")


def _robot_color(frame: ProjectionFrame) -> str:
    if frame.actual_collision or frame.out_of_bounds:
        return ROBOT_ACTUAL_COLLISION
    if frame.projection_residual:
        return ROBOT_PROJECTED_VIOLATION
    return ROBOT_NORMAL


def _draw_frame(
    ax: plt.Axes,
    frames: Sequence[ProjectionFrame],
    frame_idx: int,
    obstacles: Sequence[Sequence[Sequence[float]]],
) -> None:
    frame = frames[frame_idx]
    ax.clear()
    ax.set_xlim(-6.25, 6.25)
    ax.set_ylim(-4.45, 4.45)
    ax.set_aspect("equal")
    ax.set_facecolor(
        "#fff5f5"
        if frame.actual_collision or frame.out_of_bounds
        else "#fffaf0"
        if frame.projection_residual
        else "#ffffff"
    )
    ax.grid(True, alpha=0.22)
    ax.add_patch(Rectangle((-6.0, -4.2), 12.0, 8.4, fill=False, edgecolor="#111111", linewidth=1.5))
    _draw_obstacles(ax, obstacles)

    history = np.stack([f.position for f in frames[: frame_idx + 1]], axis=0)
    ax.plot(history[:, 0], history[:, 1], color=HISTORY_PATH, linewidth=2.2, alpha=0.80, label="executed")

    ax.plot(
        frame.raw_path[:, 0],
        frame.raw_path[:, 1],
        color=RAW_PATH,
        linewidth=1.4,
        linestyle="--",
        alpha=0.75,
        label="MINCO before obstacle projection",
    )
    ax.plot(
        frame.projected_path[:, 0],
        frame.projected_path[:, 1],
        color=PROJECTED_PATH,
        linewidth=2.4,
        alpha=0.92,
        label="projected MINCO checkpoints",
    )

    safe_points = frame.projected_path[~frame.projected_checkpoint_collision]
    violation_points = frame.projected_path[frame.projected_checkpoint_collision]
    if len(safe_points) > 0:
        ax.scatter(safe_points[:, 0], safe_points[:, 1], s=13, color=SAFE_POINT, alpha=0.55, zorder=4)
    if len(violation_points) > 0:
        ax.scatter(
            violation_points[:, 0],
            violation_points[:, 1],
            s=42,
            color=VIOLATION_POINT,
            marker="x",
            linewidth=1.6,
            zorder=6,
            label="projected checkpoint violation",
        )

    raw_violation_points = frame.raw_path[frame.raw_checkpoint_collision]
    if len(raw_violation_points) > 0:
        ax.scatter(
            raw_violation_points[:, 0],
            raw_violation_points[:, 1],
            s=28,
            facecolors="none",
            edgecolors="#fb8500",
            linewidth=1.2,
            alpha=0.75,
            zorder=5,
            label="raw checkpoint violation",
        )

    ax.scatter(
        frame.global_waypoint[0],
        frame.global_waypoint[1],
        marker="*",
        s=190,
        color=DESIRED_TARGET,
        edgecolor="#111111",
        linewidth=0.7,
        label="active waypoint",
        zorder=7,
    )
    ax.scatter(
        frame.local_target[0],
        frame.local_target[1],
        marker="^",
        s=82,
        color="#fb8500",
        edgecolor="#111111",
        linewidth=0.6,
        label="local tracking target",
        zorder=7,
    )
    ax.scatter(
        frame.dynamic_target[0],
        frame.dynamic_target[1],
        marker="x",
        s=90,
        color=DYNAMIC_TARGET,
        linewidth=2.0,
        label="after dyn limits",
        zorder=7,
    )
    ax.scatter(
        frame.projected_target[0],
        frame.projected_target[1],
        marker="+",
        s=140,
        color=PROJECTED_TARGET,
        linewidth=2.3,
        label="after obstacle projection",
        zorder=8,
    )

    marker = "X" if frame.actual_collision or frame.out_of_bounds else "D" if frame.projection_residual else "o"
    ax.scatter(
        frame.position[0],
        frame.position[1],
        s=190,
        color=_robot_color(frame),
        marker=marker,
        edgecolor="#111111",
        linewidth=1.2,
        zorder=9,
        label="robot",
    )

    status = (
        "OUT OF BOUNDS"
        if frame.out_of_bounds
        else "ACTUAL COLLISION"
        if frame.actual_collision
        else "PROJECTED PATH VIOLATION"
        if frame.projection_residual
        else "safe"
    )
    status_color = (
        ROBOT_ACTUAL_COLLISION
        if frame.actual_collision or frame.out_of_bounds
        else ROBOT_PROJECTED_VIOLATION
        if frame.projection_residual
        else ROBOT_NORMAL
    )
    ax.text(
        -6.05,
        4.05,
        (
            f"step {frame.step:03d}  t={frame.time:.2f}  waypoint={frame.waypoint_index + 1}/12\n"
            f"status: {status}\n"
            f"min clearance raw={frame.raw_min_clearance:.3f}, projected={frame.projected_min_clearance:.3f}, "
            f"target shift={frame.target_shift:.3f}, speed={frame.speed:.2f}"
        ),
        ha="left",
        va="top",
        fontsize=9,
        color="#111111",
        bbox=dict(facecolor="#ffffff", edgecolor=status_color, linewidth=1.4, alpha=0.92),
    )
    ax.set_title("Single Robot MINCO Rollout with Target Projection in 15 Convex Obstacles", fontsize=12)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.92)


def _write_animation(
    out_dir: Path,
    frames: Sequence[ProjectionFrame],
    obstacles: Sequence[Sequence[Sequence[float]]],
    *,
    fps: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 6.9))
    path = out_dir / "single_robot_complex_minco_projection.gif"
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=116):
        for idx in range(len(frames)):
            _draw_frame(ax, frames, idx, obstacles)
            writer.grab_frame()
    plt.close(fig)
    return path


def _write_keyframes(
    out_dir: Path,
    frames: Sequence[ProjectionFrame],
    obstacles: Sequence[Sequence[Sequence[float]]],
) -> Path:
    residual_indices = [idx for idx, frame in enumerate(frames) if frame.projection_residual]
    collision_indices = [idx for idx, frame in enumerate(frames) if frame.actual_collision]
    key_indices = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    if residual_indices:
        key_indices[1] = residual_indices[0]
    if collision_indices:
        key_indices[2] = collision_indices[0]
    key_indices = [max(0, min(len(frames) - 1, idx)) for idx in key_indices]

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.2))
    for ax, idx in zip(axes.ravel(), key_indices):
        _draw_frame(ax, frames, idx, obstacles)
        ax.legend().remove()
    path = out_dir / "single_robot_complex_minco_projection_keyframes.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    *,
    device: torch.device,
    elapsed_s: float,
    frames: Sequence[ProjectionFrame],
    obstacles: Sequence[Sequence[Sequence[float]]],
    gif_path: Path,
    keyframe_path: Path,
) -> Path:
    positions = np.stack([frame.position for frame in frames], axis=0)
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    residual_count = sum(frame.projection_residual for frame in frames)
    actual_collision_count = sum(frame.actual_collision for frame in frames)
    out_of_bounds_count = sum(frame.out_of_bounds for frame in frames)
    projected_frames = sum(frame.target_shift > 1e-4 for frame in frames)
    min_projected_clearance = min(frame.projected_min_clearance for frame in frames)
    min_raw_clearance = min(frame.raw_min_clearance for frame in frames)

    report = out_dir / "single_robot_complex_minco_projection_report.txt"
    report.write_text(
        "\n".join(
            [
                "single robot complex MINCO target-projection visual",
                f"device: {device}",
                f"torch: {torch.__version__}",
                f"obstacles: {len(obstacles)}",
                f"frames: {len(frames)}",
                f"elapsed_s: {elapsed_s:.3f}",
                f"displacement: {displacement:.3f}",
                f"projected_target_frames: {projected_frames}",
                f"projection_residual_frames: {residual_count}",
                f"actual_collision_frames: {actual_collision_count}",
                f"out_of_bounds_frames: {out_of_bounds_count}",
                f"min_raw_clearance: {min_raw_clearance:.4f}",
                f"min_projected_clearance: {min_projected_clearance:.4f}",
                f"gif: {gif_path}",
                f"keyframes: {keyframe_path}",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/minco_torch_transition_complex_projection",
        help="Directory where the GIF/PNG/report are written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for tensor rollout. auto uses CUDA when available.",
    )
    parser.add_argument("--steps", type=int, default=132)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)
    obstacles = _complex_obstacles()
    env = _make_env(device, obstacles)

    start = time.perf_counter()
    frames = _simulate_rollout(env, steps=args.steps)
    gif_path = _write_animation(out_dir, frames, obstacles, fps=args.fps)
    keyframe_path = _write_keyframes(out_dir, frames, obstacles)
    elapsed_s = time.perf_counter() - start
    report_path = _write_report(
        out_dir,
        device=device,
        elapsed_s=elapsed_s,
        frames=frames,
        obstacles=obstacles,
        gif_path=gif_path,
        keyframe_path=keyframe_path,
    )

    positions = np.stack([frame.position for frame in frames], axis=0)
    if len(obstacles) < 10:
        raise AssertionError("scene must contain at least 10 obstacles")
    if np.linalg.norm(positions[-1] - positions[0]) <= 2.5:
        raise AssertionError("robot did not move far enough for a continuous rollout visual")
    if not any(frame.target_shift > 1e-4 for frame in frames):
        raise AssertionError("obstacle target projection was never active")

    print(f"complex projection visual written to: {out_dir}")
    print(f"gif: {gif_path}")
    print(f"keyframes: {keyframe_path}")
    print(f"report: {report_path}")
    print(f"obstacles={len(obstacles)} frames={len(frames)} device={device}")
    print(f"projection_residual_frames={sum(frame.projection_residual for frame in frames)}")
    print(f"actual_collision_frames={sum(frame.actual_collision for frame in frames)}")
    print(f"out_of_bounds_frames={sum(frame.out_of_bounds for frame in frames)}")
    print(f"target_projection_frames={sum(frame.target_shift > 1e-4 for frame in frames)}")


if __name__ == "__main__":
    main()
