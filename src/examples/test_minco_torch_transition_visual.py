"""Visual diagnostics for minco_torch_transition.py.

This script runs real ``MincoPointEnvTransition.step_flat`` rollouts and writes
GIF/PNG evidence under the current project directory.  The plots are intended
to make rollout state changes visible: multi-agent motion, point collisions,
convex obstacle collisions, inactive agents, time horizon done, and
out-of-bounds done.

Example:
    PYTHONPATH=src conda run -n py312_numba python \
        src/examples/test_minco_torch_transition_visual.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
import time
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import torch

from examples.minco_torch_transition import MincoPointEnvStep, MincoPointEnvTransition


TEAM_COLORS = ["#2364aa", "#2b9348", "#f77f00", "#7b2cbf"]
COLLISION_COLOR = "#d62828"
OBSTACLE_COLOR = "#8a2be2"
INACTIVE_COLOR = "#777777"
TARGET_COLOR = "#111111"
DONE_FACE = "#fff1f0"
NORMAL_FACE = "#ffffff"


@dataclass(frozen=True)
class SceneFrame:
    positions: np.ndarray
    active: np.ndarray
    time: float
    done: bool
    point_collision: np.ndarray
    obstacle_collision: np.ndarray
    out_of_bounds: np.ndarray
    targets: np.ndarray | None
    note: str


def _assert(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(name)


def _tensor(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


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


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_env(
    device: torch.device,
    *,
    obstacle_vertices: Sequence[Sequence[Sequence[float]]] | torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> MincoPointEnvTransition:
    env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        dtype=dtype,
        obstacle_vertices=obstacle_vertices,
        velocity_limit=kwargs.pop("velocity_limit", 12.0),
        acceleration_limit=kwargs.pop("acceleration_limit", 18.0),
        **kwargs,
    )
    return env.to(device)


def _unpack_frame(
    env: MincoPointEnvTransition,
    flat: torch.Tensor,
    *,
    targets: torch.Tensor | None,
    info: MincoPointEnvStep | None,
    done: torch.Tensor | bool,
    note: str,
) -> SceneFrame:
    coeff, active, time_tensor = env.unpack_state(flat)
    positions = env.positions(coeff)
    done_bool = bool(done.detach().cpu()) if torch.is_tensor(done) else bool(done)
    false_mask = torch.zeros((env.num_agents,), dtype=torch.bool, device=positions.device)
    return SceneFrame(
        positions=_tensor(positions),
        active=_tensor(active.bool()),
        time=float(time_tensor.detach().cpu()),
        done=done_bool,
        point_collision=_tensor(info.point_collision_mask.bool() if info is not None else false_mask),
        obstacle_collision=_tensor(info.obstacle_collision_mask.bool() if info is not None else false_mask),
        out_of_bounds=_tensor(info.out_of_bounds_mask.bool() if info is not None else false_mask),
        targets=None if targets is None else _tensor(targets),
        note=note,
    )


def _simulate(
    env: MincoPointEnvTransition,
    init_positions: Sequence[Sequence[float]],
    target_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    steps: int,
    active: Sequence[float] | None = None,
    time0: float = 0.0,
    clamp_target: bool = True,
    target_is_delta: bool = False,
    freeze_after_done: int = 4,
    note_fn: Callable[[int, torch.Tensor | bool, MincoPointEnvStep | None], str] | None = None,
) -> list[SceneFrame]:
    device = env.robot_transition.mat_f_stab.device
    dtype = env.robot_transition.mat_f_stab.dtype
    init = torch.tensor(init_positions, dtype=dtype, device=device)
    active_tensor = None if active is None else torch.tensor(active, dtype=dtype, device=device)
    flat = env.initial_flat_state(init, active=active_tensor, time=torch.tensor(time0, dtype=dtype, device=device))
    frames = [
        _unpack_frame(
            env,
            flat,
            targets=None,
            info=None,
            done=False,
            note="initial",
        )
    ]
    frozen_left = 0
    for step_idx in range(steps):
        coeff, active_now, _ = env.unpack_state(flat)
        pos_now = env.positions(coeff)
        target = target_fn(step_idx, pos_now, active_now)
        if frozen_left > 0:
            frames.append(
                _unpack_frame(
                    env,
                    flat,
                    targets=target,
                    info=None,
                    done=True,
                    note="terminal freeze",
                )
            )
            frozen_left -= 1
            continue

        next_flat, done, info = env.step_flat(
            flat,
            target,
            clamp_target=clamp_target,
            target_is_delta=target_is_delta,
            return_info=True,
        )
        note = note_fn(step_idx, done, info) if note_fn is not None else f"step {step_idx + 1}"
        frames.append(
            _unpack_frame(
                env,
                next_flat,
                targets=info.projected_targets if clamp_target else target,
                info=info,
                done=done,
                note=note,
            )
        )
        flat = next_flat
        if bool(done.detach().cpu()):
            frozen_left = int(freeze_after_done)
    return frames


def _draw_obstacles(ax: plt.Axes, obstacles: Sequence[Sequence[Sequence[float]]]) -> None:
    colors = ["#e9f5db", "#fde2e4", "#e2ece9", "#ede7f6", "#fff3b0"]
    for obs_idx, poly in enumerate(obstacles):
        points = np.asarray(poly, dtype=np.float32)
        patch = Polygon(
            points,
            closed=True,
            facecolor=colors[obs_idx % len(colors)],
            edgecolor="#333333",
            linewidth=1.7,
            alpha=0.85,
        )
        ax.add_patch(patch)
        center = points.mean(axis=0)
        ax.text(center[0], center[1], f"O{obs_idx}", ha="center", va="center", fontsize=10)


def _agent_color(frame: SceneFrame, idx: int) -> str:
    if frame.out_of_bounds[idx]:
        return "#111111"
    if frame.obstacle_collision[idx]:
        return OBSTACLE_COLOR
    if frame.point_collision[idx]:
        return COLLISION_COLOR
    if not frame.active[idx]:
        return INACTIVE_COLOR
    return TEAM_COLORS[0 if idx < 2 else 1]


def _agent_marker(frame: SceneFrame, idx: int) -> str:
    if frame.out_of_bounds[idx]:
        return "s"
    if frame.obstacle_collision[idx]:
        return "D"
    if frame.point_collision[idx]:
        return "X"
    if not frame.active[idx]:
        return "x"
    return "o"


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        ax.scatter([], [], s=90, color=TEAM_COLORS[0], marker="o", label="team 0 active"),
        ax.scatter([], [], s=90, color=TEAM_COLORS[1], marker="o", label="team 1 active"),
        ax.scatter([], [], s=95, color=COLLISION_COLOR, marker="X", label="point collision"),
        ax.scatter([], [], s=95, color=OBSTACLE_COLOR, marker="D", label="obstacle collision"),
        ax.scatter([], [], s=95, color=INACTIVE_COLOR, marker="x", label="inactive"),
        ax.scatter([], [], s=70, color="#111111", marker="s", label="out of bounds"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)


def _draw_scene(
    ax: plt.Axes,
    frames: Sequence[SceneFrame],
    frame_idx: int,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    obstacles: Sequence[Sequence[Sequence[float]]] = (),
    title: str,
    show_targets: bool = True,
    legend: bool = True,
) -> None:
    frame = frames[frame_idx]
    ax.clear()
    ax.set_facecolor(DONE_FACE if frame.done else NORMAL_FACE)
    lower_x, upper_x = bounds[0]
    lower_y, upper_y = bounds[1]
    pad_x = max((upper_x - lower_x) * 0.05, 0.25)
    pad_y = max((upper_y - lower_y) * 0.05, 0.25)
    ax.set_xlim(lower_x - pad_x, upper_x + pad_x)
    ax.set_ylim(lower_y - pad_y, upper_y + pad_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)
    ax.add_patch(
        Rectangle(
            (lower_x, lower_y),
            upper_x - lower_x,
            upper_y - lower_y,
            fill=False,
            edgecolor="#333333",
            linewidth=1.8,
        )
    )
    _draw_obstacles(ax, obstacles)

    history = np.stack([f.positions for f in frames[: frame_idx + 1]], axis=0)
    for agent_idx in range(frame.positions.shape[0]):
        color = TEAM_COLORS[0 if agent_idx < 2 else 1]
        ax.plot(
            history[:, agent_idx, 0],
            history[:, agent_idx, 1],
            color=color,
            linewidth=1.8,
            alpha=0.58,
        )
        if show_targets and frame.targets is not None:
            target = frame.targets[agent_idx]
            ax.plot(
                [frame.positions[agent_idx, 0], target[0]],
                [frame.positions[agent_idx, 1], target[1]],
                color=TARGET_COLOR,
                linewidth=0.8,
                alpha=0.35,
                linestyle="--",
            )
            ax.scatter(
                target[0],
                target[1],
                s=34,
                color=TARGET_COLOR,
                linewidth=1.1,
                marker="+",
            )

    for agent_idx, pos in enumerate(frame.positions):
        color = _agent_color(frame, agent_idx)
        marker = _agent_marker(frame, agent_idx)
        size = 142 if marker in ("X", "D") else 116
        ax.scatter(
            pos[0],
            pos[1],
            s=size,
            color=color,
            marker=marker,
            linewidth=2.0,
            zorder=5,
        )
        label = f"A{agent_idx}"
        if frame.point_collision[agent_idx]:
            label += " point"
        elif frame.obstacle_collision[agent_idx]:
            label += " obs"
        elif frame.out_of_bounds[agent_idx]:
            label += " oob"
        elif not frame.active[agent_idx]:
            label += " off"
        ax.text(pos[0] + 0.08, pos[1] + 0.08, label, fontsize=9, zorder=6)

    event_bits = []
    if frame.point_collision.any():
        event_bits.append("point_collision")
    if frame.obstacle_collision.any():
        event_bits.append("obstacle_collision")
    if frame.out_of_bounds.any():
        event_bits.append("out_of_bounds")
    if frame.done:
        event_bits.append("done")
    event_text = ", ".join(event_bits) if event_bits else "normal"
    ax.set_title(
        f"{title}\nframe={frame_idx:02d}, t={frame.time:.2f}, {event_text} | {frame.note}",
        fontsize=11,
    )
    if legend:
        _draw_legend(ax)


def _write_scene_animation(
    out_dir: Path,
    filename: str,
    frames: Sequence[SceneFrame],
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    obstacles: Sequence[Sequence[Sequence[float]]] = (),
    title: str,
    fps: int = 8,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    path = out_dir / filename
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=118):
        for frame_idx in range(len(frames)):
            _draw_scene(
                ax,
                frames,
                frame_idx,
                bounds=bounds,
                obstacles=obstacles,
                title=title,
            )
            writer.grab_frame()
    plt.close(fig)
    return path


def _write_keyframes(
    out_dir: Path,
    filename: str,
    frames: Sequence[SceneFrame],
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    obstacles: Sequence[Sequence[Sequence[float]]] = (),
    title: str,
    key_indices: Sequence[int] | None = None,
) -> Path:
    if key_indices is None:
        key_indices = [0, len(frames) // 2, len(frames) - 1]
    key_indices = [max(0, min(len(frames) - 1, idx)) for idx in key_indices]
    fig, axes = plt.subplots(1, len(key_indices), figsize=(5.4 * len(key_indices), 5.0))
    if len(key_indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, key_indices):
        _draw_scene(
            ax,
            frames,
            idx,
            bounds=bounds,
            obstacles=obstacles,
            title=title,
            legend=False,
        )
    return _save(fig, out_dir, filename)


def visual_multi_robot_rollout(out_dir: Path, device: torch.device) -> dict:
    bounds = ((-6.0, 6.0), (-4.5, 4.5))
    env = _make_env(
        device,
        collision_radius=0.0,
        position_bounds=bounds,
        done_on_team_eliminated=False,
        done_on_out_of_bounds=False,
        done_on_obstacle_collision=False,
        tf=12.0,
    )
    waypoints = torch.tensor(
        [
            [[4.5, -2.8], [3.4, 2.8], [-4.4, -2.6], [-3.2, 2.6]],
            [[4.2, 2.5], [0.0, 3.5], [-4.2, 2.5], [0.0, -3.4]],
            [[-4.2, 2.8], [-3.2, -2.7], [4.0, 2.7], [3.1, -2.5]],
        ],
        dtype=torch.float32,
        device=device,
    )

    def target_fn(step_idx: int, pos: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return waypoints[min(step_idx // 14, waypoints.shape[0] - 1)]

    frames = _simulate(
        env,
        [[-4.6, -3.0], [-4.3, 3.0], [4.6, -3.0], [4.3, 3.0]],
        target_fn,
        steps=42,
        freeze_after_done=0,
        note_fn=lambda step, done, info: "all agents move with independent targets",
    )
    history = np.stack([f.positions for f in frames], axis=0)
    displacement = np.linalg.norm(history[-1] - history[0], axis=-1)
    _assert("multi robot rollout moves all agents", bool(np.all(displacement > 0.45)))
    _assert("multi robot rollout has no collision", not any(f.point_collision.any() for f in frames))
    _assert("multi robot rollout stays finite", bool(np.isfinite(history).all()))

    gif = _write_scene_animation(
        out_dir,
        "01_multi_robot_rollout.gif",
        frames,
        bounds=bounds,
        title="Multi-Robot MINCO Rollout",
        fps=9,
    )
    png = _write_keyframes(
        out_dir,
        "01_multi_robot_rollout_keyframes.png",
        frames,
        bounds=bounds,
        title="Multi-Robot MINCO Rollout",
        key_indices=[0, 14, 28, len(frames) - 1],
    )
    legacy_gif = out_dir / "01_minco_rollout_animation.gif"
    legacy_png = out_dir / "01_minco_rollout_projection.png"
    copyfile(gif, legacy_gif)
    copyfile(png, legacy_png)
    return {
        "animation": str(gif),
        "keyframes": str(png),
        "legacy_animation_alias": str(legacy_gif),
        "legacy_keyframe_alias": str(legacy_png),
        "max_displacement": float(displacement.max()),
        "frames": len(frames),
    }


def visual_point_collision_dynamic(out_dir: Path, device: torch.device) -> dict:
    bounds = ((-5.0, 5.0), (-3.5, 3.5))
    env = _make_env(
        device,
        collision_radius=0.72,
        position_bounds=bounds,
        done_on_team_eliminated=False,
        done_on_out_of_bounds=False,
        tf=10.0,
    )
    targets = torch.tensor(
        [[0.0, 0.0], [-4.0, 2.6], [0.0, 0.0], [4.0, 2.6]],
        dtype=torch.float32,
        device=device,
    )

    def target_fn(step_idx: int, pos: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return targets

    frames = _simulate(
        env,
        [[-3.6, 0.0], [-3.7, 1.9], [3.6, 0.0], [3.7, 1.9]],
        target_fn,
        steps=48,
        freeze_after_done=0,
        note_fn=lambda step, done, info: "red X = nearest cross-team collision",
    )
    collision_indices = [idx for idx, frame in enumerate(frames) if frame.point_collision.any()]
    _assert("point collision appears in dynamic rollout", bool(collision_indices))
    first_collision = collision_indices[0]
    _assert(
        "point collision deactivates matched agents",
        bool((~frames[first_collision].active & frames[first_collision].point_collision).any()),
    )
    _assert("point collision case keeps non-colliding agents active", bool(frames[-1].active.sum() >= 2))

    gif = _write_scene_animation(
        out_dir,
        "02_point_collision_dynamic.gif",
        frames,
        bounds=bounds,
        title="Point Collision Dynamic Case",
        fps=9,
    )
    png = _write_keyframes(
        out_dir,
        "02_point_collision_dynamic_keyframes.png",
        frames,
        bounds=bounds,
        title="Point Collision Dynamic Case",
        key_indices=[0, max(0, first_collision - 2), first_collision, len(frames) - 1],
    )
    return {
        "animation": str(gif),
        "keyframes": str(png),
        "first_collision_frame": first_collision,
        "frames": len(frames),
    }


def visual_obstacle_collision_dynamic(out_dir: Path, device: torch.device) -> dict:
    bounds = ((-5.0, 5.0), (-4.0, 4.0))
    obstacles = [
        [[-0.85, -0.85], [0.85, -0.85], [0.85, 0.85], [-0.85, 0.85]],
        [[2.15, -1.15], [3.20, -0.45], [2.75, 0.95], [1.65, 0.70], [1.45, -0.55]],
        [[-3.05, 1.05], [-2.10, 1.75], [-2.75, 2.75], [-3.65, 2.15]],
    ]
    env = _make_env(
        device,
        collision_radius=0.0,
        obstacle_vertices=obstacles,
        obstacle_collision_margin=0.08,
        position_bounds=bounds,
        done_on_team_eliminated=False,
        done_on_obstacle_collision=True,
        done_on_out_of_bounds=False,
        tf=10.0,
    )
    targets = torch.tensor(
        [[2.9, 0.0], [-2.8, 2.1], [-2.8, 0.0], [2.6, 2.4]],
        dtype=torch.float32,
        device=device,
    )

    def target_fn(step_idx: int, pos: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return targets

    frames = _simulate(
        env,
        [[-3.8, 0.0], [-4.1, 2.7], [4.0, 0.0], [4.0, 2.6]],
        target_fn,
        steps=60,
        freeze_after_done=6,
        note_fn=lambda step, done, info: "purple diamond = convex obstacle hit",
    )
    obstacle_indices = [idx for idx, frame in enumerate(frames) if frame.obstacle_collision.any()]
    done_indices = [idx for idx, frame in enumerate(frames) if frame.done]
    _assert("obstacle collision appears in dynamic rollout", bool(obstacle_indices))
    _assert("obstacle collision triggers done", bool(done_indices and done_indices[0] <= obstacle_indices[0] + 1))

    first_collision = obstacle_indices[0]
    gif = _write_scene_animation(
        out_dir,
        "03_obstacle_collision_dynamic.gif",
        frames,
        bounds=bounds,
        obstacles=obstacles,
        title="Convex Obstacle Collision Dynamic Case",
        fps=9,
    )
    png = _write_keyframes(
        out_dir,
        "03_obstacle_collision_dynamic_keyframes.png",
        frames,
        bounds=bounds,
        obstacles=obstacles,
        title="Convex Obstacle Collision Dynamic Case",
        key_indices=[0, max(0, first_collision - 2), first_collision, len(frames) - 1],
    )
    return {
        "animation": str(gif),
        "keyframes": str(png),
        "first_obstacle_collision_frame": first_collision,
        "first_done_frame": done_indices[0],
        "obstacles": len(obstacles),
    }


def visual_done_time_dynamic(out_dir: Path, device: torch.device) -> dict:
    bounds = ((-4.0, 4.0), (-3.0, 3.0))
    env = _make_env(
        device,
        collision_radius=0.0,
        position_bounds=bounds,
        done_on_team_eliminated=False,
        done_on_out_of_bounds=False,
        done_on_obstacle_collision=False,
        tf=0.65,
    )
    targets = torch.tensor(
        [[3.0, 1.0], [2.6, -1.2], [-3.0, 1.2], [-2.7, -1.1]],
        dtype=torch.float32,
        device=device,
    )

    def target_fn(step_idx: int, pos: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return targets

    frames = _simulate(
        env,
        [[-2.8, -1.8], [-2.7, 1.8], [2.8, -1.8], [2.7, 1.8]],
        target_fn,
        steps=12,
        freeze_after_done=5,
        note_fn=lambda step, done, info: "background turns red at time horizon",
    )
    done_indices = [idx for idx, frame in enumerate(frames) if frame.done]
    _assert("time horizon done appears", bool(done_indices))
    _assert("time horizon is reached", bool(frames[done_indices[0]].time >= env.tf))

    gif = _write_scene_animation(
        out_dir,
        "04_done_time_horizon.gif",
        frames,
        bounds=bounds,
        title="Done Case: Time Horizon",
        fps=7,
    )
    png = _write_keyframes(
        out_dir,
        "04_done_time_horizon_keyframes.png",
        frames,
        bounds=bounds,
        title="Done Case: Time Horizon",
        key_indices=[0, max(0, done_indices[0] - 1), done_indices[0], len(frames) - 1],
    )
    return {
        "animation": str(gif),
        "keyframes": str(png),
        "first_done_frame": done_indices[0],
        "tf": env.tf,
    }


def visual_done_bounds_dynamic(out_dir: Path, device: torch.device) -> dict:
    bounds = ((-2.0, 2.0), (-2.0, 2.0))
    env = _make_env(
        device,
        collision_radius=0.0,
        position_bounds=bounds,
        done_on_team_eliminated=False,
        done_on_obstacle_collision=False,
        done_on_out_of_bounds=True,
        bounds_check_active_only=False,
        velocity_limit=18.0,
        acceleration_limit=25.0,
        tf=8.0,
    )
    targets = torch.tensor(
        [[4.2, 0.0], [-1.2, 1.6], [1.0, -1.7], [-1.6, -1.5]],
        dtype=torch.float32,
        device=device,
    )

    def target_fn(step_idx: int, pos: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return targets

    frames = _simulate(
        env,
        [[1.35, 0.0], [-1.5, 1.1], [1.3, -1.2], [-1.4, -1.1]],
        target_fn,
        steps=50,
        clamp_target=False,
        freeze_after_done=5,
        note_fn=lambda step, done, info: "black square = out-of-bounds agent",
    )
    bounds_indices = [idx for idx, frame in enumerate(frames) if frame.out_of_bounds.any()]
    done_indices = [idx for idx, frame in enumerate(frames) if frame.done]
    _assert("out-of-bounds appears in dynamic rollout", bool(bounds_indices))
    _assert("out-of-bounds triggers done", bool(done_indices and done_indices[0] <= bounds_indices[0]))

    first_bounds = bounds_indices[0]
    gif = _write_scene_animation(
        out_dir,
        "05_done_out_of_bounds.gif",
        frames,
        bounds=bounds,
        title="Done Case: Out Of Bounds",
        fps=8,
    )
    png = _write_keyframes(
        out_dir,
        "05_done_out_of_bounds_keyframes.png",
        frames,
        bounds=bounds,
        title="Done Case: Out Of Bounds",
        key_indices=[0, max(0, first_bounds - 2), first_bounds, len(frames) - 1],
    )
    return {
        "animation": str(gif),
        "keyframes": str(png),
        "first_out_of_bounds_frame": first_bounds,
        "first_done_frame": done_indices[0],
    }


def visual_collision_matrix(out_dir: Path, device: torch.device) -> dict:
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    pentagon = [[2.0, -0.3], [2.9, 0.0], [3.0, 0.9], [2.35, 1.4], [1.75, 0.7]]
    triangle = [[-2.8, 0.0], [-1.8, 0.2], [-2.3, 1.2]]
    obstacles = [square, pentagon, triangle]
    env = _make_env(
        device,
        collision_radius=0.0,
        obstacle_vertices=obstacles,
        obstacle_collision_margin=0.0,
        position_bounds=((-3.3, 3.3), (-0.7, 1.8)),
        done_on_team_eliminated=False,
    )
    points = torch.tensor(
        [[0.5, 0.5], [1.35, 0.5], [2.35, 0.65], [-2.3, 0.35]],
        dtype=torch.float32,
        device=device,
    )
    active = torch.ones((4,), dtype=torch.float32, device=device)
    clearance = env.obstacle_clearance(points.view(1, 4, 2))[0]
    mask = env.obstacle_collision_mask(points.view(1, 4, 2), active.view(1, 4))[0]
    _assert(
        "collision matrix classifies multiple convex polygons",
        int(mask.sum().detach().cpu()) == 3,
    )

    points_np = _tensor(points)
    clearance_np = _tensor(clearance)
    mask_np = _tensor(mask)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    _draw_obstacles(axes[0], obstacles)
    for idx, point in enumerate(points_np):
        axes[0].scatter(
            point[0],
            point[1],
            s=130,
            color=OBSTACLE_COLOR if mask_np[idx] else TEAM_COLORS[0],
            marker="D" if mask_np[idx] else "o",
        )
        axes[0].text(point[0] + 0.04, point[1] + 0.05, f"P{idx}", fontsize=9)
    axes[0].set_xlim(-3.3, 3.3)
    axes[0].set_ylim(-0.7, 1.8)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.22)
    axes[0].set_title("Points Against Multiple Convex Obstacles")

    im = axes[1].imshow(clearance_np, cmap="coolwarm", aspect="auto", vmin=-0.6, vmax=0.6)
    axes[1].set_xticks(np.arange(len(obstacles)))
    axes[1].set_xticklabels([f"O{i}" for i in range(len(obstacles))])
    axes[1].set_yticks(np.arange(points_np.shape[0]))
    axes[1].set_yticklabels([f"P{i}" for i in range(points_np.shape[0])])
    for y_idx in range(clearance_np.shape[0]):
        for x_idx in range(clearance_np.shape[1]):
            inside = clearance_np[y_idx, x_idx] <= 0.0
            axes[1].text(
                x_idx,
                y_idx,
                f"{clearance_np[y_idx, x_idx]:.2f}",
                ha="center",
                va="center",
                color="white" if inside else "black",
                fontsize=9,
            )
    axes[1].set_title("max signed edge distance <= 0 means inside")
    fig.colorbar(im, ax=axes[1], shrink=0.78, label="clearance")
    path = _save(fig, out_dir, "06_obstacle_distance_matrix.png")
    return {
        "figure": str(path),
        "obstacles": len(obstacles),
        "colliding_points": int(mask.sum().detach().cpu()),
    }


def visual_performance(out_dir: Path, device: torch.device) -> dict:
    obstacles = [
        [[-2.0, -2.0], [-1.0, -2.0], [-1.0, -1.0], [-2.0, -1.0]],
        [[1.0, -2.0], [2.0, -2.0], [2.0, -1.0], [1.0, -1.0]],
        [[-2.0, 1.0], [-1.0, 1.0], [-1.0, 2.0], [-2.0, 2.0]],
        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
    ]
    candidates = [torch.device("cpu")]
    if device.type == "cuda":
        candidates.append(device)

    labels = []
    throughputs = []
    elapsed_values = []
    batch_values = []
    for dev in candidates:
        env = _make_env(
            dev,
            collision_radius=0.0,
            obstacle_vertices=obstacles,
            obstacle_collision_margin=0.05,
            position_bounds=((-5.0, 5.0), (-5.0, 5.0)),
            done_on_team_eliminated=False,
            dtype=torch.float32,
        )
        batch = 4096 if dev.type == "cpu" else 65536
        base = torch.linspace(-4.0, 4.0, batch, dtype=torch.float32, device=dev)
        positions = torch.stack(
            [
                torch.stack((base, torch.zeros_like(base)), dim=-1),
                torch.stack((base, torch.ones_like(base) * 0.5), dim=-1),
                torch.stack((base, torch.ones_like(base) * -0.5), dim=-1),
                torch.stack((base, torch.ones_like(base)), dim=-1),
            ],
            dim=1,
        )
        flat = env.initial_flat_state(positions)
        target = positions + torch.tensor([0.01, 0.0], dtype=torch.float32, device=dev)
        env.step_flat(flat, target)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        repeats = 5 if dev.type == "cuda" else 3
        t0 = time.perf_counter()
        for _ in range(repeats):
            next_flat, done, info = env.step_flat(flat, target, return_info=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / repeats
        throughput = batch / max(elapsed_ms / 1000.0, 1e-12)
        _assert(
            f"{dev.type} performance output shapes",
            next_flat.shape == (batch, env.state_dim)
            and done.shape == (batch,)
            and info.obstacle_collision_mask.shape == (batch, 4),
        )
        labels.append(dev.type)
        throughputs.append(throughput)
        elapsed_values.append(elapsed_ms)
        batch_values.append(batch)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bars = ax.bar(labels, throughputs, color=["#9fc5e8", "#b6d7a8"][: len(labels)])
    ax.set_ylabel("states / second")
    ax.set_title("Batched Obstacle Step Throughput")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, batch, ms in zip(bars, batch_values, elapsed_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.0f}/s\n{ms:.3f} ms\nB={batch}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    path = _save(fig, out_dir, "07_performance_smoke.png")
    return {
        "figure": str(path),
        "labels": labels,
        "throughputs": throughputs,
        "elapsed_ms": elapsed_values,
    }


def write_report(out_dir: Path, results: dict) -> Path:
    report = out_dir / "visual_test_report.txt"
    lines = ["minco_torch_transition visual diagnostics", ""]
    for name, data in results.items():
        lines.append(f"[{name}]")
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/minco_torch_transition_visual",
        help="Directory where PNG/GIF diagnostics are written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for tensor tests. auto uses CUDA when it can allocate tensors.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)

    results = {
        "device": {"selected": str(device), "torch": torch.__version__},
        "multi_robot_rollout": visual_multi_robot_rollout(out_dir, device),
        "point_collision_dynamic": visual_point_collision_dynamic(out_dir, device),
        "obstacle_collision_dynamic": visual_obstacle_collision_dynamic(out_dir, device),
        "done_time_horizon": visual_done_time_dynamic(out_dir, device),
        "done_out_of_bounds": visual_done_bounds_dynamic(out_dir, device),
        "obstacle_distance_matrix": visual_collision_matrix(out_dir, device),
        "performance": visual_performance(out_dir, device),
    }
    report = write_report(out_dir, results)
    print(f"visual diagnostics written to: {out_dir}")
    print(f"report: {report}")
    for key, value in results.items():
        print(f"[{key}] {value}")


if __name__ == "__main__":
    main()
