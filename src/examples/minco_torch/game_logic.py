"""环境游戏逻辑：done、step、flat step、tree gather step。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import NDIM


@dataclass(frozen=True)
class MincoPointEnvStep:
    """一次环境 step 的结构化输出。

    形状约定:
        coefficients: ``(..., agents, 6, 2)``
        active: ``(..., agents)``
        time: ``(...)``
        done: ``(...)``
        valid: ``(...)``
        collision_mask: ``(..., agents)``
        point_collision_mask: ``(..., agents)``
        obstacle_collision_mask: ``(..., agents)``
        out_of_bounds_mask: ``(..., agents)``
        projected_targets: ``(..., agents, 2)``
        obstacle_projected_targets: ``(..., agents, 2)``
        obstacle_projection_iterations: 标量
        obstacle_projection_residual_mask: ``(..., agents)``
    """

    coefficients: torch.Tensor
    active: torch.Tensor
    time: torch.Tensor
    done: torch.Tensor
    valid: torch.Tensor
    collision_mask: torch.Tensor
    point_collision_mask: torch.Tensor
    obstacle_collision_mask: torch.Tensor
    out_of_bounds_mask: torch.Tensor
    projected_targets: torch.Tensor
    obstacle_projected_targets: torch.Tensor
    obstacle_projection_iterations: torch.Tensor
    obstacle_projection_residual_mask: torch.Tensor


def done(env, active: torch.Tensor, time: torch.Tensor, valid: torch.Tensor, obstacle_collision: torch.Tensor | None = None) -> torch.Tensor:
    """根据时间、越界和 team 存活情况生成 done 信号。"""
    done_mask = torch.as_tensor(time) >= env.tf
    if env.done_on_out_of_bounds:
        done_mask = done_mask | ~torch.as_tensor(valid, device=done_mask.device).bool()
    if env.done_on_obstacle_collision and obstacle_collision is not None:
        done_mask = done_mask | torch.as_tensor(obstacle_collision, device=done_mask.device).bool().any(dim=-1)
    if env.done_on_team_eliminated:
        done_mask = done_mask | ~env.team_alive(active).all(dim=-1)
    return done_mask


def step(
    env,
    coefficients: torch.Tensor,
    active: torch.Tensor,
    time: torch.Tensor | float,
    target_pos: torch.Tensor,
    *,
    clamp_target: bool = True,
    target_is_delta: bool = False,
    return_projection_residual: bool = True,
) -> MincoPointEnvStep:
    """执行一次完整环境步进。"""
    coeff = torch.as_tensor(coefficients)
    active_tensor = torch.as_tensor(active, device=coeff.device, dtype=coeff.dtype)
    active_bool = active_tensor > 0.5
    time_tensor = torch.as_tensor(time, device=coeff.device, dtype=coeff.dtype)
    time_tensor = time_tensor.expand(coeff.shape[:-3])

    # 1) 动作解释为绝对目标点或 delta 目标点。
    target = env._target_from_action(
        coeff,
        target_pos,
        target_is_delta=target_is_delta,
    )
    # 2) 先做动态边界投影。
    projected_targets = (
        env.robot_transition.project_target(coeff, target)
        if clamp_target
        else target
    )
    # 3) 再做障碍物未来投影。
    if env.obstacle_target_projection:
        projected_targets, obstacle_projection_iterations, obstacle_projection_residual = (
            env.project_target_away_from_obstacles(
                coeff,
                projected_targets,
                active_bool,
                clamp_to_dynamic_bounds=clamp_target,
                return_residual=return_projection_residual,
            )
        )
    else:
        obstacle_projection_iterations = torch.zeros(
            (),
            dtype=torch.long,
            device=coeff.device,
        )
        obstacle_projection_residual = torch.zeros_like(active_bool)

    # 4) MINCO 系数推进。
    stepped = env.robot_transition.transition(coeff, projected_targets)
    inactive_frozen = env.zero_motion(coeff)
    next_coeff = torch.where(active_bool[..., :, None, None], stepped, inactive_frozen)

    # 5) 碰撞、越界、done。
    next_time = time_tensor + torch.as_tensor(env.dt, dtype=coeff.dtype, device=coeff.device)
    next_positions = env.positions(next_coeff)
    point_colliding = env.collision_mask(next_positions, active_bool)
    obstacle_colliding = env.obstacle_collision_mask(next_positions, active_bool)
    colliding = point_colliding | obstacle_colliding
    if env.deactivate_on_collision:
        next_active_bool = active_bool & ~point_colliding
    else:
        next_active_bool = active_bool
    if env.deactivate_on_obstacle_collision:
        next_active_bool = next_active_bool & ~obstacle_colliding

    frozen_after_collision = env.zero_motion(next_coeff)
    next_coeff = torch.where(
        next_active_bool[..., :, None, None],
        next_coeff,
        frozen_after_collision,
    )
    valid, out_of_bounds = env.bounds_validity(
        env.positions(next_coeff),
        next_active_bool,
    )
    done_mask = done(env, next_active_bool, next_time, valid, obstacle_colliding)

    return MincoPointEnvStep(
        coefficients=next_coeff,
        active=next_active_bool.to(dtype=coeff.dtype),
        time=next_time,
        done=done_mask,
        valid=valid,
        collision_mask=colliding,
        point_collision_mask=point_colliding,
        obstacle_collision_mask=obstacle_colliding,
        out_of_bounds_mask=out_of_bounds,
        projected_targets=projected_targets,
        obstacle_projected_targets=projected_targets,
        obstacle_projection_iterations=obstacle_projection_iterations,
        obstacle_projection_residual_mask=obstacle_projection_residual,
    )


def step_flat(
    env,
    flat_state: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    clamp_target: bool = True,
    target_is_delta: bool = False,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, MincoPointEnvStep]:
    """flat state 版本的环境 step。"""
    coeff, active, time = env.unpack_state(flat_state)
    step_out = step(
        env,
        coeff,
        active,
        time,
        target_pos,
        clamp_target=clamp_target,
        target_is_delta=target_is_delta,
        return_projection_residual=return_info,
    )
    next_flat = env.pack_state(step_out.coefficients, step_out.active, step_out.time)
    if return_info:
        return next_flat, step_out.done, step_out
    return next_flat, step_out.done


def gather_step_flat(
    env,
    flat_state_pool: torch.Tensor,
    tree_ids: torch.Tensor,
    node_ids: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    clamp_target: bool = True,
    target_is_delta: bool = False,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, MincoPointEnvStep]:
    """从树池中 gather 父节点，再执行一次 flat step。"""
    pool = torch.as_tensor(flat_state_pool)
    if pool.ndim != 3 or pool.shape[-1] != env.state_dim:
        raise ValueError(
            f"flat_state_pool must have shape (trees, nodes, {env.state_dim}), "
            f"got {tuple(pool.shape)}")
    trees = torch.as_tensor(tree_ids, device=pool.device, dtype=torch.long)
    nodes = torch.as_tensor(node_ids, device=pool.device, dtype=torch.long)
    parent_flat = pool[trees, nodes]
    return step_flat(
        env,
        parent_flat,
        target_pos,
        clamp_target=clamp_target,
        target_is_delta=target_is_delta,
        return_info=return_info,
    )
