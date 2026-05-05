"""动作投影与未来障碍物检查（类封装）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .collision_ops import MincoCollisionOps
from .constants import NDIM
from .scene import MincoScene
from .transition import MincoTorchTransition


class MincoProjectionOps:
    """目标动作约束投影算子。

    只封装目标点相关投影，不参与环境状态管理。
    """

    def __init__(
        self,
        *,
        scene: MincoScene,
        robot_transition: MincoTorchTransition,
        collision_ops: MincoCollisionOps,
        obstacle_collision_margin: float,
        obstacle_projection_iters: int,
        obstacle_projection_extra_margin: float,
        obstacle_projection_topk: int,
        obstacle_projection_fixes_per_iter: int,
        obstacle_projection_check_active_only: bool,
    ) -> None:
        self.scene = scene
        self.robot_transition = robot_transition
        self.collision_ops = collision_ops
        self.num_agents = int(scene.num_agents)
        self.obstacle_collision_margin = float(obstacle_collision_margin)
        self.obstacle_projection_iters = int(obstacle_projection_iters)
        self.obstacle_projection_extra_margin = float(obstacle_projection_extra_margin)
        self.obstacle_projection_topk = int(obstacle_projection_topk)
        self.obstacle_projection_fixes_per_iter = int(obstacle_projection_fixes_per_iter)
        self.obstacle_projection_check_active_only = bool(obstacle_projection_check_active_only)

    @property
    def num_obstacles(self) -> int:
        return int(self.scene.num_obstacles)

    def future_positions_from_target(self, coefficients: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        coeff = torch.as_tensor(coefficients)
        self.robot_transition.runtime_buffers(coeff)
        target = torch.as_tensor(target_pos, dtype=coeff.dtype, device=coeff.device)
        const = self.robot_transition.checkpoint_position_const
        gain = self.robot_transition.checkpoint_position_gain
        base = torch.einsum("pc,...acd->...apd", const, coeff)
        return base + gain.view(*((1,) * (base.ndim - 2)), -1, 1) * target.unsqueeze(-2)

    def future_obstacle_clearance_from_target(self, coefficients: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        future_pos = self.future_positions_from_target(coefficients, target_pos)
        if self.num_obstacles == 0:
            return torch.empty(
                (*future_pos.shape[:-2], future_pos.shape[-2], 0),
                dtype=future_pos.dtype,
                device=future_pos.device,
            )
        signed = self.collision_ops.obstacle_signed_edge_distances_any_agents(future_pos)
        return signed.amax(dim=-1)

    def project_target_away_from_obstacles(
        self,
        coefficients: torch.Tensor,
        target_pos: torch.Tensor,
        active: torch.Tensor | None = None,
        *,
        clamp_to_dynamic_bounds: bool = True,
        iterations: int | None = None,
        return_residual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coeff = torch.as_tensor(coefficients)
        target = torch.as_tensor(target_pos, dtype=coeff.dtype, device=coeff.device)
        if target.shape[-2:] != (self.num_agents, NDIM):
            raise ValueError(
                f"target_pos must end with ({self.num_agents}, {NDIM}), got {tuple(target.shape)}")
        if self.num_obstacles == 0:
            residual = torch.zeros(target.shape[:-1], dtype=torch.bool, device=target.device)
            return target, torch.zeros((), dtype=torch.long, device=target.device), residual

        iters = self.obstacle_projection_iters if iterations is None else int(iterations)
        if iters <= 0:
            if return_residual:
                clearance = self.future_obstacle_clearance_from_target(coeff, target)
                residual = (clearance <= self.obstacle_collision_margin).any(dim=(-1, -2))
            else:
                residual = torch.zeros(target.shape[:-1], dtype=torch.bool, device=target.device)
            if active is not None and self.obstacle_projection_check_active_only:
                residual = residual & torch.as_tensor(active, device=target.device).bool()
            return target, torch.zeros((), dtype=torch.long, device=target.device), residual

        active_mask = None
        if active is not None and self.obstacle_projection_check_active_only:
            active_mask = torch.as_tensor(active, device=target.device).bool()

        normals, offsets, edge_mask, _, _ = self.collision_ops.obstacle_buffers(target)
        self.robot_transition.runtime_buffers(coeff)
        pos_const = self.robot_transition.checkpoint_position_const
        pos_gain = self.robot_transition.checkpoint_position_gain

        projected = target
        margin = torch.as_tensor(
            self.obstacle_collision_margin + self.obstacle_projection_extra_margin,
            dtype=target.dtype,
            device=target.device,
        )

        future_pos = self.future_positions_from_target(coeff, projected)
        signed = self.collision_ops.obstacle_signed_edge_distances_any_agents(future_pos)
        clearance, edge_idx = signed.max(dim=-1)
        candidate_violation = torch.clamp(margin - clearance, min=0.0)
        if active_mask is not None:
            candidate_violation = torch.where(
                active_mask.unsqueeze(-1).unsqueeze(-1),
                candidate_violation,
                torch.zeros_like(candidate_violation),
            )

        candidate_count = clearance.shape[-2] * clearance.shape[-1]
        topk = min(self.obstacle_projection_topk, candidate_count)
        fixes_per_iter = min(self.obstacle_projection_fixes_per_iter, topk)
        flat_violation = candidate_violation.contiguous().view(*target.shape[:-1], candidate_count)
        top_values, top_flat_idx = torch.topk(flat_violation, k=topk, dim=-1)
        checkpoint_idx = torch.div(top_flat_idx, self.num_obstacles, rounding_mode="floor")
        obstacle_idx = top_flat_idx - checkpoint_idx * self.num_obstacles
        flat_edge_idx = edge_idx.contiguous().view(*target.shape[:-1], candidate_count)
        selected_edge_idx = torch.gather(flat_edge_idx, -1, top_flat_idx)

        p_const = torch.einsum("pc,...acd->...apd", pos_const, coeff)
        selected_p_const = torch.gather(
            p_const,
            -2,
            checkpoint_idx.unsqueeze(-1).expand(*checkpoint_idx.shape, NDIM),
        )
        selected_gain = pos_gain[checkpoint_idx]
        selected_normals = normals[obstacle_idx, selected_edge_idx]
        selected_offsets = offsets[obstacle_idx, selected_edge_idx]
        selected_edge_valid = edge_mask[obstacle_idx, selected_edge_idx]
        selected_valid = selected_edge_valid & (top_values > 0)

        a_vec = selected_gain.unsqueeze(-1) * selected_normals
        rhs = selected_offsets + margin - torch.sum(selected_p_const * selected_normals, dim=-1)
        denom = torch.sum(a_vec * a_vec, dim=-1).clamp_min(torch.finfo(target.dtype).eps)

        for _ in range(iters):
            violation = rhs - torch.sum(a_vec * projected.unsqueeze(-2), dim=-1)
            positive_violation = torch.where(
                selected_valid,
                torch.clamp(violation, min=0.0),
                torch.zeros_like(violation),
            )
            fix_values, fix_idx = torch.topk(positive_violation, k=fixes_per_iter, dim=-1)
            fix_mask = F.one_hot(fix_idx, topk).bool()
            fix_mask = (fix_mask & (fix_values > 0).unsqueeze(-1)).any(dim=-2)
            weighted_violation = torch.where(
                fix_mask,
                positive_violation,
                torch.zeros_like(positive_violation),
            )
            correction = torch.sum((weighted_violation / denom).unsqueeze(-1) * a_vec, dim=-2)
            projected = projected + correction
            if clamp_to_dynamic_bounds:
                projected = self.robot_transition.project_target(coeff, projected)

        if return_residual:
            clearance = self.future_obstacle_clearance_from_target(coeff, projected)
            residual = (clearance <= self.obstacle_collision_margin).any(dim=(-1, -2))
        else:
            residual = torch.zeros(target.shape[:-1], dtype=torch.bool, device=target.device)
        if active_mask is not None:
            residual = residual & active_mask
        return projected, torch.tensor(iters, dtype=torch.long, device=target.device), residual
