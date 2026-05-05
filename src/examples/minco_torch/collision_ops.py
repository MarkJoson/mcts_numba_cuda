"""碰撞与边界判定算子（类封装）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .scene import MincoScene


@dataclass(frozen=True)
class CollisionStepResult:
    """一次碰撞/边界判定输出。"""

    point_collision_mask: torch.Tensor
    obstacle_collision_mask: torch.Tensor
    collision_mask: torch.Tensor
    valid: torch.Tensor
    out_of_bounds_mask: torch.Tensor


class MincoCollisionOps:
    """碰撞、边界、team 存活相关判定。

    设计目标：
    1. 只依赖 `MincoScene` 公共接口；
    2. 作为可复用的纯判定组件，不参与状态推进。
    """

    def __init__(self, scene: MincoScene) -> None:
        self.scene = scene
        self.num_agents = int(scene.num_agents)
        self.max_obstacle_vertices = int(scene.max_obstacle_vertices)
        self.collision_radius = float(scene.collision_radius)
        self.obstacle_collision_margin = float(scene.obstacle_collision_margin)
        self.bounds_check_active_only = bool(scene.bounds_check_active_only)

    @property
    def num_obstacles(self) -> int:
        return int(self.scene.num_obstacles)

    def obstacle_buffers(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.scene.obstacle_buffers(reference)

    def obstacle_signed_edge_distances_any_agents(self, positions: torch.Tensor) -> torch.Tensor:
        return self.scene.obstacle_signed_edge_distances(positions)

    def obstacle_clearance_any_agents_broadphase(self, positions: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(positions)
        if self.num_obstacles == 0:
            return torch.empty((*pos.shape[:-1], 0), dtype=pos.dtype, device=pos.device)
        normals, offsets, edge_mask, aabb_lower, aabb_upper = self.obstacle_buffers(pos)
        margin = torch.as_tensor(self.obstacle_collision_margin, dtype=pos.dtype, device=pos.device)
        near = torch.all(
            (pos.unsqueeze(-2) >= (aabb_lower - margin))
            & (pos.unsqueeze(-2) <= (aabb_upper + margin)),
            dim=-1,
        )
        signed = torch.sum(pos.unsqueeze(-2).unsqueeze(-2) * normals, dim=-1) - offsets
        mask_shape = (1,) * (signed.ndim - 2) + edge_mask.shape
        clearance = signed.masked_fill(~edge_mask.view(*mask_shape), -torch.inf).amax(dim=-1)
        outside_aabb_clearance = torch.ones_like(clearance)
        return torch.where(near, clearance, outside_aabb_clearance)

    def point_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return self.scene.point_collision_mask(positions, active)

    def obstacle_signed_edge_distances(self, positions: torch.Tensor) -> torch.Tensor:
        return self.scene.obstacle_signed_edge_distances(positions)

    def obstacle_clearance(self, positions: torch.Tensor) -> torch.Tensor:
        return self.scene.obstacle_clearance(positions)

    def obstacle_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return self.scene.obstacle_collision_mask(positions, active)

    def bounds_validity(self, positions: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene.bounds_validity(positions, active)

    def team_alive(self, active: torch.Tensor) -> torch.Tensor:
        return self.scene.team_alive(active)

    def collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """与旧接口兼容：返回机器人点模型碰撞，不含障碍物碰撞。"""
        return self.point_collision_mask(positions, active)

    def full_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        point = self.point_collision_mask(positions, active)
        obstacle = self.obstacle_collision_mask(positions, active)
        return point | obstacle

    def evaluate(self, positions: torch.Tensor, active: torch.Tensor) -> CollisionStepResult:
        point = self.point_collision_mask(positions, active)
        obstacle = self.obstacle_collision_mask(positions, active)
        valid, out_of_bounds = self.bounds_validity(positions, active)
        return CollisionStepResult(
            point_collision_mask=point,
            obstacle_collision_mask=obstacle,
            collision_mask=point | obstacle,
            valid=valid,
            out_of_bounds_mask=out_of_bounds,
        )
