"""场景对象：静态障碍、边界、团队与碰撞判定。

本模块只维护“静态世界”与“判定逻辑”，不做 MINCO 状态推进。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .constants import NDIM


@dataclass(frozen=True)
class SceneCollisionResult:
    """场景碰撞判定结果。

    形状约定:
        point_collision_mask: ``(..., agents)``
        obstacle_collision_mask: ``(..., agents)``
        collision_mask: ``(..., agents)``
        out_of_bounds_mask: ``(..., agents)``
        valid: ``(...)``
    """

    point_collision_mask: torch.Tensor
    obstacle_collision_mask: torch.Tensor
    collision_mask: torch.Tensor
    out_of_bounds_mask: torch.Tensor
    valid: torch.Tensor


class MincoScene(torch.nn.Module):
    """公共场景对象（静态缓存 + 判定接口）。

    关注点：
    1. 初始化阶段：预构建并注册所有静态张量（边界、team、障碍物边法向等）。
    2. rollout 阶段：只做批量张量运算，不做缓冲重建。

    关键成员 shape：
        team_ids: ``(agents,)``，每个 agent 的 team 编号。
        team_values: ``(num_teams,)``，去重后的 team 编号。
        collision_pair_i/j: ``(num_pairs,)``，点碰撞候选 pair 索引。
        position_lower/upper: ``(2,)``，场景边界。
        obstacle_vertices: ``(obstacles, max_vertices, 2)``。
        obstacle_edge_normals: ``(obstacles, max_vertices, 2)``。
        obstacle_edge_offsets: ``(obstacles, max_vertices)``。
        obstacle_edge_mask: ``(obstacles, max_vertices)``。
        obstacle_vertex_counts: ``(obstacles,)``。
        obstacle_aabb_lower/upper: ``(obstacles, 2)``。
    """

    def __init__(
        self,
        *,
        num_agents: int,
        collision_radius: float,
        position_bounds: Sequence[Sequence[float]] | torch.Tensor | None,
        team_ids: Sequence[int] | torch.Tensor | None,
        collide_same_team: bool,
        bounds_check_active_only: bool,
        obstacle_vertices: Sequence[Sequence[Sequence[float]]] | torch.Tensor | None,
        obstacle_vertex_counts: Sequence[int] | torch.Tensor | None,
        obstacle_collision_margin: float,
        obstacle_projection_check_active_only: bool,
        max_obstacle_vertices: int,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> None:
        """构造静态场景缓存。

        输入参数：
            num_agents: agent 数量。
            collision_radius: 点模型碰撞半径。
            position_bounds: 位置边界，shape ``(2, 2)`` 或 None（无限边界）。
            team_ids: 每个 agent 的 team id，shape ``(agents,)``；None 时自动二分 team。
            collide_same_team: 同队是否参与碰撞检测 pair。
            bounds_check_active_only: 越界是否只检查 active agent。
            obstacle_vertices: 障碍物顶点，shape ``(obstacles, vertices, 2)`` 或 list。
            obstacle_vertex_counts: 每个障碍有效顶点数，shape ``(obstacles,)``。
            obstacle_collision_margin: 点到障碍物碰撞余量。
            obstacle_projection_check_active_only: 障碍投影残差是否只看 active。
            max_obstacle_vertices: 单个障碍最大边数上限。
            dtype/device: 运行时缓存的 dtype/device。

        关键路径：
            1. 预计算 team/pair；
            2. 预处理边界；
            3. 预构建障碍物边法向、offset、mask、AABB；
            4. 全部 register_buffer，避免 rollout 中重复构造。
        """
        super().__init__()
        self.num_agents = int(num_agents)
        self.collision_radius = float(collision_radius)
        self.bounds_check_active_only = bool(bounds_check_active_only)
        self.obstacle_collision_margin = float(obstacle_collision_margin)
        self.obstacle_projection_check_active_only = bool(obstacle_projection_check_active_only)
        self.max_obstacle_vertices = int(max_obstacle_vertices)

        if team_ids is None:
            split = max(1, self.num_agents // 2)
            team_ids = [0 if i < split else 1 for i in range(self.num_agents)]
        team_tensor = torch.as_tensor(team_ids, dtype=torch.long, device=device)
        if team_tensor.shape != (self.num_agents,):
            raise ValueError(f"team_ids must have shape ({self.num_agents},), got {tuple(team_tensor.shape)}")
        self.register_buffer("team_ids", team_tensor)
        self.register_buffer("team_values", torch.unique(team_tensor, sorted=True))

        team_list = team_tensor.detach().cpu().tolist()
        pair_i = []
        pair_j = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (not collide_same_team) and team_list[i] == team_list[j]:
                    continue
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer("collision_pair_i", torch.tensor(pair_i, dtype=torch.long, device=device))
        self.register_buffer("collision_pair_j", torch.tensor(pair_j, dtype=torch.long, device=device))

        if position_bounds is None:
            lower = torch.full((NDIM,), -torch.inf, dtype=dtype, device=device)
            upper = torch.full((NDIM,), torch.inf, dtype=dtype, device=device)
        else:
            bounds = torch.as_tensor(position_bounds, dtype=dtype, device=device)
            if bounds.shape != (NDIM, 2):
                raise ValueError(f"position_bounds must have shape ({NDIM}, 2), got {tuple(bounds.shape)}")
            lower = bounds[:, 0]
            upper = bounds[:, 1]
        self.register_buffer("position_lower", lower)
        self.register_buffer("position_upper", upper)

        (
            obstacle_vertices_tensor,
            obstacle_edge_normals,
            obstacle_edge_offsets,
            obstacle_edge_mask,
            obstacle_vertex_count_tensor,
            obstacle_aabb_lower,
            obstacle_aabb_upper,
        ) = self._prepare_obstacle_buffers(
            obstacle_vertices,
            obstacle_vertex_counts,
            max_vertices=self.max_obstacle_vertices,
            dtype=dtype,
            device=device,
        )
        self.register_buffer("obstacle_vertices", obstacle_vertices_tensor)
        self.register_buffer("obstacle_edge_normals", obstacle_edge_normals)
        self.register_buffer("obstacle_edge_offsets", obstacle_edge_offsets)
        self.register_buffer("obstacle_edge_mask", obstacle_edge_mask)
        self.register_buffer("obstacle_vertex_counts", obstacle_vertex_count_tensor)
        self.register_buffer("obstacle_aabb_lower", obstacle_aabb_lower)
        self.register_buffer("obstacle_aabb_upper", obstacle_aabb_upper)
        
        self.obstacle_vertices: torch.Tensor
        self.obstacle_edge_normals: torch.Tensor
        self.obstacle_edge_offsets: torch.Tensor
        self.obstacle_edge_mask: torch.Tensor
        self.obstacle_vertex_counts: torch.Tensor
        self.obstacle_aabb_lower: torch.Tensor
        self.obstacle_aabb_upper: torch.Tensor

    @property
    def num_obstacles(self) -> int:
        """返回障碍物数量。"""
        return int(self.obstacle_vertex_counts.numel())

    def obstacle_buffers(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回投影热路径需要的障碍物缓存。

        输入:
            reference: 用于检查 dtype/device 的张量。
        输出:
            normals: ``(num_obstacles, max_vertices, 2)``
            offsets: ``(num_obstacles, max_vertices)``
            edge_mask: ``(num_obstacles, max_vertices)``
            aabb_lower: ``(num_obstacles, 2)``
            aabb_upper: ``(num_obstacles, 2)``
        """
        return self._buffers_for(reference)

    def _prepare_obstacle_buffers(
        self,
        obstacle_vertices: Sequence[Sequence[Sequence[float]]] | torch.Tensor | None,
        obstacle_vertex_counts: Sequence[int] | torch.Tensor | None,
        *,
        max_vertices: int,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """预构建凸多边形障碍物相关缓存（初始化阶段）。

        输入：
            obstacle_vertices:
                - Tensor: ``(obstacles, vertices, 2)``
                - 或 list[polygon]，每个 polygon 是 ``(vertices, 2)``
            obstacle_vertex_counts: ``(obstacles,)``，每个障碍有效顶点数。
            max_vertices: 顶点上限。
            dtype/device: 输出缓存张量 dtype/device。

        输出：
            vertices: ``(obstacles, max_vertices, 2)``
            outward_normals: ``(obstacles, max_vertices, 2)``
            edge_offsets: ``(obstacles, max_vertices)``
            edge_mask: ``(obstacles, max_vertices)``，True 表示有效边。
            counts: ``(obstacles,)``
            aabb_lower: ``(obstacles, 2)``
            aabb_upper: ``(obstacles, 2)``

        关键路径：
            1. 标准化输入到统一 padded tensor；
            2. 构造边向量与长度，检查零长度边；
            3. 用 signed area 自动识别 CW/CCW，统一外法向；
            4. 计算半平面 offset（n·x - b <= 0 在障碍内部）；
            5. 计算 AABB，供后续粗筛或可视化。
        """
        if obstacle_vertices is None:
            empty_vertices = torch.empty((0, max_vertices, NDIM), dtype=dtype, device=device)
            return (
                empty_vertices,
                empty_vertices.clone(),
                torch.empty((0, max_vertices), dtype=dtype, device=device),
                torch.empty((0, max_vertices), dtype=torch.bool, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0, NDIM), dtype=dtype, device=device),
                torch.empty((0, NDIM), dtype=dtype, device=device),
            )

        try:
            vertices = torch.as_tensor(obstacle_vertices, dtype=dtype, device=device)
        except (TypeError, ValueError):
            polygons = [torch.as_tensor(poly, dtype=dtype, device=device) for poly in obstacle_vertices]
            counts = torch.tensor([poly.shape[0] for poly in polygons], dtype=torch.long, device=device)
            max_count = int(counts.max().detach().cpu())
            if max_count > max_vertices:
                raise ValueError(f"obstacle polygon has {max_count} vertices; max is {max_vertices}")
            vertices = torch.zeros((len(polygons), max_count, NDIM), dtype=dtype, device=device)
            for i, poly in enumerate(polygons):
                if poly.ndim != 2 or poly.shape[-1] != NDIM:
                    raise ValueError(f"each obstacle polygon must have shape (vertices, {NDIM})")
                vertices[i, : poly.shape[0], :] = poly
        else:
            if vertices.ndim != 3 or vertices.shape[-1] != NDIM:
                raise ValueError(f"obstacle_vertices must have shape (obstacles, vertices, {NDIM}), got {tuple(vertices.shape)}")
            if vertices.shape[1] > max_vertices:
                raise ValueError(f"obstacle polygon has {vertices.shape[1]} vertices; max is {max_vertices}")
            if obstacle_vertex_counts is None:
                counts = torch.full((vertices.shape[0],), vertices.shape[1], dtype=torch.long, device=device)
            else:
                counts = torch.as_tensor(obstacle_vertex_counts, dtype=torch.long, device=device)

        if counts.ndim != 1 or counts.shape[0] != vertices.shape[0]:
            raise ValueError(f"obstacle_vertex_counts must have shape ({vertices.shape[0]},), got {tuple(counts.shape)}")
        if bool((counts < 3).any().detach().cpu()) or bool((counts > vertices.shape[1]).any().detach().cpu()):
            raise ValueError("each obstacle must have 3..max_vertices valid vertices")

        # edge_mask: (obstacles, max_vertices)，标记每个障碍哪些边有效。
        edge_ids = torch.arange(vertices.shape[1], dtype=torch.long, device=device)
        edge_mask = edge_ids.unsqueeze(0) < counts.unsqueeze(1)
        # next_vertices: (obstacles, max_vertices, 2)，通过循环索引构造每条边的终点。
        next_ids = (edge_ids.unsqueeze(0) + 1) % counts.unsqueeze(1)
        next_vertices = torch.gather(vertices, dim=1, index=next_ids.unsqueeze(-1).expand(-1, -1, NDIM))
        edges = next_vertices - vertices        # [..., max_verts, 2]
        edge_lengths = torch.linalg.norm(edges, dim=-1)
        if bool(((edge_lengths <= 1e-12) & edge_mask).any().detach().cpu()):
            raise ValueError("obstacle polygons cannot contain zero-length edges")

        cross = vertices[..., 0] * next_vertices[..., 1] - vertices[..., 1] * next_vertices[..., 0]
        signed_area2 = torch.sum(torch.where(edge_mask, cross, torch.zeros_like(cross)), dim=-1)
        if bool((torch.abs(signed_area2) <= 1e-12).any().detach().cpu()):
            raise ValueError("obstacle polygons must have non-zero signed area")

        # right_normals: 对每条边取“右法向”，再根据多边形朝向统一为外法向。
        right_normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1)
        orientation = torch.where(signed_area2 >= 0, torch.ones_like(signed_area2), -torch.ones_like(signed_area2))
        outward_normals = right_normals * orientation[:, None, None]
        outward_normals = outward_normals / torch.clamp(edge_lengths[..., None], min=1e-12)
        outward_normals = torch.where(edge_mask[..., None], outward_normals, torch.zeros_like(outward_normals))
        # edge_offsets: b，满足每条边约束 n·x - b。
        edge_offsets = torch.sum(vertices * outward_normals, dim=-1)
        edge_offsets = torch.where(edge_mask, edge_offsets, torch.zeros_like(edge_offsets))
        inf = torch.full_like(vertices, torch.inf)
        valid_vertices_for_min = torch.where(edge_mask[..., None], vertices, inf)
        valid_vertices_for_max = torch.where(edge_mask[..., None], vertices, -inf)
        aabb_lower = valid_vertices_for_min.amin(dim=1)
        aabb_upper = valid_vertices_for_max.amax(dim=1)
        return vertices, outward_normals, edge_offsets, edge_mask, counts, aabb_lower, aabb_upper

    def _buffers_for(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """内部版障碍物缓存检查；公共调用请优先使用 `obstacle_buffers`。

        输入：
            reference: 任意参与计算的参考张量，仅用于检查 dtype/device。
        输出：
            normals: ``(obstacles, max_vertices, 2)``
            offsets: ``(obstacles, max_vertices)``
            edge_mask: ``(obstacles, max_vertices)``
            aabb_lower: ``(obstacles, 2)``
            aabb_upper: ``(obstacles, 2)``
        """
        expected_device = self.obstacle_edge_normals.device
        expected_dtype = self.obstacle_edge_normals.dtype
        if reference.device != expected_device or reference.dtype != expected_dtype:
            raise ValueError(
                "inputs must use the same dtype/device as scene buffers; "
                f"got dtype={reference.dtype}, device={reference.device}, "
                f"expected dtype={expected_dtype}, device={expected_device}."
            )
        return (
            self.obstacle_edge_normals,
            self.obstacle_edge_offsets,
            self.obstacle_edge_mask,
            self.obstacle_aabb_lower,
            self.obstacle_aabb_upper,
        )

    def point_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """计算点模型 one-to-one 最近配对碰撞。

        输入：
            positions: ``(..., agents, 2)``
            active: ``(..., agents)``
        输出：
            matched: ``(..., agents)``，True 表示该 agent 在本步发生点碰撞。

        关键路径：
            1. 使用预计算 pair 索引做 pairwise 距离；
            2. 按距离排序后贪心匹配，保证 one-to-one；
            3. 仅 active pair 参与碰撞判定。
        """
        pos = torch.as_tensor(positions)
        if pos.device != self.position_lower.device or pos.dtype != self.position_lower.dtype:
            raise ValueError(
                "positions must use the same dtype/device as scene buffers; "
                f"got dtype={pos.dtype}, device={pos.device}, "
                f"expected dtype={self.position_lower.dtype}, device={self.position_lower.device}."
            )
        active_bool = torch.as_tensor(active, device=pos.device).bool()
        if self.collision_radius <= 0.0:
            return torch.zeros_like(active_bool)

        pair_i_t = self.collision_pair_i
        pair_j_t = self.collision_pair_j
        if pair_i_t.numel() == 0:
            return torch.zeros_like(active_bool)

        # pos_i/pos_j: (..., pairs, 2)
        pos_i = pos.index_select(-2, pair_i_t)
        pos_j = pos.index_select(-2, pair_j_t)
        dist_sq = torch.sum((pos_i - pos_j) * (pos_i - pos_j), dim=-1)
        active_pair = active_bool.index_select(-1, pair_i_t) & active_bool.index_select(-1, pair_j_t)
        candidate = active_pair & (dist_sq < self.collision_radius * self.collision_radius)
        sorted_dist, sorted_idx = torch.sort(dist_sq, dim=-1)
        sorted_candidate = torch.gather(candidate, -1, sorted_idx) & torch.isfinite(sorted_dist)
        sorted_i = pair_i_t[sorted_idx]
        sorted_j = pair_j_t[sorted_idx]

        # matched: (..., agents)，逐 pair 贪心更新。
        matched = torch.zeros_like(active_bool)
        for k in range(int(pair_i_t.numel())):
            i_idx = sorted_i[..., k]
            j_idx = sorted_j[..., k]
            i_taken = torch.gather(matched, -1, i_idx.unsqueeze(-1)).squeeze(-1)
            j_taken = torch.gather(matched, -1, j_idx.unsqueeze(-1)).squeeze(-1)
            selected = sorted_candidate[..., k] & ~i_taken & ~j_taken
            matched = matched | (
                F.one_hot(i_idx, self.num_agents).bool()
                | F.one_hot(j_idx, self.num_agents).bool()
            ) & selected.unsqueeze(-1)
        return matched

    def obstacle_signed_edge_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """批量计算点到所有障碍边的有符号距离。

        输入：
            positions: ``(..., 2)`` 或 ``(..., agents, 2)``
        输出：
            signed: ``(..., obstacles, max_vertices)``
                有效边为 ``n·x-b``，无效 padded 边填 ``-inf``。

        关键路径：
            1. `einsum` 直接做批量 half-space 距离；
            2. 用 edge_mask 将无效边抹成 -inf，便于后续 `amax`。
        """
        pos = torch.as_tensor(positions)
        if self.num_obstacles == 0:
            return torch.empty(
                (*pos.shape[:-1], 0, self.max_obstacle_vertices),
                dtype=pos.dtype,
                device=pos.device,
            )
        normals, offsets, edge_mask, _, _ = self._buffers_for(pos)
        # signed: (..., obstacles, max_vertices)
        signed = torch.einsum("...d,oed->...oe", pos, normals) - offsets
        mask_shape = (1,) * (signed.ndim - 2) + edge_mask.shape
        return signed.masked_fill(~edge_mask.view(*mask_shape), -torch.inf)

    def obstacle_clearance(self, positions: torch.Tensor) -> torch.Tensor:
        """计算点到每个障碍物的 clearance（最大边约束值）。

        输入：
            positions: ``(..., 2)`` 或 ``(..., agents, 2)``
        输出：
            clearance: ``(..., obstacles)``

        说明：
            对凸多边形，`max_e(n_e·x-b_e)` <= 0 表示在障碍物内（或边上）。
        """
        signed = self.obstacle_signed_edge_distances(positions)
        if signed.shape[-2] == 0:
            return torch.empty((*signed.shape[:-2], 0), dtype=signed.dtype, device=signed.device)
        return signed.amax(dim=-1)

    def obstacle_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """根据障碍物 clearance 判定碰撞。

        输入：
            positions: ``(..., agents, 2)``
            active: ``(..., agents)``
        输出：
            mask: ``(..., agents)``

        判定规则：
            若任一障碍 `clearance <= obstacle_collision_margin`，则判定碰撞。
        """
        pos = torch.as_tensor(positions)
        active_bool = torch.as_tensor(active, device=pos.device).bool()
        if self.num_obstacles == 0:
            return torch.zeros_like(active_bool)
        collides = self.obstacle_clearance(pos) <= self.obstacle_collision_margin
        return active_bool & collides.any(dim=-1)

    def bounds_validity(self, positions: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """检查边界合法性。

        输入：
            positions: ``(..., agents, 2)``
            active: ``(..., agents)``
        输出：
            valid: ``(...)``，当前批次是否整体合法（无越界）。
            out_of_bounds: ``(..., agents)``，每个 agent 是否越界。

        关键路径：
            1. 逐维比较 lower/upper；
            2. 按配置决定是否只对 active agent 生效。
        """
        pos = torch.as_tensor(positions)
        if pos.device != self.position_lower.device or pos.dtype != self.position_lower.dtype:
            raise ValueError(
                "positions must use the same dtype/device as scene buffers; "
                f"got dtype={pos.dtype}, device={pos.device}, "
                f"expected dtype={self.position_lower.dtype}, device={self.position_lower.device}."
            )
        lower = self.position_lower
        upper = self.position_upper
        in_bounds = torch.all((pos >= lower) & (pos <= upper), dim=-1)
        active_bool = torch.as_tensor(active, device=pos.device).bool()
        if self.bounds_check_active_only:
            out_of_bounds = active_bool & ~in_bounds
        else:
            out_of_bounds = ~in_bounds
        return ~out_of_bounds.any(dim=-1), out_of_bounds

    def team_alive(self, active: torch.Tensor) -> torch.Tensor:
        """统计每个 team 是否仍有存活 agent。

        输入：
            active: ``(..., agents)``
        输出：
            alive: ``(..., num_teams)``
        """
        active_bool = torch.as_tensor(active).bool()
        if active_bool.device != self.team_ids.device:
            raise ValueError(
                "active masks must use the same device as scene buffers; "
                f"got device={active_bool.device}, expected device={self.team_ids.device}."
            )
        teams = self.team_ids
        values = self.team_values
        alive = [(active_bool & (teams == team_value)).any(dim=-1) for team_value in values]
        return torch.stack(alive, dim=-1)

    def collision_summary(self, positions: torch.Tensor, active: torch.Tensor) -> SceneCollisionResult:
        """一次性返回点碰撞、障碍碰撞、越界与有效性汇总。

        输入：
            positions: ``(..., agents, 2)``
            active: ``(..., agents)``
        输出：
            SceneCollisionResult，各字段 shape 见 dataclass 文档。
        """
        point = self.point_collision_mask(positions, active)
        obstacle = self.obstacle_collision_mask(positions, active)
        valid, out_of_bounds = self.bounds_validity(positions, active)
        return SceneCollisionResult(
            point_collision_mask=point,
            obstacle_collision_mask=obstacle,
            collision_mask=point | obstacle,
            out_of_bounds_mask=out_of_bounds,
            valid=valid,
        )
