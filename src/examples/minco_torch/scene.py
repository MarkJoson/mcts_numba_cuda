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
        team_ids: ``(agents,)``。
        team_values: ``(num_teams,)``。
        collision_pair_i/j: ``(num_pairs,)``。
        position_lower/upper: ``(2,)``。
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

        参数：
            num_agents: agent 数量。
            collision_radius: 点模型碰撞半径。
            position_bounds: 位置边界，shape ``(2, 2)``；None 表示无界。
            team_ids: 每个 agent 的 team id，shape ``(agents,)``；None 时自动二分。
            collide_same_team: 是否允许同队 pair 参与点碰撞检测。
            bounds_check_active_only: 越界是否只检查 active agent。
            obstacle_vertices: 障碍物顶点，shape ``(obstacles, vertices, 2)`` 或嵌套序列。
            obstacle_vertex_counts: 每个障碍有效顶点数，shape ``(obstacles,)``。
            obstacle_collision_margin: 点到障碍物碰撞余量。
            obstacle_projection_check_active_only: 障碍投影是否只检查 active agent。
            max_obstacle_vertices: 单个障碍允许的最大边数。
            dtype/device: 运行时缓存 dtype / device。

        关键路径：
            1. 预计算 team 与候选碰撞 pair；
            2. 预处理边界上下界；
            3. 预构建障碍物边法向、offset、mask 与 AABB；
            4. 全部 register_buffer，避免 rollout 时重复构造。

        输出：
            无。静态缓存全部写入模块 buffer。
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
        tids = torch.as_tensor(team_ids, dtype=torch.long, device=device)
        if tids.shape != (self.num_agents,):
            raise ValueError(f"team_ids must have shape ({self.num_agents},), got {tuple(tids.shape)}")
        self.register_buffer("team_ids", tids)
        self.register_buffer("team_values", torch.unique(tids, sorted=True))

        tl = tids.detach().cpu().tolist()
        pi = []
        pj = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (not collide_same_team) and tl[i] == tl[j]:
                    continue
                pi.append(i)
                pj.append(j)
        self.register_buffer("collision_pair_i", torch.tensor(pi, dtype=torch.long, device=device))
        self.register_buffer("collision_pair_j", torch.tensor(pj, dtype=torch.long, device=device))

        if position_bounds is None:
            lo = torch.full((NDIM,), -torch.inf, dtype=dtype, device=device)
            hi = torch.full((NDIM,), torch.inf, dtype=dtype, device=device)
        else:
            bnd = torch.as_tensor(position_bounds, dtype=dtype, device=device)
            if bnd.shape != (NDIM, 2):
                raise ValueError(f"position_bounds must have shape ({NDIM}, 2), got {tuple(bnd.shape)}")
            lo = bnd[:, 0]
            hi = bnd[:, 1]
        self.register_buffer("position_lower", lo)
        self.register_buffer("position_upper", hi)

        (
            v,
            n,
            b,
            em,
            cnt,
            lo_aabb,
            hi_aabb,
        ) = self._prepare_obstacle_buffers(
            obstacle_vertices,
            obstacle_vertex_counts,
            max_vertices=self.max_obstacle_vertices,
            dtype=dtype,
            device=device,
        )
        self.register_buffer("obstacle_vertices", v)
        self.register_buffer("obstacle_edge_normals", n)
        self.register_buffer("obstacle_edge_offsets", b)
        self.register_buffer("obstacle_edge_mask", em)
        self.register_buffer("obstacle_vertex_counts", cnt)
        self.register_buffer("obstacle_aabb_lower", lo_aabb)
        self.register_buffer("obstacle_aabb_upper", hi_aabb)

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

        参数：
            reference: 仅用于校验 dtype / device 的参考张量，通常是当前 rollout 的位置张量。

        返回：
            normals: ``(obstacles, max_vertices, 2)``
            offsets: ``(obstacles, max_vertices)``
            edge_mask: ``(obstacles, max_vertices)``
            aabb_lower: ``(obstacles, 2)``
            aabb_upper: ``(obstacles, 2)``

        关键路径：
            1. 只做 dtype / device 校验；
            2. 直接返回已注册 buffer，不触发额外拷贝或重建。
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

        参数：
            obstacle_vertices:
                - Tensor: ``(obstacles, vertices, 2)``
                - 或 list[polygon]，每个 polygon 的 shape 为 ``(vertices, 2)``
            obstacle_vertex_counts: ``(obstacles,)``，每个障碍有效顶点数。
            max_vertices: 单障碍最大顶点数。
            dtype/device: 输出缓存 dtype / device。

        输出 shape：
            v: ``(obstacles, max_vertices, 2)``
            n: ``(obstacles, max_vertices, 2)``
            b: ``(obstacles, max_vertices)``
            em: ``(obstacles, max_vertices)``
            cnt: ``(obstacles,)``
            lo_aabb/hi_aabb: ``(obstacles, 2)``

        关键路径：
            1. 标准化输入到统一 padded tensor；
            2. 计算每条边的终点、边向量与长度；
            3. 校验零长度边与零面积多边形；
            4. 根据 signed area 自动统一外法向方向；
            5. 计算 half-space offset 与 AABB，供后续快速碰撞/投影使用。
        """
        if obstacle_vertices is None:
            ev = torch.empty((0, max_vertices, NDIM), dtype=dtype, device=device)
            return (
                ev,
                ev.clone(),
                torch.empty((0, max_vertices), dtype=dtype, device=device),
                torch.empty((0, max_vertices), dtype=torch.bool, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0, NDIM), dtype=dtype, device=device),
                torch.empty((0, NDIM), dtype=dtype, device=device),
            )

        try:
            v = torch.as_tensor(obstacle_vertices, dtype=dtype, device=device)
        except (TypeError, ValueError):
            polys = [torch.as_tensor(poly, dtype=dtype, device=device) for poly in obstacle_vertices]
            cnt = torch.tensor([poly.shape[0] for poly in polys], dtype=torch.long, device=device)
            max_cnt = int(cnt.max().detach().cpu())
            if max_cnt > max_vertices:
                raise ValueError(f"obstacle polygon has {max_cnt} vertices; max is {max_vertices}")
            v = torch.zeros((len(polys), max_cnt, NDIM), dtype=dtype, device=device)
            for i, poly in enumerate(polys):
                if poly.ndim != 2 or poly.shape[-1] != NDIM:
                    raise ValueError(f"each obstacle polygon must have shape (vertices, {NDIM})")
                v[i, : poly.shape[0], :] = poly
        else:
            if v.ndim != 3 or v.shape[-1] != NDIM:
                raise ValueError(f"obstacle_vertices must have shape (obstacles, vertices, {NDIM}), got {tuple(v.shape)}")
            if v.shape[1] > max_vertices:
                raise ValueError(f"obstacle polygon has {v.shape[1]} vertices; max is {max_vertices}")
            if obstacle_vertex_counts is None:
                cnt = torch.full((v.shape[0],), v.shape[1], dtype=torch.long, device=device)
            else:
                cnt = torch.as_tensor(obstacle_vertex_counts, dtype=torch.long, device=device)

        if cnt.ndim != 1 or cnt.shape[0] != v.shape[0]:
            raise ValueError(f"obstacle_vertex_counts must have shape ({v.shape[0]},), got {tuple(cnt.shape)}")
        if bool((cnt < 3).any().detach().cpu()) or bool((cnt > v.shape[1]).any().detach().cpu()):
            raise ValueError("each obstacle must have 3..max_vertices valid vertices")

        eid = torch.arange(v.shape[1], dtype=torch.long, device=device)
        em = eid.unsqueeze(0) < cnt.unsqueeze(1)
        nid = (eid.unsqueeze(0) + 1) % cnt.unsqueeze(1)
        nv = torch.gather(v, dim=1, index=nid.unsqueeze(-1).expand(-1, -1, NDIM))
        ed = nv - v
        el = torch.linalg.norm(ed, dim=-1)
        if bool(((el <= 1e-12) & em).any().detach().cpu()):
            raise ValueError("obstacle polygons cannot contain zero-length edges")

        cr = v[..., 0] * nv[..., 1] - v[..., 1] * nv[..., 0]
        area2 = torch.sum(torch.where(em, cr, torch.zeros_like(cr)), dim=-1)
        if bool((torch.abs(area2) <= 1e-12).any().detach().cpu()):
            raise ValueError("obstacle polygons must have non-zero signed area")

        rnorm = torch.stack((ed[..., 1], -ed[..., 0]), dim=-1)
        ori = torch.where(area2 >= 0, torch.ones_like(area2), -torch.ones_like(area2))
        n = rnorm * ori[:, None, None]
        n = n / torch.clamp(el[..., None], min=1e-12)
        n = torch.where(em[..., None], n, torch.zeros_like(n))
        b = torch.sum(v * n, dim=-1)
        b = torch.where(em, b, torch.zeros_like(b))
        inf = torch.full_like(v, torch.inf)
        vmin = torch.where(em[..., None], v, inf)
        vmax = torch.where(em[..., None], v, -inf)
        lo_aabb = vmin.amin(dim=1)
        hi_aabb = vmax.amax(dim=1)
        return v, n, b, em, cnt, lo_aabb, hi_aabb

    def _buffers_for(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """内部版障碍物缓存检查；公共调用请优先使用 `obstacle_buffers`。

        参数：
            reference: 用于校验 dtype / device 的参考张量。

        返回：
            normals: ``(obstacles, max_vertices, 2)``
            offsets: ``(obstacles, max_vertices)``
            edge_mask: ``(obstacles, max_vertices)``
            aabb_lower: ``(obstacles, 2)``
            aabb_upper: ``(obstacles, 2)``
        """
        exp_dev = self.obstacle_edge_normals.device
        exp_dt = self.obstacle_edge_normals.dtype
        if reference.device != exp_dev or reference.dtype != exp_dt:
            raise ValueError(
                "inputs must use the same dtype/device as scene buffers; "
                f"got dtype={reference.dtype}, device={reference.device}, "
                f"expected dtype={exp_dt}, device={exp_dev}."
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

        参数：
            positions: ``(..., agents, 2)``，每个 batch 下的 agent 位置。
            active: ``(..., agents)``，对应 agent 是否参与判定。

        返回：
            ``(..., agents)`` 的布尔张量，表示最终被判定为点碰撞的 agent。

        关键路径：
            1. 取出预计算的候选 pair；
            2. 计算 pair 间平方距离并排序；
            3. 按从近到远的顺序贪心选择，保证每个 agent 最多只参与一次碰撞；
            4. 输出与输入 batch 形状一致的碰撞 mask。
        """
        pos = torch.as_tensor(positions)
        if pos.device != self.position_lower.device or pos.dtype != self.position_lower.dtype:
            raise ValueError(
                "positions must use the same dtype/device as scene buffers; "
                f"got dtype={pos.dtype}, device={pos.device}, "
                f"expected dtype={self.position_lower.dtype}, device={self.position_lower.device}."
            )
        act = torch.as_tensor(active, device=pos.device).bool()
        if self.collision_radius <= 0.0:
            return torch.zeros_like(act)

        pi = self.collision_pair_i
        pj = self.collision_pair_j
        if pi.numel() == 0:
            return torch.zeros_like(act)

        xi = pos.index_select(-2, pi)
        xj = pos.index_select(-2, pj)
        d2 = torch.sum((xi - xj) * (xi - xj), dim=-1)
        ap = act.index_select(-1, pi) & act.index_select(-1, pj)
        cand = ap & (d2 < self.collision_radius * self.collision_radius)
        sd, si = torch.sort(d2, dim=-1)
        sc = torch.gather(cand, -1, si) & torch.isfinite(sd)
        s_i = pi[si]
        s_j = pj[si]

        mat = torch.zeros_like(act)
        for k in range(int(pi.numel())):
            i_idx = s_i[..., k]
            j_idx = s_j[..., k]
            i_tk = torch.gather(mat, -1, i_idx.unsqueeze(-1)).squeeze(-1)
            j_tk = torch.gather(mat, -1, j_idx.unsqueeze(-1)).squeeze(-1)
            sel = sc[..., k] & ~i_tk & ~j_tk
            mat = mat | (
                F.one_hot(i_idx, self.num_agents).bool()
                | F.one_hot(j_idx, self.num_agents).bool()
            ) & sel.unsqueeze(-1)
        return mat

    def obstacle_signed_edge_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """批量计算点到所有障碍边的有符号距离。

        参数：
            positions: ``(..., 2)`` 或 ``(..., agents, 2)``，末维必须是二维坐标。

        返回：
            ``(..., obstacles, max_vertices)`` 的有符号边距离。
            无效 padding 边会被写成 ``-inf``，便于后续直接做 ``amax``。

        关键路径：
            1. 取出预处理好的边法向与 offset；
            2. 计算点到每条边所在半空间的 signed distance；
            3. 用 edge mask 屏蔽 padding 边，避免污染后续 max 聚合。
        """
        pos = torch.as_tensor(positions)
        if self.num_obstacles == 0:
            return torch.empty(
                (*pos.shape[:-1], 0, self.max_obstacle_vertices),
                dtype=pos.dtype,
                device=pos.device,
            )
        n, b, em, _, _ = self._buffers_for(pos)
        sd = torch.einsum("...d,oed->...oe", pos, n) - b
        ms = (1,) * (sd.ndim - 2) + em.shape
        return sd.masked_fill(~em.view(*ms), -torch.inf)

    def obstacle_clearance(self, positions: torch.Tensor) -> torch.Tensor:
        """计算点到每个障碍物的 clearance（最大边约束值）。

        参数：
            positions: ``(..., 2)`` 或 ``(..., agents, 2)``。

        返回：
            ``(..., obstacles)``，每个障碍的最大边约束值。
            值越大表示越“靠外”；当 ``<= obstacle_collision_margin`` 时视作碰撞。

        关键路径：
            1. 先算所有边的 signed distance；
            2. 再沿边维度做 ``amax``，得到每个障碍的最紧约束。
        """
        sd = self.obstacle_signed_edge_distances(positions)
        if sd.shape[-2] == 0:
            return torch.empty((*sd.shape[:-2], 0), dtype=sd.dtype, device=sd.device)
        return sd.amax(dim=-1)

    def obstacle_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """根据障碍物 clearance 判定碰撞。

        参数：
            positions: ``(..., agents, 2)``。
            active: ``(..., agents)``。

        返回：
            ``(..., agents)`` 的布尔 mask，表示哪些 agent 与任一障碍物发生碰撞。

        关键路径：
            1. 计算每个 agent 对所有障碍的 clearance；
            2. 与碰撞余量比较得到 hit；
            3. 对障碍维做 ``any``，得到每个 agent 的最终碰撞结果。
        """
        pos = torch.as_tensor(positions)
        act = torch.as_tensor(active, device=pos.device).bool()
        if self.num_obstacles == 0:
            return torch.zeros_like(act)
        hit = self.obstacle_clearance(pos) <= self.obstacle_collision_margin
        return act & hit.any(dim=-1)

    def bounds_validity(self, positions: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """检查边界合法性。

        参数：
            positions: ``(..., agents, 2)``。
            active: ``(..., agents)``。

        返回：
            valid: ``(...)``，表示该 batch 是否仍然全部合法。
            oob: ``(..., agents)``，每个 agent 是否越界。

        关键路径：
            1. 逐坐标检查是否落在 ``position_lower/upper`` 内；
            2. 根据 ``bounds_check_active_only`` 决定只检查 active 还是检查全部；
            3. 归约得到 batch 级 ``valid``。
        """
        pos = torch.as_tensor(positions)
        if pos.device != self.position_lower.device or pos.dtype != self.position_lower.dtype:
            raise ValueError(
                "positions must use the same dtype/device as scene buffers; "
                f"got dtype={pos.dtype}, device={pos.device}, "
                f"expected dtype={self.position_lower.dtype}, device={self.position_lower.device}."
            )
        lo = self.position_lower
        hi = self.position_upper
        inb = torch.all((pos >= lo) & (pos <= hi), dim=-1)
        act = torch.as_tensor(active, device=pos.device).bool()
        if self.bounds_check_active_only:
            oob = act & ~inb
        else:
            oob = ~inb
        return ~oob.any(dim=-1), oob

    def team_alive(self, active: torch.Tensor) -> torch.Tensor:
        """统计每个 team 是否仍有存活 agent。

        参数：
            active: ``(..., agents)``。

        返回：
            ``(..., num_teams)``，每个 team 是否还存在 active agent。

        关键路径：
            1. 按 team_id 做布尔筛选；
            2. 在 agent 维度上做 ``any``；
            3. 叠成 team 级别状态向量。
        """
        act = torch.as_tensor(active).bool()
        if act.device != self.team_ids.device:
            raise ValueError(
                "active masks must use the same device as scene buffers; "
                f"got device={act.device}, expected device={self.team_ids.device}."
            )
        tids = self.team_ids
        tvals = self.team_values
        alv = [(act & (tids == tv)).any(dim=-1) for tv in tvals]
        return torch.stack(alv, dim=-1)

    def collision_summary(self, positions: torch.Tensor, active: torch.Tensor) -> SceneCollisionResult:
        """一次性返回点碰撞、障碍碰撞、越界与有效性汇总。

        参数：
            positions: ``(..., agents, 2)``。
            active: ``(..., agents)``。

        返回：
            `SceneCollisionResult`，其中包含：
            - point_collision_mask: ``(..., agents)``
            - obstacle_collision_mask: ``(..., agents)``
            - collision_mask: ``(..., agents)``
            - out_of_bounds_mask: ``(..., agents)``
            - valid: ``(...)``

        关键路径：
            1. 分别计算点碰撞与障碍碰撞；
            2. 计算越界与 batch 有效性；
            3. 聚合成一个轻量结果对象，便于上层 rollout 直接消费。
        """
        pt = self.point_collision_mask(positions, active)
        obs = self.obstacle_collision_mask(positions, active)
        ok, oob = self.bounds_validity(positions, active)
        return SceneCollisionResult(
            point_collision_mask=pt,
            obstacle_collision_mask=obs,
            collision_mask=pt | obs,
            out_of_bounds_mask=oob,
            valid=ok,
        )
