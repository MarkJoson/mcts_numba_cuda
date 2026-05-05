"""基于 MINCO 的多智能体环境编排入口。"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .collision_ops import MincoCollisionOps
from .constants import DEFAULT_RATIO, NDIM, NCOFF
from .game_logic import MincoPointEnvStep, gather_step_flat, step, step_flat
from .projection_ops import MincoProjectionOps
from .scene import MincoScene
from .transition import MincoTorchTransition


class MincoPointEnvTransition(torch.nn.Module):
    """基于 MINCO 的多智能体点模型环境。

    flat state layout:
        ``[time, active_0..active_N-1, coeff(agent0), ..., coeff(agentN-1)]``
    """

    def __init__(
        self,
        *,
        num_agents: int = 4,
        piece_t: float = 0.1,
        dt: float | None = None,
        ratio: float = DEFAULT_RATIO,
        n_checkpoints: int = 20,
        velocity_limit: float = 2.0,
        acceleration_limit: float = 3.0,
        collision_radius: float = 1.0,
        team_ids: Sequence[int] | torch.Tensor | None = None,
        position_bounds: Sequence[Sequence[float]] | torch.Tensor | None = ((-10.0, 10.0), (-10.0, 10.0)),
        obstacle_vertices: Sequence[Sequence[Sequence[float]]] | torch.Tensor | None = None,
        obstacle_vertex_counts: Sequence[int] | torch.Tensor | None = None,
        obstacle_collision_margin: float = 0.0,
        obstacle_target_projection: bool = False,
        obstacle_projection_iters: int = 3,
        obstacle_projection_extra_margin: float = 1e-3,
        obstacle_projection_topk: int = 8,
        obstacle_projection_fixes_per_iter: int = 2,
        obstacle_projection_check_active_only: bool = True,
        max_obstacle_vertices: int = 8,
        tf: float = 80.0,
        dtype: torch.dtype | None = torch.float32,
        device: torch.device | str | None = None,
        collide_same_team: bool = False,
        deactivate_on_collision: bool = True,
        deactivate_on_obstacle_collision: bool = True,
        done_on_out_of_bounds: bool = True,
        done_on_obstacle_collision: bool = True,
        done_on_team_eliminated: bool = True,
        bounds_check_active_only: bool = False,
    ) -> None:
        super().__init__()
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        if collision_radius < 0:
            raise ValueError(f"collision_radius must be non-negative, got {collision_radius}")
        if obstacle_collision_margin < 0:
            raise ValueError(
                f"obstacle_collision_margin must be non-negative, got {obstacle_collision_margin}")
        if obstacle_projection_iters < 0:
            raise ValueError(
                f"obstacle_projection_iters must be non-negative, got {obstacle_projection_iters}")
        if obstacle_projection_extra_margin < 0:
            raise ValueError(
                "obstacle_projection_extra_margin must be non-negative, "
                f"got {obstacle_projection_extra_margin}")
        if obstacle_projection_topk <= 0:
            raise ValueError(
                f"obstacle_projection_topk must be positive, got {obstacle_projection_topk}")
        if obstacle_projection_fixes_per_iter <= 0:
            raise ValueError(
                "obstacle_projection_fixes_per_iter must be positive, "
                f"got {obstacle_projection_fixes_per_iter}")
        if max_obstacle_vertices < 3:
            raise ValueError(
                f"max_obstacle_vertices must be at least 3, got {max_obstacle_vertices}")
        if tf <= 0:
            raise ValueError(f"tf must be positive, got {tf}")

        self.num_agents = int(num_agents)
        self.dt = float(piece_t if dt is None else dt)
        self.tf = float(tf)
        self.collision_radius = float(collision_radius)
        self.obstacle_collision_margin = float(obstacle_collision_margin)
        self.obstacle_target_projection = bool(obstacle_target_projection)
        self.obstacle_projection_iters = int(obstacle_projection_iters)
        self.obstacle_projection_extra_margin = float(obstacle_projection_extra_margin)
        self.obstacle_projection_topk = int(obstacle_projection_topk)
        self.obstacle_projection_fixes_per_iter = int(obstacle_projection_fixes_per_iter)
        self.obstacle_projection_check_active_only = bool(obstacle_projection_check_active_only)
        self.collide_same_team = bool(collide_same_team)
        self.deactivate_on_collision = bool(deactivate_on_collision)
        self.deactivate_on_obstacle_collision = bool(deactivate_on_obstacle_collision)
        self.done_on_out_of_bounds = bool(done_on_out_of_bounds)
        self.done_on_obstacle_collision = bool(done_on_obstacle_collision)
        self.done_on_team_eliminated = bool(done_on_team_eliminated)
        self.bounds_check_active_only = bool(bounds_check_active_only)
        self.max_obstacle_vertices = int(max_obstacle_vertices)

        runtime_dtype = dtype or torch.float32
        build_device = torch.device(device) if device is not None else None

        self.robot_transition = MincoTorchTransition(
            piece_t,
            ratio=ratio,
            n_checkpoints=n_checkpoints,
            velocity_limit=velocity_limit,
            acceleration_limit=acceleration_limit,
            dtype=runtime_dtype,
            device=build_device,
        )
        self.scene = MincoScene(
            num_agents=self.num_agents,
            collision_radius=self.collision_radius,
            position_bounds=position_bounds,
            team_ids=team_ids,
            collide_same_team=self.collide_same_team,
            bounds_check_active_only=self.bounds_check_active_only,
            obstacle_vertices=obstacle_vertices,
            obstacle_vertex_counts=obstacle_vertex_counts,
            obstacle_collision_margin=self.obstacle_collision_margin,
            obstacle_projection_check_active_only=self.obstacle_projection_check_active_only,
            max_obstacle_vertices=self.max_obstacle_vertices,
            dtype=runtime_dtype,
            device=build_device,
        )
        self.collision_ops = MincoCollisionOps(self.scene)
        self.projection_ops = MincoProjectionOps(
            scene=self.scene,
            robot_transition=self.robot_transition,
            collision_ops=self.collision_ops,
            obstacle_collision_margin=self.obstacle_collision_margin,
            obstacle_projection_iters=self.obstacle_projection_iters,
            obstacle_projection_extra_margin=self.obstacle_projection_extra_margin,
            obstacle_projection_topk=self.obstacle_projection_topk,
            obstacle_projection_fixes_per_iter=self.obstacle_projection_fixes_per_iter,
            obstacle_projection_check_active_only=self.obstacle_projection_check_active_only,
        )

    @property
    def coeff_state_dim(self) -> int:
        return NCOFF * NDIM

    @property
    def state_dim(self) -> int:
        return 1 + self.num_agents + self.num_agents * self.coeff_state_dim

    @property
    def num_obstacles(self) -> int:
        return int(self.scene.num_obstacles)

    @property
    def team_ids(self) -> torch.Tensor:
        return self.scene.team_ids

    @property
    def team_values(self) -> torch.Tensor:
        return self.scene.team_values

    @property
    def collision_pair_i(self) -> torch.Tensor:
        return self.scene.collision_pair_i

    @property
    def collision_pair_j(self) -> torch.Tensor:
        return self.scene.collision_pair_j

    @property
    def position_lower(self) -> torch.Tensor:
        return self.scene.position_lower

    @property
    def position_upper(self) -> torch.Tensor:
        return self.scene.position_upper

    @property
    def obstacle_vertices(self) -> torch.Tensor:
        return self.scene.obstacle_vertices

    @property
    def obstacle_edge_normals(self) -> torch.Tensor:
        return self.scene.obstacle_edge_normals

    @property
    def obstacle_edge_offsets(self) -> torch.Tensor:
        return self.scene.obstacle_edge_offsets

    @property
    def obstacle_edge_mask(self) -> torch.Tensor:
        return self.scene.obstacle_edge_mask

    @property
    def obstacle_vertex_counts(self) -> torch.Tensor:
        return self.scene.obstacle_vertex_counts

    @property
    def obstacle_aabb_lower(self) -> torch.Tensor:
        return self.scene.obstacle_aabb_lower

    @property
    def obstacle_aabb_upper(self) -> torch.Tensor:
        return self.scene.obstacle_aabb_upper

    @property
    def obstacle_edge_points(self) -> torch.Tensor:
        return self.scene.obstacle_vertices

    def positions(self, coefficients: torch.Tensor) -> torch.Tensor:
        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-3:] != (self.num_agents, NCOFF, NDIM):
            raise ValueError(
                f"coefficients must end with ({self.num_agents}, {NCOFF}, {NDIM}), got {tuple(coeff.shape)}")
        return coeff[..., :, 0, :]

    def zero_motion(self, coefficients: torch.Tensor) -> torch.Tensor:
        coeff = torch.as_tensor(coefficients)
        frozen = torch.zeros_like(coeff)
        frozen[..., :, 0, :] = coeff[..., :, 0, :]
        return frozen

    def initial_coefficients(self, positions: torch.Tensor | Iterable[float]) -> torch.Tensor:
        pos = torch.as_tensor(
            positions,
            dtype=self.robot_transition.mat_f_stab.dtype,
            device=self.robot_transition.mat_f_stab.device,
        )
        if pos.shape[-2:] != (self.num_agents, NDIM):
            raise ValueError(f"positions must end with ({self.num_agents}, {NDIM}), got {tuple(pos.shape)}")
        coeff = torch.zeros((*pos.shape[:-2], self.num_agents, NCOFF, NDIM), dtype=pos.dtype, device=pos.device)
        coeff[..., :, 0, :] = pos
        return coeff

    def pack_state(
        self,
        coefficients: torch.Tensor,
        active: torch.Tensor | Iterable[float] | None = None,
        time: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-3:] != (self.num_agents, NCOFF, NDIM):
            raise ValueError(f"coefficients must end with ({self.num_agents}, {NCOFF}, {NDIM}), got {tuple(coeff.shape)}")
        batch_shape = coeff.shape[:-3]
        if active is None:
            active_tensor = torch.ones((*batch_shape, self.num_agents), dtype=coeff.dtype, device=coeff.device)
        else:
            active_tensor = torch.as_tensor(active, dtype=coeff.dtype, device=coeff.device)
            active_tensor = active_tensor.expand((*batch_shape, self.num_agents))
        time_tensor = torch.as_tensor(time, dtype=coeff.dtype, device=coeff.device)
        time_tensor = time_tensor.expand(batch_shape)
        return torch.cat(
            [time_tensor.unsqueeze(-1), active_tensor, coeff.view(*batch_shape, self.num_agents * self.coeff_state_dim)],
            dim=-1,
        )

    def unpack_state(self, flat_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = torch.as_tensor(flat_state)
        if flat.shape[-1] != self.state_dim:
            raise ValueError(f"flat_state must end with {self.state_dim}, got {tuple(flat.shape)}")
        time = flat[..., 0]
        active = flat[..., 1:1 + self.num_agents]
        coeff = flat[..., 1 + self.num_agents:].view(*flat.shape[:-1], self.num_agents, NCOFF, NDIM)
        return coeff, active, time

    def initial_flat_state(
        self,
        positions: torch.Tensor | Iterable[float],
        active: torch.Tensor | Iterable[float] | None = None,
        time: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        return self.pack_state(self.initial_coefficients(positions), active=active, time=time)

    def _target_from_action(
        self,
        coefficients: torch.Tensor,
        target_or_delta: torch.Tensor,
        *,
        target_is_delta: bool,
    ) -> torch.Tensor:
        target = torch.as_tensor(target_or_delta, dtype=coefficients.dtype, device=coefficients.device)
        if target.shape[-2:] != (self.num_agents, NDIM):
            raise ValueError(f"target positions/actions must end with ({self.num_agents}, {NDIM}), got {tuple(target.shape)}")
        if target_is_delta:
            target = self.positions(coefficients) + target
        return target

    def _future_positions_from_target(self, coefficients: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        return self.projection_ops.future_positions_from_target(coefficients, target_pos)

    def future_obstacle_clearance_from_target(self, coefficients: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        return self.projection_ops.future_obstacle_clearance_from_target(coefficients, target_pos)

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
        return self.projection_ops.project_target_away_from_obstacles(
            coefficients,
            target_pos,
            active,
            clamp_to_dynamic_bounds=clamp_to_dynamic_bounds,
            iterations=iterations,
            return_residual=return_residual,
        )

    def collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.collision_mask(positions, active)

    def obstacle_signed_edge_distances(self, positions: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.obstacle_signed_edge_distances(positions)

    def _obstacle_buffers_for(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.collision_ops.obstacle_buffers(reference)

    def _obstacle_signed_edge_distances_any_agents(self, positions: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.obstacle_signed_edge_distances_any_agents(positions)

    def _obstacle_clearance_any_agents_broadphase(self, positions: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.obstacle_clearance_any_agents_broadphase(positions)

    def obstacle_clearance(self, positions: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.obstacle_clearance(positions)

    def obstacle_collision_mask(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.obstacle_collision_mask(positions, active)

    def bounds_validity(self, positions: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.collision_ops.bounds_validity(positions, active)

    def team_alive(self, active: torch.Tensor) -> torch.Tensor:
        return self.collision_ops.team_alive(active)

    def done(
        self,
        active: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        obstacle_collision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        done_mask = torch.as_tensor(time) >= self.tf
        if self.done_on_out_of_bounds:
            done_mask = done_mask | ~torch.as_tensor(valid, device=done_mask.device).bool()
        if self.done_on_obstacle_collision and obstacle_collision is not None:
            done_mask = done_mask | torch.as_tensor(obstacle_collision, device=done_mask.device).bool().any(dim=-1)
        if self.done_on_team_eliminated:
            done_mask = done_mask | ~self.team_alive(active).all(dim=-1)
        return done_mask

    def step(
        self,
        coefficients: torch.Tensor,
        active: torch.Tensor,
        time: torch.Tensor | float,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        target_is_delta: bool = False,
        return_projection_residual: bool = True,
    ) -> MincoPointEnvStep:
        return step(
            self,
            coefficients,
            active,
            time,
            target_pos,
            clamp_target=clamp_target,
            target_is_delta=target_is_delta,
            return_projection_residual=return_projection_residual,
        )

    def step_flat(
        self,
        flat_state: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        target_is_delta: bool = False,
        return_info: bool = False,
    ):
        return step_flat(
            self,
            flat_state,
            target_pos,
            clamp_target=clamp_target,
            target_is_delta=target_is_delta,
            return_info=return_info,
        )

    def gather_step_flat(
        self,
        flat_state_pool: torch.Tensor,
        tree_ids: torch.Tensor,
        node_ids: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        target_is_delta: bool = False,
        return_info: bool = False,
    ):
        return gather_step_flat(
            self,
            flat_state_pool,
            tree_ids,
            node_ids,
            target_pos,
            clamp_target=clamp_target,
            target_is_delta=target_is_delta,
            return_info=return_info,
        )
