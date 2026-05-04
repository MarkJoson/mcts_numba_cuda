"""Batched PyTorch port of the ``experiment2.py`` MINCO transition.

The original script builds a small closed-loop MINCO planner and advances one
6x2 coefficient state at a time.  This module keeps all one-time algebra in
``__init__`` and leaves only batched tensor operations for rollout/expand:

* initialization: basis matrices, Riccati/LQR gain, stable transition matrix,
  and checkpoint constraint coefficients;
* per step: optional target projection with precomputed constraints, then one
  batched linear transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


NCOFF = 6
NDIM = 2
S = 3
DEFAULT_RATIO = 0.1


def _as_dtype(dtype: torch.dtype | None) -> torch.dtype:
    return torch.float32 if dtype is None else dtype


def construct_beta_t(
    t: float | torch.Tensor,
    rank: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return beta(t, rank) with shape ``(6, 1)``.

    This mirrors ``experiment2.constructBetaT`` exactly, including the
    derivative-at-zero special case.
    """

    if not 0 <= int(rank) < NCOFF:
        raise ValueError(f"rank must be in [0, {NCOFF}), got {rank}")

    beta = torch.zeros((NCOFF, 1), dtype=dtype, device=device)
    t_tensor = torch.as_tensor(t, dtype=dtype, device=device)
    is_python_zero = isinstance(t, (int, float)) and float(t) == 0.0

    for i in range(rank, NCOFF):
        if (not is_python_zero) or i - rank == 0:
            coeff = math.factorial(i) / math.factorial(i - rank)
            beta[i, 0] = coeff * torch.pow(t_tensor, i - rank)
    return beta


def construct_bb_t_int(
    piece_t: float,
    rank: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``integral beta beta^T dt`` used by the MINCO/LQR objective."""

    beta = construct_beta_t(piece_t, rank, dtype=dtype, device=device)
    bbint = torch.zeros((NCOFF, NCOFF), dtype=dtype, device=device)
    piece = torch.as_tensor(piece_t, dtype=dtype, device=device)
    for i in range(NCOFF):
        for j in range(NCOFF):
            denom = i + j - 2 * rank + 1
            if denom <= 0:
                continue
            bbint[i, j] = beta[i, 0] * beta[j, 0] * piece / float(denom)
    return bbint


def construct_minco_m2(
    piece_t: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the modified MINCO boundary matrix from ``experiment2.py``."""

    mat_m = torch.zeros((NCOFF, NCOFF), dtype=dtype, device=device)
    for i in range(NCOFF - 2):
        mat_m[i, :] = construct_beta_t(0.0, i, dtype=dtype, device=device).T

    mat_m[-2, :] = construct_beta_t(piece_t, 0, dtype=dtype, device=device).T
    mat_m[-1, :] = construct_beta_t(piece_t, 1, dtype=dtype, device=device).T

    mat_m_inv = torch.linalg.inv(mat_m)
    selector = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                            dtype=dtype, device=device)
    mat_supp = selector @ mat_m_inv @ construct_bb_t_int(
        piece_t, S, dtype=dtype, device=device)
    mat_m[-1, :] = mat_supp[-1, :]
    return mat_m


def construct_mat_r(
    piece_t: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return rows ``beta(piece_t, rank)^T`` for ranks 0..5."""

    rows = [
        construct_beta_t(piece_t, i, dtype=dtype, device=device).T
        for i in range(NCOFF)
    ]
    return torch.cat(rows, dim=0)


def solve_discrete_lqr_gain(
    a_mat: torch.Tensor,
    b_mat: torch.Tensor,
    q_mat: torch.Tensor,
    r_mat: torch.Tensor,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Solve the infinite-horizon discrete LQR gain by Riccati iteration.

    ``python-control`` is not needed at runtime.  The problem is only 6x1, so
    this one-time fixed-point iteration is negligible next to MCTS rollout.
    The returned gain has the same sign convention as ``control.dlqr``:
    ``x_next = (A - B @ K) x``.
    """

    p_mat = q_mat.clone()
    for _ in range(int(max_iter)):
        bt_p = b_mat.T @ p_mat
        gain = torch.linalg.solve(r_mat + bt_p @ b_mat, bt_p @ a_mat)
        next_p = q_mat + a_mat.T @ p_mat @ a_mat - a_mat.T @ p_mat @ b_mat @ gain
        delta = torch.linalg.norm(next_p - p_mat)
        scale = torch.clamp(torch.linalg.norm(p_mat), min=torch.tensor(
            1.0, dtype=p_mat.dtype, device=p_mat.device))
        p_mat = next_p
        if float((delta / scale).detach().cpu()) <= tol:
            break

    bt_p = b_mat.T @ p_mat
    return torch.linalg.solve(r_mat + bt_p @ b_mat, bt_p @ a_mat)


class MincoTorchTransition(torch.nn.Module):
    """Vectorized MINCO closed-loop transition for MCTS expand/rollout.

    State layout is ``(..., 6, 2)`` where the last axis is ``x/y`` and the
    coefficient axis matches ``experiment2.py``.  Flattened ``(..., 12)``
    helpers are provided for tree state arrays that prefer a vector layout.
    """

    def __init__(
        self,
        piece_t: float,
        *,
        ratio: float = DEFAULT_RATIO,
        n_checkpoints: int = 20,
        velocity_limit: float = 2.0,
        acceleration_limit: float = 3.0,
        dtype: torch.dtype | None = torch.float32,
        device: torch.device | str | None = None,
        lqr_max_iter: int = 10_000,
        lqr_tol: float = 1e-12,
    ) -> None:
        super().__init__()
        if piece_t <= 0:
            raise ValueError(f"piece_t must be positive, got {piece_t}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if n_checkpoints <= 0:
            raise ValueError(f"n_checkpoints must be positive, got {n_checkpoints}")

        self.piece_t = float(piece_t)
        self.ratio = float(ratio)
        self.planner_piece_t = float(piece_t) / float(ratio)
        self.n_checkpoints = int(n_checkpoints)
        self.velocity_limit = float(velocity_limit)
        self.acceleration_limit = float(acceleration_limit)

        runtime_dtype = _as_dtype(dtype)
        build_dtype = torch.float64
        build_device = torch.device(device) if device is not None else None

        mat_m = construct_minco_m2(
            self.planner_piece_t, dtype=build_dtype, device=build_device)
        mat_m_inv = torch.linalg.inv(mat_m)
        mat_r = construct_mat_r(
            self.piece_t, dtype=build_dtype, device=build_device)
        mat_s = torch.diag(torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            dtype=build_dtype,
            device=build_device,
        ))
        mat_u = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]],
                             dtype=build_dtype, device=build_device)

        mat_f = mat_m_inv @ mat_s @ mat_r
        mat_g = mat_m_inv @ mat_u

        q_mat = construct_bb_t_int(
            self.piece_t, S, dtype=build_dtype, device=build_device)
        r_mat = torch.ones((1, 1), dtype=build_dtype, device=build_device)
        lqr_k = solve_discrete_lqr_gain(
            mat_f, mat_g, q_mat, r_mat,
            max_iter=lqr_max_iter,
            tol=lqr_tol,
        )

        kpp = torch.linalg.pinv(mat_g) @ (
            torch.eye(NCOFF, dtype=build_dtype, device=build_device) - mat_f
        ) + lqr_k
        mat_f_stab = mat_f - mat_g @ lqr_k
        mat_g_stab = mat_g @ kpp @ torch.tensor(
            [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            dtype=build_dtype,
            device=build_device,
        )

        constraint_const, constraint_gain, constraint_limit = (
            self._build_constraint_tensors(
                mat_f_stab,
                mat_g_stab,
                dtype=build_dtype,
                device=build_device,
            )
        )
        signed_limit = torch.where(
            constraint_gain < 0.0,
            -constraint_limit,
            constraint_limit,
        )
        inv_gain = torch.reciprocal(constraint_gain)
        constraint_const_scaled = constraint_const * inv_gain.reshape(-1, 1)
        constraint_lower_bias = -signed_limit * inv_gain
        constraint_upper_bias = signed_limit * inv_gain

        self.register_buffer("mat_f", mat_f.to(runtime_dtype))
        self.register_buffer("mat_g", mat_g.to(runtime_dtype))
        self.register_buffer("lqr_k", lqr_k.to(runtime_dtype))
        self.register_buffer("kpp", kpp.to(runtime_dtype))
        self.register_buffer("mat_f_stab", mat_f_stab.to(runtime_dtype))
        self.register_buffer("mat_g_stab", mat_g_stab.to(runtime_dtype))
        self.register_buffer("constraint_const", constraint_const.to(runtime_dtype))
        self.register_buffer("constraint_gain", constraint_gain.to(runtime_dtype))
        self.register_buffer("constraint_signed_limit", signed_limit.to(runtime_dtype))
        self.register_buffer("constraint_const_scaled", constraint_const_scaled.to(runtime_dtype))
        self.register_buffer("constraint_lower_bias", constraint_lower_bias.to(runtime_dtype))
        self.register_buffer("constraint_upper_bias", constraint_upper_bias.to(runtime_dtype))

    def _build_constraint_tensors(
        self,
        mat_f_stab: torch.Tensor,
        mat_g_stab: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        const_rows = []
        gain_vals = []
        limits = []
        for rank, limit in ((1, self.velocity_limit), (2, self.acceleration_limit)):
            for i in range(self.n_checkpoints):
                t = (i + 1) / (self.n_checkpoints + 1) * self.planner_piece_t
                beta_t = construct_beta_t(t, rank, dtype=dtype, device=device).T
                const_rows.append((beta_t @ mat_f_stab).reshape(NCOFF))
                gain_vals.append((beta_t @ mat_g_stab).reshape(()))
                limits.append(float(limit))

        return (
            torch.stack(const_rows, dim=0),
            torch.stack(gain_vals, dim=0),
            torch.tensor(limits, dtype=dtype, device=device),
        )

    @property
    def state_dim(self) -> int:
        return NCOFF * NDIM

    def initial_coefficients(
        self,
        init_pos: torch.Tensor | Iterable[float],
        *,
        batch_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Create a coefficient state with only the constant term populated."""

        pos = torch.as_tensor(init_pos, dtype=self.mat_f_stab.dtype,
                              device=self.mat_f_stab.device)
        if pos.shape[-1] != NDIM:
            raise ValueError(f"init_pos must end with dimension {NDIM}, got {pos.shape}")
        if batch_shape == () and pos.ndim > 1:
            batch_shape = tuple(pos.shape[:-1])
        coeff = torch.zeros(
            (*batch_shape, NCOFF, NDIM),
            dtype=self.mat_f_stab.dtype,
            device=self.mat_f_stab.device,
        )
        coeff[..., 0, :] = pos.expand((*batch_shape, NDIM))
        return coeff

    def _buffers_for(self, reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.mat_f_stab.to(device=reference.device, dtype=reference.dtype),
            self.mat_g_stab.to(device=reference.device, dtype=reference.dtype),
            self.constraint_const_scaled.to(device=reference.device, dtype=reference.dtype),
            self.constraint_lower_bias.to(device=reference.device, dtype=reference.dtype),
            self.constraint_upper_bias.to(device=reference.device, dtype=reference.dtype),
        )

    def bounds(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target lower/upper bounds with shape ``(..., 2)``."""

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-2:] != (NCOFF, NDIM):
            raise ValueError(
                f"coefficients must have trailing shape {(NCOFF, NDIM)}, "
                f"got {tuple(coeff.shape)}")

        _, _, const_coeff, lower_bias, upper_bias = self._buffers_for(coeff)
        const_val = torch.einsum("ci,...id->...cd", const_coeff, coeff)

        view_shape = (1,) * (const_val.ndim - 2) + (-1, 1)
        lower_bias = lower_bias.reshape(view_shape)
        upper_bias = upper_bias.reshape(view_shape)

        lower_each = lower_bias - const_val
        upper_each = upper_bias - const_val
        lower = torch.max(lower_each, dim=-2).values
        upper = torch.min(upper_each, dim=-2).values
        return lower, upper

    def project_target(
        self,
        coefficients: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Clip target positions to the precomputed velocity/acceleration bounds."""

        coeff = torch.as_tensor(coefficients)
        target = torch.as_tensor(target_pos, device=coeff.device,
                                 dtype=coeff.dtype)
        if target.shape[-2:] == (1, NDIM):
            target = target.squeeze(-2)
        if target.shape[-1] != NDIM:
            raise ValueError(f"target_pos must end with dimension {NDIM}, got {target.shape}")

        lower, upper = self.bounds(coeff)
        return torch.minimum(torch.maximum(target, lower), upper)

    def transition(
        self,
        coefficients: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``F_stab @ coeff + G_stab @ target`` in batch."""

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-2:] != (NCOFF, NDIM):
            raise ValueError(
                f"coefficients must have trailing shape {(NCOFF, NDIM)}, "
                f"got {tuple(coeff.shape)}")
        target = torch.as_tensor(target_pos, device=coeff.device, dtype=coeff.dtype)
        if target.shape[-2:] == (1, NDIM):
            target = target.squeeze(-2)
        if target.shape[-1] != NDIM:
            raise ValueError(f"target_pos must end with dimension {NDIM}, got {target.shape}")

        mat_f, mat_g, _, _, _ = self._buffers_for(coeff)
        linear = torch.einsum("ij,...jd->...id", mat_f, coeff)
        forced = mat_g.reshape(NCOFF, 1) * target.unsqueeze(-2)
        return linear + forced

    def step(
        self,
        coefficients: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_projected: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project the target if requested, then advance one MINCO state."""

        coeff = torch.as_tensor(coefficients)
        target = self.project_target(coeff, target_pos) if clamp_target else target_pos
        next_coeff = self.transition(coeff, target)
        if return_projected:
            return next_coeff, torch.as_tensor(target, device=next_coeff.device,
                                               dtype=next_coeff.dtype)
        return next_coeff

    def rollout(
        self,
        coefficients: torch.Tensor,
        target_sequence: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_history: bool = False,
    ) -> torch.Tensor:
        """Run a batched recurrent rollout over a leading time dimension.

        ``target_sequence`` must have shape ``(horizon, ..., 2)``.  The batch
        dimensions after ``horizon`` are broadcast against ``coefficients``.
        """

        cur = torch.as_tensor(coefficients)
        targets = torch.as_tensor(target_sequence, device=cur.device,
                                  dtype=cur.dtype)
        if targets.ndim < 2 or targets.shape[-1] != NDIM:
            raise ValueError(
                f"target_sequence must have shape (horizon, ..., {NDIM}), "
                f"got {tuple(targets.shape)}")

        history = [cur]
        for step_idx in range(targets.shape[0]):
            cur = self.step(cur, targets[step_idx], clamp_target=clamp_target)
            if return_history:
                history.append(cur)
        return torch.stack(history, dim=0) if return_history else cur

    def evaluate(
        self,
        coefficients: torch.Tensor,
        t: float,
        rank: int,
    ) -> torch.Tensor:
        """Evaluate position/velocity/acceleration-like derivatives in batch."""

        coeff = torch.as_tensor(coefficients)
        beta = construct_beta_t(
            float(t), int(rank), dtype=coeff.dtype, device=coeff.device,
        ).reshape(NCOFF)
        return torch.einsum("i,...id->...d", beta, coeff)

    def flatten_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Convert ``(..., 6, 2)`` states to ``(..., 12)`` vectors."""

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-2:] != (NCOFF, NDIM):
            raise ValueError(
                f"coefficients must have trailing shape {(NCOFF, NDIM)}, "
                f"got {tuple(coeff.shape)}")
        return coeff.reshape(*coeff.shape[:-2], NCOFF * NDIM)

    def unflatten_coefficients(self, flat_state: torch.Tensor) -> torch.Tensor:
        """Convert ``(..., 12)`` vectors to ``(..., 6, 2)`` states."""

        flat = torch.as_tensor(flat_state)
        if flat.shape[-1] != NCOFF * NDIM:
            raise ValueError(
                f"flat_state must end with dimension {NCOFF * NDIM}, got {flat.shape}")
        return flat.reshape(*flat.shape[:-1], NCOFF, NDIM)

    def step_flat(
        self,
        flat_state: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_projected: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Flat-state version of :meth:`step`, useful for MCTS tree buffers."""

        coeff = self.unflatten_coefficients(flat_state)
        stepped = self.step(
            coeff, target_pos,
            clamp_target=clamp_target,
            return_projected=return_projected,
        )
        if return_projected:
            next_coeff, projected = stepped
            return self.flatten_coefficients(next_coeff), projected
        return self.flatten_coefficients(stepped)

    def rollout_flat(
        self,
        flat_state: torch.Tensor,
        target_sequence: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_history: bool = False,
    ) -> torch.Tensor:
        """Flat-state version of :meth:`rollout`."""

        coeff = self.unflatten_coefficients(flat_state)
        result = self.rollout(
            coeff,
            target_sequence,
            clamp_target=clamp_target,
            return_history=return_history,
        )
        if return_history:
            return result.reshape(*result.shape[:-2], NCOFF * NDIM)
        return self.flatten_coefficients(result)

    def gather_step_flat(
        self,
        flat_state_pool: torch.Tensor,
        tree_ids: torch.Tensor,
        node_ids: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_projected: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Gather parent states from ``(trees, nodes, 12)`` and step them.

        This matches the DUCT/PUCT expand data flow: selection emits parent
        ``tree,node`` pairs plus a target point for each new edge; PyTorch
        gathers the parent coefficients and computes all next states in one
        batched call.  The caller can then scatter the returned flat states to
        newly allocated child node slots.
        """

        pool = torch.as_tensor(flat_state_pool)
        if pool.ndim != 3 or pool.shape[-1] != NCOFF * NDIM:
            raise ValueError(
                f"flat_state_pool must have shape (trees, nodes, {NCOFF * NDIM}), "
                f"got {tuple(pool.shape)}")

        trees = torch.as_tensor(tree_ids, device=pool.device, dtype=torch.long)
        nodes = torch.as_tensor(node_ids, device=pool.device, dtype=torch.long)
        parent_flat = pool[trees, nodes]
        return self.step_flat(
            parent_flat,
            target_pos,
            clamp_target=clamp_target,
            return_projected=return_projected,
        )


@dataclass(frozen=True)
class MincoPointEnvStep:
    """Structured output for a batched point-model environment step."""

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


class MincoPointEnvTransition(torch.nn.Module):
    """Batched multi-point environment transition on top of MINCO dynamics.

    The canonical flat state layout is:

    ``[time, active_0..active_N-1, coeff(agent0), ..., coeff(agentN-1)]``

    where each agent coefficient block is the flattened ``(6, 2)`` MINCO state.
    This keeps rollout/expand data contiguous while still allowing structured
    access to positions, active masks, and time.
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
        self.collide_same_team = bool(collide_same_team)
        self.deactivate_on_collision = bool(deactivate_on_collision)
        self.deactivate_on_obstacle_collision = bool(deactivate_on_obstacle_collision)
        self.done_on_out_of_bounds = bool(done_on_out_of_bounds)
        self.done_on_obstacle_collision = bool(done_on_obstacle_collision)
        self.done_on_team_eliminated = bool(done_on_team_eliminated)
        self.bounds_check_active_only = bool(bounds_check_active_only)
        self.max_obstacle_vertices = int(max_obstacle_vertices)

        runtime_dtype = _as_dtype(dtype)
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

        if team_ids is None:
            split = max(1, self.num_agents // 2)
            team_ids = [0 if i < split else 1 for i in range(self.num_agents)]
        team_tensor = torch.as_tensor(team_ids, dtype=torch.long, device=build_device)
        if team_tensor.shape != (self.num_agents,):
            raise ValueError(
                f"team_ids must have shape ({self.num_agents},), got {tuple(team_tensor.shape)}")
        self.register_buffer("team_ids", team_tensor)
        self.register_buffer("team_values", torch.unique(team_tensor, sorted=True))
        team_list = torch.as_tensor(team_ids, dtype=torch.long).detach().cpu().tolist()
        pair_i = []
        pair_j = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (not self.collide_same_team) and team_list[i] == team_list[j]:
                    continue
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer(
            "collision_pair_i",
            torch.tensor(pair_i, dtype=torch.long, device=build_device),
        )
        self.register_buffer(
            "collision_pair_j",
            torch.tensor(pair_j, dtype=torch.long, device=build_device),
        )

        if position_bounds is None:
            lower = torch.full((NDIM,), -torch.inf, dtype=runtime_dtype, device=build_device)
            upper = torch.full((NDIM,), torch.inf, dtype=runtime_dtype, device=build_device)
        else:
            bounds = torch.as_tensor(position_bounds, dtype=runtime_dtype, device=build_device)
            if bounds.shape != (NDIM, 2):
                raise ValueError(
                    f"position_bounds must have shape ({NDIM}, 2), got {tuple(bounds.shape)}")
            lower = bounds[:, 0]
            upper = bounds[:, 1]
        self.register_buffer("position_lower", lower)
        self.register_buffer("position_upper", upper)

        (
            obstacle_edge_points,
            obstacle_edge_normals,
            obstacle_edge_offsets,
            obstacle_edge_mask,
            obstacle_vertex_count_tensor,
        ) = self._prepare_obstacle_buffers(
            obstacle_vertices,
            obstacle_vertex_counts,
            max_vertices=self.max_obstacle_vertices,
            dtype=runtime_dtype,
            device=build_device,
        )
        self.register_buffer("obstacle_edge_points", obstacle_edge_points)
        self.register_buffer("obstacle_edge_normals", obstacle_edge_normals)
        self.register_buffer("obstacle_edge_offsets", obstacle_edge_offsets)
        self.register_buffer("obstacle_edge_mask", obstacle_edge_mask)
        self.register_buffer("obstacle_vertex_counts", obstacle_vertex_count_tensor)

    @property
    def coeff_state_dim(self) -> int:
        return NCOFF * NDIM

    @property
    def state_dim(self) -> int:
        return 1 + self.num_agents + self.num_agents * self.coeff_state_dim

    @property
    def num_obstacles(self) -> int:
        return int(self.obstacle_vertex_counts.numel())

    def _prepare_obstacle_buffers(
        self,
        obstacle_vertices: Sequence[Sequence[Sequence[float]]] | torch.Tensor | None,
        obstacle_vertex_counts: Sequence[int] | torch.Tensor | None,
        *,
        max_vertices: int,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obstacle_vertices is None:
            empty_edges = torch.empty((0, max_vertices, NDIM), dtype=dtype, device=device)
            return (
                empty_edges,
                empty_edges.clone(),
                torch.empty((0, max_vertices), dtype=dtype, device=device),
                torch.empty((0, max_vertices), dtype=torch.bool, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        try:
            vertices = torch.as_tensor(obstacle_vertices, dtype=dtype, device=device)
        except (TypeError, ValueError):
            polygons = [
                torch.as_tensor(poly, dtype=dtype, device=device)
                for poly in obstacle_vertices
            ]
            if not polygons:
                raise ValueError("obstacle_vertices cannot be an empty sequence")
            counts = torch.tensor([poly.shape[0] for poly in polygons],
                                  dtype=torch.long, device=device)
            max_count = int(counts.max().detach().cpu())
            if max_count > max_vertices:
                raise ValueError(
                    f"obstacle polygon has {max_count} vertices; max is {max_vertices}")
            vertices = torch.zeros((len(polygons), max_count, NDIM), dtype=dtype, device=device)
            for i, poly in enumerate(polygons):
                if poly.ndim != 2 or poly.shape[-1] != NDIM:
                    raise ValueError(
                        f"each obstacle polygon must have shape (vertices, {NDIM})")
                vertices[i, :poly.shape[0], :] = poly
        else:
            if vertices.ndim != 3 or vertices.shape[-1] != NDIM:
                raise ValueError(
                    f"obstacle_vertices must have shape (obstacles, vertices, {NDIM}), "
                    f"got {tuple(vertices.shape)}")
            if vertices.shape[1] > max_vertices:
                raise ValueError(
                    f"obstacle polygon has {vertices.shape[1]} vertices; max is {max_vertices}")
            if obstacle_vertex_counts is None:
                counts = torch.full((vertices.shape[0],), vertices.shape[1],
                                    dtype=torch.long, device=device)
            else:
                counts = torch.as_tensor(obstacle_vertex_counts, dtype=torch.long, device=device)

        if counts.ndim != 1 or counts.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"obstacle_vertex_counts must have shape ({vertices.shape[0]},), "
                f"got {tuple(counts.shape)}")
        if bool((counts < 3).any().detach().cpu()) or bool((counts > vertices.shape[1]).any().detach().cpu()):
            raise ValueError("each obstacle must have 3..max_vertices valid vertices")

        edge_ids = torch.arange(vertices.shape[1], dtype=torch.long, device=device)
        edge_mask = edge_ids.unsqueeze(0) < counts.unsqueeze(1)
        next_ids = (edge_ids.unsqueeze(0) + 1) % counts.unsqueeze(1)
        next_vertices = torch.gather(
            vertices,
            dim=1,
            index=next_ids.unsqueeze(-1).expand(-1, -1, NDIM),
        )
        edges = next_vertices - vertices
        edge_lengths = torch.linalg.norm(edges, dim=-1)
        if bool(((edge_lengths <= 1e-12) & edge_mask).any().detach().cpu()):
            raise ValueError("obstacle polygons cannot contain zero-length edges")

        cross = vertices[..., 0] * next_vertices[..., 1] - vertices[..., 1] * next_vertices[..., 0]
        signed_area2 = torch.sum(torch.where(edge_mask, cross, torch.zeros_like(cross)), dim=-1)
        if bool((torch.abs(signed_area2) <= 1e-12).any().detach().cpu()):
            raise ValueError("obstacle polygons must have non-zero signed area")

        # For CCW polygons the right-hand normal is outward; for CW polygons it
        # must be flipped.  Distances are signed along the outward unit normal.
        right_normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1)
        orientation = torch.where(
            signed_area2 >= 0,
            torch.ones_like(signed_area2),
            -torch.ones_like(signed_area2),
        )
        outward_normals = right_normals * orientation[:, None, None]
        outward_normals = outward_normals / torch.clamp(edge_lengths[..., None], min=1e-12)
        outward_normals = torch.where(edge_mask[..., None], outward_normals, torch.zeros_like(outward_normals))
        edge_offsets = torch.sum(vertices * outward_normals, dim=-1)
        edge_offsets = torch.where(edge_mask, edge_offsets, torch.zeros_like(edge_offsets))
        return vertices, outward_normals, edge_offsets, edge_mask, counts

    def positions(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Return current point positions with shape ``(..., agents, 2)``."""

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-3:] != (self.num_agents, NCOFF, NDIM):
            raise ValueError(
                f"coefficients must end with ({self.num_agents}, {NCOFF}, {NDIM}), "
                f"got {tuple(coeff.shape)}")
        return coeff[..., :, 0, :]

    def zero_motion(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Keep current positions and zero all higher-order coefficients."""

        coeff = torch.as_tensor(coefficients)
        frozen = torch.zeros_like(coeff)
        frozen[..., :, 0, :] = coeff[..., :, 0, :]
        return frozen

    def initial_coefficients(
        self,
        positions: torch.Tensor | Iterable[float],
    ) -> torch.Tensor:
        """Create multi-agent coefficients from point positions."""

        pos = torch.as_tensor(
            positions,
            dtype=self.robot_transition.mat_f_stab.dtype,
            device=self.robot_transition.mat_f_stab.device,
        )
        if pos.shape[-2:] != (self.num_agents, NDIM):
            raise ValueError(
                f"positions must end with ({self.num_agents}, {NDIM}), got {tuple(pos.shape)}")
        coeff = torch.zeros(
            (*pos.shape[:-2], self.num_agents, NCOFF, NDIM),
            dtype=pos.dtype,
            device=pos.device,
        )
        coeff[..., :, 0, :] = pos
        return coeff

    def pack_state(
        self,
        coefficients: torch.Tensor,
        active: torch.Tensor | Iterable[float] | None = None,
        time: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """Pack structured state tensors into the canonical flat layout."""

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-3:] != (self.num_agents, NCOFF, NDIM):
            raise ValueError(
                f"coefficients must end with ({self.num_agents}, {NCOFF}, {NDIM}), "
                f"got {tuple(coeff.shape)}")

        batch_shape = coeff.shape[:-3]
        if active is None:
            active_tensor = torch.ones(
                (*batch_shape, self.num_agents),
                dtype=coeff.dtype,
                device=coeff.device,
            )
        else:
            active_tensor = torch.as_tensor(active, dtype=coeff.dtype, device=coeff.device)
            active_tensor = active_tensor.expand((*batch_shape, self.num_agents))

        time_tensor = torch.as_tensor(time, dtype=coeff.dtype, device=coeff.device)
        time_tensor = time_tensor.expand(batch_shape)

        return torch.cat(
            [
                time_tensor.unsqueeze(-1),
                active_tensor,
                coeff.reshape(*batch_shape, self.num_agents * self.coeff_state_dim),
            ],
            dim=-1,
        )

    def unpack_state(self, flat_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack flat states into ``(coefficients, active, time)``."""

        flat = torch.as_tensor(flat_state)
        if flat.shape[-1] != self.state_dim:
            raise ValueError(f"flat_state must end with {self.state_dim}, got {tuple(flat.shape)}")
        time = flat[..., 0]
        active = flat[..., 1:1 + self.num_agents]
        coeff = flat[..., 1 + self.num_agents:].reshape(
            *flat.shape[:-1], self.num_agents, NCOFF, NDIM)
        return coeff, active, time

    def initial_flat_state(
        self,
        positions: torch.Tensor | Iterable[float],
        active: torch.Tensor | Iterable[float] | None = None,
        time: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """Create a packed flat state from point positions."""

        return self.pack_state(self.initial_coefficients(positions), active=active, time=time)

    def _target_from_action(
        self,
        coefficients: torch.Tensor,
        target_or_delta: torch.Tensor,
        *,
        target_is_delta: bool,
    ) -> torch.Tensor:
        target = torch.as_tensor(
            target_or_delta,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        if target.shape[-2:] != (self.num_agents, NDIM):
            raise ValueError(
                f"target positions/actions must end with ({self.num_agents}, {NDIM}), "
                f"got {tuple(target.shape)}")
        if target_is_delta:
            target = self.positions(coefficients) + target
        return target

    def collision_mask(
        self,
        positions: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-agent collision mask under a one-to-one point model."""

        pos = torch.as_tensor(positions)
        active_bool = torch.as_tensor(active, device=pos.device).bool()
        if self.collision_radius <= 0.0:
            return torch.zeros_like(active_bool)

        pair_i_t = self.collision_pair_i.to(device=pos.device)
        pair_j_t = self.collision_pair_j.to(device=pos.device)
        if pair_i_t.numel() == 0:
            return torch.zeros_like(active_bool)

        pos_i = pos.index_select(-2, pair_i_t)
        pos_j = pos.index_select(-2, pair_j_t)
        dist_sq = torch.sum((pos_i - pos_j) * (pos_i - pos_j), dim=-1)
        active_pair = (
            active_bool.index_select(-1, pair_i_t)
            & active_bool.index_select(-1, pair_j_t)
        )
        candidate = active_pair & (dist_sq < self.collision_radius * self.collision_radius)
        sorted_dist, sorted_idx = torch.sort(dist_sq, dim=-1)
        sorted_candidate = torch.gather(candidate, -1, sorted_idx)
        sorted_candidate = sorted_candidate & torch.isfinite(sorted_dist)
        sorted_i = pair_i_t[sorted_idx]
        sorted_j = pair_j_t[sorted_idx]

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
        """Signed distances from points to every obstacle half-plane.

        Shape: ``(..., agents, obstacles, edges)``.  Positive values are outside
        the corresponding convex polygon edge, negative values are inside.
        Invalid padded edges are filled with ``-inf`` so that ``amax`` ignores
        them.
        """

        pos = torch.as_tensor(positions)
        if self.num_obstacles == 0:
            return torch.empty(
                (*pos.shape[:-2], self.num_agents, 0, self.max_obstacle_vertices),
                dtype=pos.dtype,
                device=pos.device,
            )
        normals = self.obstacle_edge_normals.to(device=pos.device, dtype=pos.dtype)
        offsets = self.obstacle_edge_offsets.to(device=pos.device, dtype=pos.dtype)
        edge_mask = self.obstacle_edge_mask.to(device=pos.device)
        signed = torch.einsum("...ad,oed->...aoe", pos, normals) - offsets
        mask_shape = (1,) * (signed.ndim - 2) + edge_mask.shape
        return signed.masked_fill(~edge_mask.reshape(mask_shape), -torch.inf)

    def obstacle_clearance(self, positions: torch.Tensor) -> torch.Tensor:
        """Return max signed half-plane distance per point/obstacle."""

        signed = self.obstacle_signed_edge_distances(positions)
        if signed.shape[-2] == 0:
            return torch.empty(
                (*signed.shape[:-2], 0),
                dtype=signed.dtype,
                device=signed.device,
            )
        return signed.amax(dim=-1)

    def obstacle_collision_mask(
        self,
        positions: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-agent obstacle collision mask for convex polygons."""

        pos = torch.as_tensor(positions)
        active_bool = torch.as_tensor(active, device=pos.device).bool()
        if self.num_obstacles == 0:
            return torch.zeros_like(active_bool)
        clearance = self.obstacle_clearance(pos)
        collides = clearance <= self.obstacle_collision_margin
        return active_bool & collides.any(dim=-1)

    def bounds_validity(
        self,
        positions: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(valid, per-agent out_of_bounds)`` for point positions."""

        pos = torch.as_tensor(positions)
        lower = self.position_lower.to(device=pos.device, dtype=pos.dtype)
        upper = self.position_upper.to(device=pos.device, dtype=pos.dtype)
        in_bounds = torch.all((pos >= lower) & (pos <= upper), dim=-1)
        active_bool = torch.as_tensor(active, device=pos.device).bool()
        if self.bounds_check_active_only:
            out_of_bounds = active_bool & ~in_bounds
        else:
            out_of_bounds = ~in_bounds
        return ~out_of_bounds.any(dim=-1), out_of_bounds

    def team_alive(self, active: torch.Tensor) -> torch.Tensor:
        """Return a boolean ``(..., n_teams)`` tensor indicating live teams."""

        active_bool = torch.as_tensor(active).bool()
        teams = self.team_ids.to(device=active_bool.device)
        values = self.team_values.to(device=active_bool.device)
        alive = [
            (active_bool & (teams == team_value)).any(dim=-1)
            for team_value in values
        ]
        return torch.stack(alive, dim=-1)

    def done(
        self,
        active: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        obstacle_collision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute terminal flags from time, validity, and team activity."""

        done = torch.as_tensor(time) >= self.tf
        if self.done_on_out_of_bounds:
            done = done | ~torch.as_tensor(valid, device=done.device).bool()
        if self.done_on_obstacle_collision and obstacle_collision is not None:
            done = done | torch.as_tensor(obstacle_collision, device=done.device).bool().any(dim=-1)
        if self.done_on_team_eliminated:
            done = done | ~self.team_alive(active).all(dim=-1)
        return done

    def step(
        self,
        coefficients: torch.Tensor,
        active: torch.Tensor,
        time: torch.Tensor | float,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        target_is_delta: bool = False,
    ) -> MincoPointEnvStep:
        """Advance all active points, resolve collisions, time, and done."""

        coeff = torch.as_tensor(coefficients)
        active_tensor = torch.as_tensor(active, device=coeff.device, dtype=coeff.dtype)
        active_bool = active_tensor > 0.5
        time_tensor = torch.as_tensor(time, device=coeff.device, dtype=coeff.dtype)
        time_tensor = time_tensor.expand(coeff.shape[:-3])

        target = self._target_from_action(
            coeff,
            target_pos,
            target_is_delta=target_is_delta,
        )
        stepped, projected_targets = self.robot_transition.step(
            coeff,
            target,
            clamp_target=clamp_target,
            return_projected=True,
        )

        inactive_frozen = self.zero_motion(coeff)
        next_coeff = torch.where(active_bool[..., :, None, None], stepped, inactive_frozen)

        next_time = time_tensor + torch.as_tensor(self.dt, dtype=coeff.dtype, device=coeff.device)
        next_positions = self.positions(next_coeff)
        point_colliding = self.collision_mask(next_positions, active_bool)
        obstacle_colliding = self.obstacle_collision_mask(next_positions, active_bool)
        colliding = point_colliding | obstacle_colliding
        if self.deactivate_on_collision:
            next_active_bool = active_bool & ~point_colliding
        else:
            next_active_bool = active_bool
        if self.deactivate_on_obstacle_collision:
            next_active_bool = next_active_bool & ~obstacle_colliding

        frozen_after_collision = self.zero_motion(next_coeff)
        next_coeff = torch.where(
            next_active_bool[..., :, None, None],
            next_coeff,
            frozen_after_collision,
        )
        valid, out_of_bounds = self.bounds_validity(
            self.positions(next_coeff),
            next_active_bool,
        )
        done = self.done(next_active_bool, next_time, valid, obstacle_colliding)

        return MincoPointEnvStep(
            coefficients=next_coeff,
            active=next_active_bool.to(dtype=coeff.dtype),
            time=next_time,
            done=done,
            valid=valid,
            collision_mask=colliding,
            point_collision_mask=point_colliding,
            obstacle_collision_mask=obstacle_colliding,
            out_of_bounds_mask=out_of_bounds,
            projected_targets=projected_targets,
        )

    def step_flat(
        self,
        flat_state: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        target_is_delta: bool = False,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, MincoPointEnvStep]:
        """Flat-state environment step.

        Returns ``(next_flat, done)`` by default.  With ``return_info=True`` it
        returns ``(next_flat, done, structured_step)``.
        """

        coeff, active, time = self.unpack_state(flat_state)
        step = self.step(
            coeff,
            active,
            time,
            target_pos,
            clamp_target=clamp_target,
            target_is_delta=target_is_delta,
        )
        next_flat = self.pack_state(step.coefficients, step.active, step.time)
        if return_info:
            return next_flat, step.done, step
        return next_flat, step.done

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, MincoPointEnvStep]:
        """Gather parent states from a tree pool and perform ``step_flat``."""

        pool = torch.as_tensor(flat_state_pool)
        if pool.ndim != 3 or pool.shape[-1] != self.state_dim:
            raise ValueError(
                f"flat_state_pool must have shape (trees, nodes, {self.state_dim}), "
                f"got {tuple(pool.shape)}")
        trees = torch.as_tensor(tree_ids, device=pool.device, dtype=torch.long)
        nodes = torch.as_tensor(node_ids, device=pool.device, dtype=torch.long)
        parent_flat = pool[trees, nodes]
        return self.step_flat(
            parent_flat,
            target_pos,
            clamp_target=clamp_target,
            target_is_delta=target_is_delta,
            return_info=return_info,
        )
