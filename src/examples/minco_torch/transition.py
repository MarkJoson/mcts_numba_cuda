"""Batched MINCO coefficient-state transition operators."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

from .constants import DEFAULT_RATIO, NDIM, NCOFF, S
from .math import (
    construct_bb_t_int,
    construct_beta_t,
    construct_mat_r,
    construct_minco_m2,
    solve_discrete_lqr_gain,
)

class MincoTorchTransition(torch.nn.Module):
    """向量化的 MINCO 轨迹闭环控制器.

    Shape 约定：
        B: 任意 batch 前缀，可以是 ``()``, ``(batch,)``, ``(trees, nodes)``
           或 rollout/expand 中的其它前缀。
        coefficients: ``(*B, NCOFF=6, NDIM=2)``，最后两维是 6 阶多项式系数和 x/y。
        flat_state: ``(*B, 12)``，即 ``coefficients`` 展平后的树节点状态。
        target_pos: ``(*B, 2)`` 或可 broadcast 到 ``(*B, 2)`` 的目标点。
        target_sequence: ``(horizon, *B, 2)``。
        C: ``2 * n_checkpoints``，速度和加速度约束行数。
        E: 预构建 evaluation time 数量。

    性能约定：
        ``coefficients``/``flat_state`` 应该在初始化 rollout/expand state pool
        时就放到和 module buffer 一样的 dtype/device。热路径中不会隐式
        ``buffer.to(...)``；dtype/device 不一致会直接报错，避免每步隐藏拷贝。
        热路径使用 ``view`` 而不是 ``reshape``，因此 state pool 应保持 contiguous；
        非连续切片需要在进入 rollout/expand 前显式处理。
        ``evaluate`` 只使用初始化时预构建的 ``evaluation_basis``；需要额外
        采样时间时，通过 ``evaluation_times`` 参数在初始化阶段传入。

    注册的 MINCO/LQR buffer shape：
        mat_f: ``(6, 6)``，原始闭环前状态转移矩阵。
        mat_g: ``(6, 1)``，原始目标输入矩阵。
        lqr_k: ``(1, 6)``，LQR feedback gain。
        kpp: ``(1, 6)``，目标前馈 gain。
        mat_f_stab: ``(6, 6)``，稳定化后的状态转移矩阵。
        mat_g_stab: ``(6, 1)``，稳定化后的目标输入矩阵。
        constraint_const: ``(C, 6)``，每个 checkpoint 的状态约束系数。
        constraint_gain: ``(C,)``，每个 checkpoint 的目标约束系数。
        constraint_signed_limit: ``(C,)``，带符号速度/加速度限幅。
        constraint_const_scaled: ``(C, 6)``，除以 gain 后用于 target bound。
        constraint_lower_bias: ``(C,)``，每行 target 下界 bias。
        constraint_upper_bias: ``(C,)``，每行 target 上界 bias。
        checkpoint_times: ``(C/2,)``，障碍投影/未来位置检查时间点。
        checkpoint_position_const: ``(C/2, 6)``，未来位置的状态线性项。
        checkpoint_position_gain: ``(C/2,)``，未来位置的 target 标量系数。
        evaluation_times: ``(E,)``，预构建 beta 的时间点。
        evaluation_basis: ``(E, 6, 6)``，``[time_id, rank, coeff_id]``。
    """

    def __init__(
        self,
        piece_t: float,                             # 每段真实执行时间，scalar
        *,
        ratio: float = DEFAULT_RATIO,               # 真实执行时间 / planner 前瞻时间比例
        n_checkpoints: int = 8,                     # 每种约束的 checkpoint 数；C=2*n_checkpoints
        velocity_limit: float = 2.0,                # 速度限幅
        acceleration_limit: float = 3.0,            # 加速度限幅
        dtype: torch.dtype | None = torch.float32,  # runtime buffer dtype
        device: torch.device | str | None = None,
        lqr_max_iter: int = 10_000,                 # lqr控制器求解迭代次数
        lqr_tol: float = 1e-12,                     # lqr控制器求解精度
        evaluation_times: Sequence[float] | torch.Tensor | None = None,  # 预构建 evaluate beta 的 t
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

        runtime_dtype = dtype or torch.float32
        build_dtype = torch.float64
        build_device = torch.device(device) if device is not None else None

        # One-time MINCO algebra, all in float64 for stable matrix inversion.
        # mat_m/mat_m_inv/mat_r/mat_s: (6, 6); mat_u: (6, 1).
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

        mat_f = mat_m_inv @ mat_s @ mat_r  # (6, 6)
        mat_g = mat_m_inv @ mat_u          # (6, 1)

        # LQR solve: q_mat (6, 6), r_mat (1, 1), lqr_k (1, 6).
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
        ) + lqr_k                                      # (1, 6)
        mat_f_stab = mat_f - mat_g @ lqr_k             # (6, 6)
        mat_g_stab = mat_g @ kpp @ torch.tensor(
            [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            dtype=build_dtype,
            device=build_device,
        )                                               # (6, 1)

        constraint_const, constraint_gain, constraint_limit = (
            self._build_constraint_tensors(
                mat_f_stab,
                mat_g_stab,
                dtype=build_dtype,
                device=build_device,
            )
        )
        # Constraint tensors use C = 2 * n_checkpoints rows:
        # rows 0..n_checkpoints-1 are velocity, remaining rows are acceleration.
        # constraint_const: (C, 6), constraint_gain/limit: (C,).
        signed_limit = torch.where(
            constraint_gain < 0.0,
            -constraint_limit,
            constraint_limit,
        )
        inv_gain = torch.reciprocal(constraint_gain)
        constraint_const_scaled = constraint_const * inv_gain.view(-1, 1)
        constraint_lower_bias = -signed_limit * inv_gain
        constraint_upper_bias = signed_limit * inv_gain
        checkpoint_times, checkpoint_position_const, checkpoint_position_gain = (
            self._build_position_checkpoint_tensors(
                mat_f_stab,
                mat_g_stab,
                dtype=build_dtype,
                device=build_device,
            )
        )
        eval_times, eval_basis, eval_time_values = self._build_evaluation_tensors(
            evaluation_times,
            dtype=build_dtype,
            device=build_device,
        )

        self.register_buffer("mat_f", mat_f.to(runtime_dtype))                              # (6, 6)
        self.register_buffer("mat_g", mat_g.to(runtime_dtype))                              # (6, 1)
        self.register_buffer("lqr_k", lqr_k.to(runtime_dtype))                              # (1, 6)
        self.register_buffer("kpp", kpp.to(runtime_dtype))                                  # (1, 6)
        self.register_buffer("mat_f_stab", mat_f_stab.to(runtime_dtype))                    # (6, 6)
        self.register_buffer("mat_g_stab", mat_g_stab.to(runtime_dtype))                    # (6, 1)
        self.register_buffer("constraint_const", constraint_const.to(runtime_dtype))        # (C, 6)
        self.register_buffer("constraint_gain", constraint_gain.to(runtime_dtype))          # (C,)
        self.register_buffer("constraint_signed_limit", signed_limit.to(runtime_dtype))     # (C,)
        self.register_buffer("constraint_const_scaled", constraint_const_scaled.to(runtime_dtype))  # (C, 6)
        self.register_buffer("constraint_lower_bias", constraint_lower_bias.to(runtime_dtype))      # (C,)
        self.register_buffer("constraint_upper_bias", constraint_upper_bias.to(runtime_dtype))      # (C,)
        self.register_buffer("checkpoint_times", checkpoint_times.to(runtime_dtype))                # (C/2,)
        self.register_buffer("checkpoint_position_const", checkpoint_position_const.to(runtime_dtype))  # (C/2, 6)
        self.register_buffer("checkpoint_position_gain", checkpoint_position_gain.to(runtime_dtype))    # (C/2,)
        self.register_buffer("evaluation_times", eval_times.to(runtime_dtype))              # (E,)
        self.register_buffer("evaluation_basis", eval_basis.to(runtime_dtype))              # (E, 6, 6)
        self._evaluation_time_values = eval_time_values
        self._compiled_step_flat = None

        self.mat_f:torch.Tensor
        self.mat_g:torch.Tensor
        self.lqr_k:torch.Tensor
        self.kpp:torch.Tensor
        self.mat_f_stab:torch.Tensor
        self.mat_g_stab:torch.Tensor
        self.constraint_const:torch.Tensor
        self.constraint_gain:torch.Tensor
        self.constraint_signed_limit:torch.Tensor
        self.constraint_const_scaled:torch.Tensor
        self.constraint_lower_bias:torch.Tensor
        self.constraint_upper_bias:torch.Tensor
        self.checkpoint_times:torch.Tensor
        self.checkpoint_position_const:torch.Tensor
        self.checkpoint_position_gain:torch.Tensor
        self.evaluation_times:torch.Tensor
        self.evaluation_basis:torch.Tensor
        self._compiled_step_flat:object | None

    def _build_constraint_tensors(
        self,
        mat_f_stab: torch.Tensor,
        mat_g_stab: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build velocity/acceleration target-bound rows.

        Inputs:
            mat_f_stab: ``(6, 6)`` stable state transition.
            mat_g_stab: ``(6, 1)`` stable target input matrix.
        Outputs:
            constraint_const: ``(C, 6)`` state coefficients.
            constraint_gain: ``(C,)`` target coefficients.
            constraint_limit: ``(C,)`` positive limit for each row.
        """
        const_rows = []
        gain_vals = []
        limits = []
        for rank, limit in ((1, self.velocity_limit), (2, self.acceleration_limit)):
            for i in range(self.n_checkpoints):
                t = (i + 1) / (self.n_checkpoints + 1) * self.planner_piece_t
                beta_t = construct_beta_t(t, rank, dtype=dtype, device=device).T
                const_rows.append((beta_t @ mat_f_stab).view(NCOFF))
                gain_vals.append((beta_t @ mat_g_stab).view(()))
                limits.append(float(limit))

        return (
            torch.stack(const_rows, dim=0),
            torch.stack(gain_vals, dim=0),
            torch.tensor(limits, dtype=dtype, device=device),
        )

    def _build_position_checkpoint_tensors(
        self,
        mat_f_stab: torch.Tensor,
        mat_g_stab: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build future-position affine rows for obstacle target projection.

        Future checkpoint position after applying a target is:
        ``p(t) = checkpoint_position_const[t] @ coeff + gain[t] * target``.

        Outputs:
            checkpoint_times: ``(P,)``.
            checkpoint_position_const: ``(P, 6)``.
            checkpoint_position_gain: ``(P,)``.
        """
        times = torch.tensor(
            [
                (i + 1) / (self.n_checkpoints + 1) * self.planner_piece_t
                for i in range(self.n_checkpoints)
            ],
            dtype=dtype,
            device=device,
        )
        const_rows = []
        gain_vals = []
        for t in times:
            beta_t = construct_beta_t(float(t.detach().cpu()), 0, dtype=dtype, device=device).T
            const_rows.append((beta_t @ mat_f_stab).view(NCOFF))
            gain_vals.append((beta_t @ mat_g_stab).view(()))
        return times, torch.stack(const_rows, dim=0), torch.stack(gain_vals, dim=0)

    def _build_evaluation_tensors(
        self,
        evaluation_times: Sequence[float] | torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...]]:
        """Prebuild beta rows for ``evaluate``.

        Inputs:
            evaluation_times: optional sequence of scalar times.
        Outputs:
            eval_times: ``(E,)`` sorted unique times.
            eval_basis: ``(E, 6, 6)``, indexed by ``time_id, rank, coeff_id``.
            eval_time_values: Python tuple used for exact/near lookup.
        """
        if evaluation_times is None:
            times_tensor = torch.tensor([0.0, self.piece_t], dtype=dtype, device=device)
        else:
            times_tensor = torch.as_tensor(evaluation_times, dtype=dtype, device=device).view(-1)
            if times_tensor.numel() == 0:
                raise ValueError("evaluation_times cannot be empty")
            times_tensor = torch.cat(
                [torch.tensor([0.0, self.piece_t], dtype=dtype, device=device), times_tensor],
                dim=0,
            )

        times_tensor = torch.unique(times_tensor.detach(), sorted=True)
        if bool((times_tensor < 0.0).any().detach().cpu()) or bool(
            (times_tensor > self.planner_piece_t).any().detach().cpu()
        ):
            raise ValueError(
                "evaluation_times must lie within [0, planner_piece_t]; "
                f"got {times_tensor.detach().cpu().tolist()}"
            )

        basis = torch.stack(
            [
                torch.stack(
                    [
                        construct_beta_t(float(t.detach().cpu()), rank, dtype=dtype, device=device).view(NCOFF)
                        for rank in range(NCOFF)
                    ],
                    dim=0,
                )
                for t in times_tensor
            ],
            dim=0,
        )
        time_values = tuple(float(t) for t in times_tensor.detach().cpu().tolist())
        return times_tensor, basis, time_values

    @property
    def state_dim(self) -> int:
        return NCOFF * NDIM

    def initial_coefficients(
        self,
        init_pos: torch.Tensor | Iterable[float],
        *,
        batch_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """给定初始位置，生成状态变量。

        Inputs:
            init_pos: ``(*B, 2)`` or ``(2,)``.
            batch_shape: optional explicit ``B`` when ``init_pos`` is ``(2,)``.
        Output:
            coefficients: ``(*B, 6, 2)``.
        """

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
        """Return fixed runtime buffers without per-call conversion.

        Input:
            reference: tensor that must already match module dtype/device.
        Outputs:
            mat_f_stab: ``(6, 6)``.
            mat_g_stab: ``(6, 1)``.
            constraint_const_scaled: ``(C, 6)``.
            constraint_lower_bias: ``(C,)``.
            constraint_upper_bias: ``(C,)``.
        """
        expected_device = self.mat_f_stab.device
        expected_dtype = self.mat_f_stab.dtype
        if reference.device != expected_device or reference.dtype != expected_dtype:
            raise ValueError(
                "coefficients must use the same dtype/device as MincoTorchTransition "
                f"buffers; got dtype={reference.dtype}, device={reference.device}, "
                f"expected dtype={expected_dtype}, device={expected_device}. "
                "Move/cast the rollout state pool during initialization instead "
                "of relying on per-step buffer conversion."
            )
        return (
            self.mat_f_stab,
            self.mat_g_stab,
            self.constraint_const_scaled,
            self.constraint_lower_bias,
            self.constraint_upper_bias,
        )

    def runtime_buffers(self, reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Public alias for hot-path runtime buffer validation."""

        return self._buffers_for(reference)

    def bounds(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target lower/upper bounds.

        Input:
            coefficients: ``(*B, 6, 2)``.
        Internal:
            const_val: ``(*B, C, 2)`` from ``(C, 6) x (*B, 6, 2)``.
            lower_each/upper_each: ``(*B, C, 2)`` row-wise target bounds.
        Outputs:
            lower: ``(*B, 2)``.
            upper: ``(*B, 2)``.
        """

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-2:] != (NCOFF, NDIM):
            raise ValueError(
                f"coefficients must have trailing shape {(NCOFF, NDIM)}, "
                f"got {tuple(coeff.shape)}")

        _, _, const_coeff, lower_bias, upper_bias = self._buffers_for(coeff)
        const_val = torch.einsum("ci,...id->...cd", const_coeff, coeff)

        view_shape = (1,) * (const_val.ndim - 2) + (-1, 1)
        lower_bias = lower_bias.view(*view_shape)
        upper_bias = upper_bias.view(*view_shape)

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
        """Clip target positions to precomputed velocity/acceleration bounds.

        Inputs:
            coefficients: ``(*B, 6, 2)``.
            target_pos: ``(*B, 2)`` or ``(*B, 1, 2)``; must broadcast with ``B``.
        Output:
            projected_target: ``(*B, 2)``.
        """

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
        """Apply ``F_stab @ coeff + G_stab @ target`` in batch.

        Inputs:
            coefficients: ``(*B, 6, 2)``.
            target_pos: ``(*B, 2)`` or ``(*B, 1, 2)``.
        Internal:
            linear: ``(*B, 6, 2)`` from ``(6, 6) x (*B, 6, 2)``.
            forced: ``(*B, 6, 2)`` from ``(6, 1) * (*B, 1, 2)``.
        Output:
            next_coefficients: ``(*B, 6, 2)``.
        """

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
        forced = mat_g.view(NCOFF, 1) * target.unsqueeze(-2)
        return (linear + forced).contiguous()

    def step(
        self,
        coefficients: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_projected: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project the target if requested, then advance one MINCO state.

        Inputs:
            coefficients: ``(*B, 6, 2)``.
            target_pos: ``(*B, 2)`` or ``(*B, 1, 2)``.
        Outputs:
            next_coefficients: ``(*B, 6, 2)``.
            If ``return_projected=True`` also returns projected target ``(*B, 2)``.
        """

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

        Inputs:
            coefficients: ``(*B, 6, 2)``.
            target_sequence: ``(horizon, *B, 2)``.  The batch dimensions after
                ``horizon`` are broadcast against ``coefficients``.
        Output:
            If ``return_history=False``: final coefficients ``(*B, 6, 2)``.
            If ``return_history=True``: history ``(horizon + 1, *B, 6, 2)``.
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
        """Evaluate position/velocity/acceleration-like derivatives in batch.

        Inputs:
            coefficients: ``(*B, 6, 2)``.
            t: scalar time pre-registered in ``evaluation_times``.
            rank: derivative order; ``0`` gives position, ``1`` velocity, etc.
        Output:
            value: ``(*B, 2)``.
        """

        coeff = torch.as_tensor(coefficients)
        self._buffers_for(coeff)
        rank_int = int(rank)
        if not 0 <= rank_int < NCOFF:
            raise ValueError(f"rank must be in [0, {NCOFF}), got {rank}")

        t_float = float(t)
        time_idx = None
        time_tol = (
            1e-6
            if self.evaluation_times.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else 1e-12
        )
        for idx, value in enumerate(self._evaluation_time_values):
            if math.isclose(t_float, value, rel_tol=0.0, abs_tol=time_tol):
                time_idx = idx
                break
        if time_idx is None:
            raise ValueError(
                f"t={t_float} is not prebuilt for evaluate(); initialize "
                f"MincoTorchTransition with evaluation_times including this value. "
                f"Available times: {self._evaluation_time_values}"
            )

        beta = self.evaluation_basis[time_idx, rank_int]
        return torch.einsum("i,...id->...d", beta, coeff)

    def flatten_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Convert ``(*B, 6, 2)`` coefficient states to ``(*B, 12)`` vectors."""

        coeff = torch.as_tensor(coefficients)
        if coeff.shape[-2:] != (NCOFF, NDIM):
            raise ValueError(
                f"coefficients must have trailing shape {(NCOFF, NDIM)}, "
                f"got {tuple(coeff.shape)}")
        return coeff.view(*coeff.shape[:-2], NCOFF * NDIM)

    def unflatten_coefficients(self, flat_state: torch.Tensor) -> torch.Tensor:
        """Convert ``(*B, 12)`` vectors to ``(*B, 6, 2)`` coefficient states."""

        flat = torch.as_tensor(flat_state)
        if flat.shape[-1] != NCOFF * NDIM:
            raise ValueError(
                f"flat_state must end with dimension {NCOFF * NDIM}, got {flat.shape}")
        return flat.view(*flat.shape[:-1], NCOFF, NDIM)

    def step_flat(
        self,
        flat_state: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_projected: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Flat-state version of :meth:`step`, useful for MCTS tree buffers.

        Inputs:
            flat_state: ``(*B, 12)``.
            target_pos: ``(*B, 2)`` or ``(*B, 1, 2)``.
        Outputs:
            next_flat_state: ``(*B, 12)``.
            If ``return_projected=True`` also returns projected target ``(*B, 2)``.
        """

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

    def _step_flat_tensor(
        self,
        flat_state: torch.Tensor,
        target_pos: torch.Tensor,
        clamp_target: bool,
    ) -> torch.Tensor:
        """Tensor-only flat step used by ``torch.compile`` wrappers."""

        coeff = flat_state.view(*flat_state.shape[:-1], NCOFF, NDIM)
        target = target_pos
        if clamp_target:
            if target.shape[-2:] == (1, NDIM):
                target = target.squeeze(-2)
            const_val = torch.matmul(self.constraint_const_scaled, coeff)
            lower = (self.constraint_lower_bias.view(*((1,) * (const_val.ndim - 2)), -1, 1) - const_val).amax(dim=-2)
            upper = (self.constraint_upper_bias.view(*((1,) * (const_val.ndim - 2)), -1, 1) - const_val).amin(dim=-2)
            target = torch.minimum(torch.maximum(target, lower), upper)
        next_coeff = torch.matmul(self.mat_f_stab, coeff) + self.mat_g_stab.view(NCOFF, 1) * target.unsqueeze(-2)
        return next_coeff.contiguous().view(*flat_state.shape[:-1], NCOFF * NDIM)

    def compile_step_flat(
        self,
        *,
        mode: str | None = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool | None = False,
    ) -> None:
        """Compile the no-info flat MINCO step for fixed-shape rollout loops.

        This compiles the tensor-only robot transition path.  It is most useful
        when MCTS repeatedly expands the same batch shape.  The public
        ``step_flat`` API remains available as a fallback and for
        ``return_projected=True``.
        """

        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        self._compiled_step_flat = torch.compile(
            self._step_flat_tensor,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

    def step_flat_compiled(
        self,
        flat_state: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        clamp_target: bool = True,
    ) -> torch.Tensor:
        """Run the compiled flat step, compiling lazily on first use."""

        if self._compiled_step_flat is None:
            self.compile_step_flat()
        return self._compiled_step_flat(flat_state, target_pos, clamp_target)

    def rollout_flat(
        self,
        flat_state: torch.Tensor,
        target_sequence: torch.Tensor,
        *,
        clamp_target: bool = True,
        return_history: bool = False,
    ) -> torch.Tensor:
        """Flat-state version of :meth:`rollout`.

        Inputs:
            flat_state: ``(*B, 12)``.
            target_sequence: ``(horizon, *B, 2)``.
        Output:
            If ``return_history=False``: final flat state ``(*B, 12)``.
            If ``return_history=True``: history ``(horizon + 1, *B, 12)``.
        """

        coeff = self.unflatten_coefficients(flat_state)
        result = self.rollout(
            coeff,
            target_sequence,
            clamp_target=clamp_target,
            return_history=return_history,
        )
        if return_history:
            return result.view(*result.shape[:-2], NCOFF * NDIM)
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
        """Gather parent states from a tree pool and step them.

        Inputs:
            flat_state_pool: ``(num_trees, num_nodes, 12)``.
            tree_ids: ``(*B,)`` long indices.
            node_ids: ``(*B,)`` long indices.
            target_pos: ``(*B, 2)``.
        Outputs:
            next_flat_state: ``(*B, 12)``.
            If ``return_projected=True`` also returns projected target ``(*B, 2)``.

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
