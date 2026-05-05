"""动作投影与未来障碍物检查（类封装）。"""

from __future__ import annotations

from collections import defaultdict
import torch
import torch.nn.functional as F

from .collision_ops import MincoCollisionOps
from .constants import NDIM
from .scene import MincoScene
from .transition import MincoTorchTransition


class MincoProjectionOps:
    """目标动作约束投影算子（仅处理目标点，不管理环境状态）。

    关键职责：
    1. 基于 MINCO 线性形式，将目标点 ``target`` 映射到未来检查点位置；
    2. 评估未来检查点到凸障碍物半空间约束的余量；
    3. 对目标点做近似投影修正，降低未来时域内碰撞违规。

    约定形状（省略可选前缀 batch 维，用 ``...`` 表示）：
    - 目标点：``target[..., A, 2]``，其中 ``A=num_agents``。
    - MINCO 系数：``coeff[..., A, C, 2]``，其中 ``C`` 为每轴多项式系数数目。
    - 未来检查点位置：``future_pos[..., A, P, 2]``，其中 ``P=num_checkpoints``。
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
        """初始化投影算子与投影超参数。

        Args:
            scene: 场景对象，提供障碍物与 agent 全局配置。
            robot_transition: MINCO 动力学映射与边界投影算子。
            collision_ops: 障碍物边/半空间距离计算算子。
            obstacle_collision_margin: 碰撞安全余量（基础 margin）。
            obstacle_projection_iters: 单次动作投影迭代次数。
            obstacle_projection_extra_margin: 投影时附加 margin。
            obstacle_projection_topk: 每轮优先挑选的违规候选上限。
            obstacle_projection_fixes_per_iter: 每轮实际修正候选数量。
            obstacle_projection_check_active_only: 是否仅对 active agent 执行约束检测。
        """
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
        # 性能分析开关：默认关闭，避免常规运行引入额外开销。
        self.prof_nvtx = False
        # 迭代修正策略：
        # - "filter": 旧策略，层层筛选（top-k + fix 子集）
        # - "jacobi": Jacobi 并行批量修正（不过滤 fix 子集，统一乘修正系数）
        self.iter_update_mode = "filter"
        self.jacobi_relax = 0.5
        # select 有效/无效比例统计开关（默认关闭）与计数器。
        self.prof_select_stats = False
        self._sel_num: dict[str, int] = defaultdict(int)
        self._sel_den: dict[str, int] = defaultdict(int)
        # torch.compile：默认始终启用（懒编译一次，后续复用）。
        self._compiled_project_target_away = None

    @property
    def num_obstacles(self) -> int:
        """障碍物数量。"""
        return int(self.scene.num_obstacles)

    def set_prof_nvtx(self, enabled: bool) -> None:
        """设置 NVTX 细粒度打点开关（仅用于性能分析）。"""
        self.prof_nvtx = bool(enabled)

    def set_iter_update_mode(self, mode: str) -> None:
        """设置迭代修正策略。"""
        m = str(mode).strip().lower()
        if m not in ("filter", "jacobi"):
            raise ValueError(f"unsupported iter_update_mode: {mode}")
        self.iter_update_mode = m
        # 迭代策略变化后清理已编译图，后续按新策略懒编译。
        self._compiled_project_target_away = None

    def set_jacobi_relax(self, alpha: float) -> None:
        """设置 Jacobi 批量修正系数。"""
        a = float(alpha)
        if a <= 0:
            raise ValueError(f"jacobi_relax must be positive, got {alpha}")
        self.jacobi_relax = a
        # 修正系数变化后清理已编译图，后续按新系数懒编译。
        self._compiled_project_target_away = None

    def _ensure_compiled_project_target_away(self) -> None:
        """懒编译障碍投影 fast-path（始终开启）。"""
        if self._compiled_project_target_away is not None:
            return
        if not hasattr(torch, "compile"):
            return

        def _compiled_fn(
            coefficients: torch.Tensor,
            target_pos: torch.Tensor,
            active: torch.Tensor,
        ) -> torch.Tensor:
            projected, _, _ = self._project_target_away_from_obstacles_eager(
                coefficients,
                target_pos,
                active,
                clamp_to_dynamic_bounds=True,
                iterations=None,
                return_residual=False,
            )
            return projected

        self._compiled_project_target_away = torch.compile(
            _compiled_fn,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )

    def set_prof_select_stats(self, enabled: bool, *, reset: bool = True) -> None:
        """设置 select 统计开关。"""
        self.prof_select_stats = bool(enabled)
        if reset:
            self.reset_prof_select_stats()

    def reset_prof_select_stats(self) -> None:
        """重置 select 统计计数器。"""
        self._sel_num.clear()
        self._sel_den.clear()

    def get_prof_select_stats(self) -> dict[str, dict[str, float]]:
        """获取 select 统计结果。

        Returns:
            ``{name: {"valid": int, "total": int, "valid_ratio": float, "invalid_ratio": float}}``
        """
        out: dict[str, dict[str, float]] = {}
        keys = set(self._sel_den.keys()) | set(self._sel_num.keys())
        for name in sorted(keys):
            total = int(self._sel_den.get(name, 0))
            valid = int(self._sel_num.get(name, 0))
            ratio = float(valid / total) if total > 0 else 0.0
            out[name] = {
                "valid": float(valid),
                "total": float(total),
                "valid_ratio": ratio,
                "invalid_ratio": 1.0 - ratio if total > 0 else 0.0,
            }
        return out

    def _sel_count(self, name: str, valid_mask: torch.Tensor, total_mask: torch.Tensor | None = None) -> None:
        """累计某个 select 点的有效/总数。"""
        if not self.prof_select_stats:
            return
        vm = torch.as_tensor(valid_mask).bool()
        if total_mask is None:
            tm = torch.ones_like(vm, dtype=torch.bool)
        else:
            tm = torch.as_tensor(total_mask, device=vm.device).bool()
        valid = (vm & tm).sum().detach().cpu().item()
        total = tm.sum().detach().cpu().item()
        self._sel_num[name] += int(valid)
        self._sel_den[name] += int(total)

    def future_positions_from_target(self, coefficients: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """由目标点生成未来检查点位置（MINCO 线性映射）。

        关键路径：
        1. 预取/缓存 runtime buffer（检查点常数项与增益）；
        2. 计算常数项贡献 ``base``；
        3. 计算目标项贡献 ``gain * target`` 并相加。

        Args:
            coefficients: MINCO 系数，形状 ``[..., A, C, 2]``。
            target_pos: 目标点，形状 ``[..., A, 2]``。

        Returns:
            未来检查点位置，形状 ``[..., A, P, 2]``。
        """
        coeff = torch.as_tensor(coefficients)
        # 关键路径：确保 runtime 缓冲可直接复用，避免重复构造张量。
        self.robot_transition.runtime_buffers(coeff)
        target = torch.as_tensor(target_pos, dtype=coeff.dtype, device=coeff.device)
        const = self.robot_transition.checkpoint_position_const
        gain = self.robot_transition.checkpoint_position_gain
        # base: [..., A, P, 2]
        base = torch.einsum("pc,...acd->...apd", const, coeff)
        # gain.view(..., P, 1) 与 target.unsqueeze(-2): [..., A, 1, 2] 广播后得到 [..., A, P, 2]
        return base + gain.view(*((1,) * (base.ndim - 2)), -1, 1) * target.unsqueeze(-2)

    def future_obstacle_clearance_from_target(self, coefficients: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """计算未来检查点对各障碍物的半空间余量（越小越危险）。

        关键路径：
        1. 目标点 -> 未来检查点位置；
        2. 计算点到所有障碍物所有边的有符号距离；
        3. 对“边”维做 ``amax``，得到每个障碍物的约束余量。

        Args:
            coefficients: MINCO 系数，形状 ``[..., A, C, 2]``。
            target_pos: 目标点，形状 ``[..., A, 2]``。

        Returns:
            clearance，形状 ``[..., A, P, O]``，``O=num_obstacles``。
        """
        future_pos = self.future_positions_from_target(coefficients, target_pos)
        if self.num_obstacles == 0:
            return torch.empty(
                (*future_pos.shape[:-2], future_pos.shape[-2], 0),
                dtype=future_pos.dtype,
                device=future_pos.device,
            )
        # signed: [..., A, P, O, Emax]
        signed = self.collision_ops.obstacle_signed_edge_distances_any_agents(future_pos)
        # 对每个 obstacle 的所有边取最大值 -> [..., A, P, O]
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
        """将目标点投影到“未来检查点障碍物约束”近可行域。

        说明：
        - 默认走 eager 实现；
        - 当满足 fixed-shape fast-path 条件时，可自动走 compiled 分支：
          ``clamp_to_dynamic_bounds=True``、``iterations=None``、
          ``return_residual=False``。
        """
        can_use_compiled = (
            (not self.prof_nvtx)
            and (not self.prof_select_stats)
            and clamp_to_dynamic_bounds
            and (iterations is None)
            and (not return_residual)
        )
        if can_use_compiled:
            self._ensure_compiled_project_target_away()
        if can_use_compiled and self._compiled_project_target_away is not None:
            coeff = torch.as_tensor(coefficients)
            target = torch.as_tensor(target_pos, dtype=coeff.dtype, device=coeff.device)
            batch_shape = target.shape[:-2]
            if active is None:
                active_t = torch.ones((*batch_shape, self.num_agents), dtype=torch.bool, device=target.device)
            else:
                active_t = torch.as_tensor(active, device=target.device).bool().view(*batch_shape, self.num_agents)
            projected = self._compiled_project_target_away(coeff, target, active_t)
            iters_used = torch.tensor(
                self.obstacle_projection_iters if iterations is None else int(iterations),
                dtype=torch.long,
                device=target.device,
            )
            residual = torch.zeros(target.shape[:-1], dtype=torch.bool, device=target.device)
            if active is not None and self.obstacle_projection_check_active_only:
                residual = residual & active_t
            return projected, iters_used, residual

        return self._project_target_away_from_obstacles_eager(
            coefficients,
            target_pos,
            active,
            clamp_to_dynamic_bounds=clamp_to_dynamic_bounds,
            iterations=iterations,
            return_residual=return_residual,
        )

    def _project_target_away_from_obstacles_eager(
        self,
        coefficients: torch.Tensor,
        target_pos: torch.Tensor,
        active: torch.Tensor | None = None,
        *,
        clamp_to_dynamic_bounds: bool = True,
        iterations: int | None = None,
        return_residual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将目标点投影到“未来检查点障碍物约束”近可行域。

        核心思想（高效近似）：
        1. 在当前目标点处，先评估未来 ``A x P`` 检查点对所有障碍物的违规量；
        2. 扁平化后取 top-k 最大违规候选，构造线性不等式近似；
        3. 每轮只修正前 ``fixes_per_iter`` 个违规（稀疏修正），迭代 ``iters`` 轮。

        关键张量形状：
        - target / projected: ``[..., A, 2]``
        - candidate_violation: ``[..., A, P, O]``
        - flat_violation: ``[..., A, P*O]``
        - selected_normals / selected_offsets: ``[..., A, K, 2]`` / ``[..., A, K]``
        - a_vec / rhs / denom: ``[..., A, K, 2]`` / ``[..., A, K]`` / ``[..., A, K]``

        Args:
            coefficients: MINCO 系数，形状 ``[..., A, C, 2]``。
            target_pos: 原始目标点，形状 ``[..., A, 2]``。
            active: 可选 active mask，形状 ``[..., A]``。
            clamp_to_dynamic_bounds: 每轮修正后是否执行动力学边界裁剪。
            iterations: 覆盖默认迭代次数；``None`` 时使用配置值。
            return_residual: 是否返回最终残余违规标记。

        Returns:
            projected: 投影后目标点，形状 ``[..., A, 2]``。
            iters_used: 实际迭代次数（标量 long tensor）。
            residual: 是否仍有障碍物违规，形状 ``[..., A]``。

        性能注记（2026-05-05 实测，RTX 4090，b=65536, ckpt=8, obs=16）：
        - 外层 ``projection_iter`` 平均约 17.35 ms/iter。
        - 障碍投影本体（调用该函数）在外层中的占比约 99.25%；
          外层初始动态投影仅约 0.128 ms/iter（~0.74%）。
        - 本函数内部最重路径是 ``iter_refine``，其中
          ``iter.clamp_target`` 与 ``iter.select_fix`` 是 refine 内主耗时项。
        """
        coeff = torch.as_tensor(coefficients)
        target = torch.as_tensor(target_pos, dtype=coeff.dtype, device=coeff.device)
        nvtx_on = bool(self.prof_nvtx and target.is_cuda)
        if nvtx_on:
            def _push(name: str) -> None:
                torch.cuda.nvtx.range_push(name)
            def _pop() -> None:
                torch.cuda.nvtx.range_pop()
        else:
            def _push(name: str) -> None:
                return None
            def _pop() -> None:
                return None
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

        _push("proj.prepare_buffers")
        # 障碍几何缓冲：normals/offsets/edge_mask 形状分别为 [O, Emax, 2] / [O, Emax] / [O, Emax]
        # [Profile: projection_only, CUDA, b=65536, ckpt=8, obs=16]
        # proj.prepare_buffers ≈ 0.005 ms/iter（~0.03% of projection_iter）
        normals, offsets, edge_mask, _, _ = self.collision_ops.obstacle_buffers(target)
        self.robot_transition.runtime_buffers(coeff)
        pos_const = self.robot_transition.checkpoint_position_const
        pos_gain = self.robot_transition.checkpoint_position_gain
        _pop()

        projected = target          # [..., A, 2]
        margin = torch.as_tensor(
            self.obstacle_collision_margin + self.obstacle_projection_extra_margin,
            dtype=target.dtype,
            device=target.device,
        )

        _push("proj.eval_candidate")
        # 关键路径 1：一次性评估初始目标点的违规候选。
        # [Profile] proj.eval_candidate ≈ 0.242 ms/iter（~1.39%）
        future_pos = self.future_positions_from_target(coeff, projected)
        # signed: [..., A, P, O, Emax]
        signed = self.collision_ops.obstacle_signed_edge_distances_any_agents(future_pos)
        # clearance / edge_idx: [..., A, P, O], 相对障碍物的【距离】
        clearance, edge_idx = signed.max(dim=-1)
        # candidate_violation: [..., A, P, O], 相对障碍物的【违反距离】
        candidate_violation = torch.clamp(margin - clearance, min=0.0)
        if active_mask is not None:
            # select 统计：active gating 的有效率（参与后续计算的候选比例）。
            self._sel_count(
                "candidate_active_gate",
                active_mask.unsqueeze(-1).unsqueeze(-1),
                torch.ones_like(candidate_violation, dtype=torch.bool),
            )
            candidate_violation = torch.where(
                active_mask.unsqueeze(-1).unsqueeze(-1),
                candidate_violation,
                torch.zeros_like(candidate_violation),
            )
        _pop()

        _push("proj.topk_select")
        # 关键路径 2：展平 [P, O]，做 top-k 候选筛选。
        # [Profile] proj.topk_select ≈ 0.075 ms/iter（~0.43%）
        # 备注：ncu 显示 top-k 相关 kernel（gatherTopK + bitonicSort）仍是主要 GPU 时间来源之一。
        candidate_count = clearance.shape[-2] * clearance.shape[-1]
        topk = min(self.obstacle_projection_topk, candidate_count)
        fixes_per_iter = min(self.obstacle_projection_fixes_per_iter, topk)
        # flat_violation: [..., A, P*O]
        flat_violation = candidate_violation.contiguous().view(*target.shape[:-1], candidate_count)
        # top_values/top_flat_idx: [..., A, K]
        top_values, top_flat_idx = torch.topk(flat_violation, k=topk, dim=-1)
        # checkpoint_idx / obstacle_idx: [..., A, K], 计算 点id / 障碍物id / 边id
        checkpoint_idx = torch.div(top_flat_idx, self.num_obstacles, rounding_mode="floor")
        obstacle_idx = top_flat_idx - checkpoint_idx * self.num_obstacles
        flat_edge_idx = edge_idx.contiguous().view(*target.shape[:-1], candidate_count)  # [..., A, P*O]
        selected_edge_idx = torch.gather(flat_edge_idx, -1, top_flat_idx)
        # select 统计：top-k 候选里正违规（>0）比例。
        self._sel_count(
            "topk_positive_violation",
            top_values > 0,
            torch.ones_like(top_values, dtype=torch.bool),
        )
        _pop()

        _push("proj.build_linear")
        # p_const: [..., A, P, 2]；selected_p_const: [..., A, K, 2]
        # [Profile] proj.build_linear ≈ 0.182 ms/iter（~1.05%）
        p_const = torch.einsum("pc,...acd->...apd", pos_const, coeff)
        selected_p_const = torch.gather(
            p_const,
            -2,
            checkpoint_idx.unsqueeze(-1).expand(*checkpoint_idx.shape, NDIM),
        )
        # selected_gain: [..., A, K], A个Agent, 每个 Agent 选出 Top-K 个
        selected_gain = pos_gain[checkpoint_idx]
        # selected_normals / selected_offsets / selected_edge_valid: [..., A, K, 2] / [..., A, K] / [..., A, K]
        selected_normals = normals[obstacle_idx, selected_edge_idx]
        selected_offsets = offsets[obstacle_idx, selected_edge_idx]
        selected_edge_valid = edge_mask[obstacle_idx, selected_edge_idx]
        selected_valid = selected_edge_valid & (top_values > 0)
        # select 统计：被 gather 到的边里，真实有效边比例（非 padding）。
        self._sel_count(
            "selected_edge_valid",
            selected_edge_valid,
            torch.ones_like(selected_edge_valid, dtype=torch.bool),
        )
        # select 统计：最终有效约束比例（边有效 且 正违规）。
        self._sel_count(
            "selected_constraint_valid",
            selected_valid,
            torch.ones_like(selected_valid, dtype=torch.bool),
        )

        # 关键路径 3：构造关于 target 的线性约束 a_vec * target <= rhs
        a_vec = selected_gain.unsqueeze(-1) * selected_normals          # [..., A, K, 2]
        rhs = selected_offsets + margin - torch.sum(selected_p_const * selected_normals, dim=-1)
        denom = torch.sum(a_vec * a_vec, dim=-1).clamp_min(torch.finfo(target.dtype).eps)
        _pop()

        _push("proj.iter_refine")
        # [Profile] proj.iter_refine ≈ 0.835 ms/iter（~4.81%）
        if self.iter_update_mode == "jacobi":
            for _ in range(iters):
                _push("proj.iter.violation")
                # Jacobi 批量修正：不过滤 fix 子集，直接对 K 个候选并行累积修正。
                violation = rhs - torch.sum(a_vec * projected.unsqueeze(-2), dim=-1)
                weighted_violation = torch.where(
                    selected_valid,
                    torch.clamp(violation, min=0.0),
                    torch.zeros_like(violation),
                )
                _pop()
                _push("proj.iter.apply_correction")
                correction = self.jacobi_relax * torch.sum(
                    (weighted_violation / denom).unsqueeze(-1) * a_vec,
                    dim=-2,
                )
                projected = projected + correction
                _pop()
                if clamp_to_dynamic_bounds:
                    _push("proj.iter.clamp_target")
                    projected = self.robot_transition.project_target(coeff, projected)
                    _pop()
        else:
            for _ in range(iters):
                _push("proj.iter.violation")
                # violation: [..., A, K]
                # [Profile] proj.iter.violation ≈ 0.140 ms/iter（~0.81%，约占 iter_refine 的 16.8%）
                violation = rhs - torch.sum(a_vec * projected.unsqueeze(-2), dim=-1)
                positive_violation = torch.where(
                    selected_valid,
                    torch.clamp(violation, min=0.0),
                    torch.zeros_like(violation),
                )
                _pop()
                _push("proj.iter.select_fix")
                # 每轮只修正最严重的 fixes_per_iter 个 violation，降低计算开销。
                # [Profile] proj.iter.select_fix ≈ 0.264 ms/iter（~1.52%，约占 iter_refine 的 31.6%）
                fix_values, fix_idx = torch.topk(positive_violation, k=fixes_per_iter, dim=-1)
                # select 统计：fix top-k 里正违规比例。
                self._sel_count(
                    "fix_topk_positive",
                    fix_values > 0,
                    torch.ones_like(fix_values, dtype=torch.bool),
                )
                fix_mask = F.one_hot(fix_idx, topk).bool()
                fix_mask = (fix_mask & (fix_values > 0).unsqueeze(-1)).any(dim=-2)
                # select 统计：每轮 K 个候选中真正进入修正的比例。
                self._sel_count(
                    "fix_mask_selected",
                    fix_mask,
                    torch.ones_like(fix_mask, dtype=torch.bool),
                )
                weighted_violation = torch.where(
                    fix_mask,
                    positive_violation,
                    torch.zeros_like(positive_violation),
                )
                _pop()
                _push("proj.iter.apply_correction")
                # correction: [..., A, 2]
                # [Profile] proj.iter.apply_correction ≈ 0.094 ms/iter（~0.54%，约占 iter_refine 的 11.3%）
                correction = torch.sum((weighted_violation / denom).unsqueeze(-1) * a_vec, dim=-2)
                projected = projected + correction
                _pop()
                if clamp_to_dynamic_bounds:
                    _push("proj.iter.clamp_target")
                    # [Profile] proj.iter.clamp_target ≈ 0.330 ms/iter（~1.90%，约占 iter_refine 的 39.6%）
                    projected = self.robot_transition.project_target(coeff, projected)
                    _pop()
        _pop()

        if return_residual:
            _push("proj.final_residual")
            clearance = self.future_obstacle_clearance_from_target(coeff, projected)
            residual = (clearance <= self.obstacle_collision_margin).any(dim=(-1, -2))
            _pop()
        else:
            residual = torch.zeros(target.shape[:-1], dtype=torch.bool, device=target.device)
        if active_mask is not None:
            residual = residual & active_mask
        return projected, torch.tensor(iters, dtype=torch.long, device=target.device), residual
