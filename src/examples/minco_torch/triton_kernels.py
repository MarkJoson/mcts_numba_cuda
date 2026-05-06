"""Triton 融合内核：MINCO 投影热路径。"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - 无 Triton 环境时走 eager 回退
    triton = None
    tl = None


def triton_available() -> bool:
    """是否可用 Triton 内核。"""
    return (triton is not None) and (tl is not None) and torch.cuda.is_available()


if triton is not None and tl is not None:

    @triton.jit
    def _jacobi_update_kernel(
        p_ptr,             # [N, 2]
        a_ptr,             # [N, K, 2]
        rhs_ptr,           # [N, K]
        den_ptr,           # [N, K]
        val_ptr,           # [N, K] (int8: 0/1)
        n_rows,            # N
        k_cols,            # K
        s_pn, s_pd,        # projected strides
        s_an, s_ak, s_ad,  # a_vec strides
        s_rn, s_rk,        # rhs strides
        s_dn, s_dk,        # denom strides
        s_vn, s_vk,        # valid strides
        alpha,             # jacobi_relax
        BLOCK_K: tl.constexpr,
    ):
        """单行融合：violation + clamp + valid mask + reduce + projected update。"""
        row = tl.program_id(0)
        if row >= n_rows:
            return

        x0 = tl.load(p_ptr + row * s_pn + 0 * s_pd)
        x1 = tl.load(p_ptr + row * s_pn + 1 * s_pd)

        offs_k = tl.arange(0, BLOCK_K)
        mk = offs_k < k_cols

        a0 = tl.load(a_ptr + row * s_an + offs_k * s_ak + 0 * s_ad, mask=mk, other=0.0)
        a1 = tl.load(a_ptr + row * s_an + offs_k * s_ak + 1 * s_ad, mask=mk, other=0.0)
        rhs = tl.load(rhs_ptr + row * s_rn + offs_k * s_rk, mask=mk, other=0.0)
        den = tl.load(den_ptr + row * s_dn + offs_k * s_dk, mask=mk, other=1.0)
        val = tl.load(val_ptr + row * s_vn + offs_k * s_vk, mask=mk, other=0)

        viol = rhs - (a0 * x0 + a1 * x1)
        pos_viol = tl.maximum(viol, 0.0)
        pos_viol = tl.where(val != 0, pos_viol, 0.0)
        scale = pos_viol / den

        corr0 = tl.sum(scale * a0, axis=0)
        corr1 = tl.sum(scale * a1, axis=0)

        x0 = x0 + alpha * corr0
        x1 = x1 + alpha * corr1

        tl.store(p_ptr + row * s_pn + 0 * s_pd, x0)
        tl.store(p_ptr + row * s_pn + 1 * s_pd, x1)

    @triton.jit
    def _jacobi_update_clamp_kernel(
        p_ptr,             # [N, 2]
        a_ptr,             # [N, K, 2]
        rhs_ptr,           # [N, K]
        den_ptr,           # [N, K]
        val_ptr,           # [N, K] (int8: 0/1)
        lo_ptr,            # [N, 2]
        hi_ptr,            # [N, 2]
        n_rows,            # N
        k_cols,            # K
        s_pn, s_pd,        # projected strides
        s_an, s_ak, s_ad,  # a_vec strides
        s_rn, s_rk,        # rhs strides
        s_dn, s_dk,        # denom strides
        s_vn, s_vk,        # valid strides
        s_ln, s_ld,        # lower strides
        s_hn, s_hd,        # upper strides
        alpha,             # jacobi_relax
        BLOCK_K: tl.constexpr,
    ):
        """单行融合：violation + clamp + valid + reduce + update + bounds clamp。"""
        row = tl.program_id(0)
        if row >= n_rows:
            return

        x0 = tl.load(p_ptr + row * s_pn + 0 * s_pd)
        x1 = tl.load(p_ptr + row * s_pn + 1 * s_pd)

        offs_k = tl.arange(0, BLOCK_K)
        mk = offs_k < k_cols

        a0 = tl.load(a_ptr + row * s_an + offs_k * s_ak + 0 * s_ad, mask=mk, other=0.0)
        a1 = tl.load(a_ptr + row * s_an + offs_k * s_ak + 1 * s_ad, mask=mk, other=0.0)
        rhs = tl.load(rhs_ptr + row * s_rn + offs_k * s_rk, mask=mk, other=0.0)
        den = tl.load(den_ptr + row * s_dn + offs_k * s_dk, mask=mk, other=1.0)
        val = tl.load(val_ptr + row * s_vn + offs_k * s_vk, mask=mk, other=0)

        viol = rhs - (a0 * x0 + a1 * x1)
        pos_viol = tl.maximum(viol, 0.0)
        pos_viol = tl.where(val != 0, pos_viol, 0.0)
        scale = pos_viol / den

        corr0 = tl.sum(scale * a0, axis=0)
        corr1 = tl.sum(scale * a1, axis=0)

        x0 = x0 + alpha * corr0
        x1 = x1 + alpha * corr1

        lo0 = tl.load(lo_ptr + row * s_ln + 0 * s_ld)
        lo1 = tl.load(lo_ptr + row * s_ln + 1 * s_ld)
        hi0 = tl.load(hi_ptr + row * s_hn + 0 * s_hd)
        hi1 = tl.load(hi_ptr + row * s_hn + 1 * s_hd)
        x0 = tl.minimum(tl.maximum(x0, lo0), hi0)
        x1 = tl.minimum(tl.maximum(x1, lo1), hi1)

        tl.store(p_ptr + row * s_pn + 0 * s_pd, x0)
        tl.store(p_ptr + row * s_pn + 1 * s_pd, x1)

    @triton.jit
    def _jacobi_update_megakernel(
        p_ptr,               # [N, 2]
        g_ptr,               # [N, K]
        n_ptr,               # [N, K, 2]
        o_ptr,               # [N, K]
        pc_ptr,              # [N, K, 2]
        v_ptr,               # [N, K] int8
        lo_ptr,              # [N, 2] or nullptr
        hi_ptr,              # [N, 2] or nullptr
        n_rows,              # N
        k_cols,              # K
        margin,              # scalar
        alpha,               # scalar
        do_clamp,            # int
        s_pn, s_pd,
        s_gn, s_gk,
        s_nn, s_nk, s_nd,
        s_on, s_ok,
        s_pcn, s_pck, s_pcd,
        s_vn, s_vk,
        s_ln, s_ld,
        s_hn, s_hd,
        BLOCK_K: tl.constexpr,
    ):
        """Megakernel: constraint build + violation + update (+ optional clamp)."""
        row = tl.program_id(0)
        if row >= n_rows:
            return

        x0 = tl.load(p_ptr + row * s_pn + 0 * s_pd)
        x1 = tl.load(p_ptr + row * s_pn + 1 * s_pd)

        offs_k = tl.arange(0, BLOCK_K)
        mk = offs_k < k_cols

        gain = tl.load(g_ptr + row * s_gn + offs_k * s_gk, mask=mk, other=0.0)
        n0 = tl.load(n_ptr + row * s_nn + offs_k * s_nk + 0 * s_nd, mask=mk, other=0.0)
        n1 = tl.load(n_ptr + row * s_nn + offs_k * s_nk + 1 * s_nd, mask=mk, other=0.0)
        off = tl.load(o_ptr + row * s_on + offs_k * s_ok, mask=mk, other=0.0)
        pc0 = tl.load(pc_ptr + row * s_pcn + offs_k * s_pck + 0 * s_pcd, mask=mk, other=0.0)
        pc1 = tl.load(pc_ptr + row * s_pcn + offs_k * s_pck + 1 * s_pcd, mask=mk, other=0.0)
        val = tl.load(v_ptr + row * s_vn + offs_k * s_vk, mask=mk, other=0)

        a0 = gain * n0
        a1 = gain * n1
        rhs = off + margin - (pc0 * n0 + pc1 * n1)
        den = tl.maximum(a0 * a0 + a1 * a1, 1.0e-12)

        viol = rhs - (a0 * x0 + a1 * x1)
        pos_viol = tl.maximum(viol, 0.0)
        pos_viol = tl.where(val != 0, pos_viol, 0.0)
        scale = pos_viol / den
        corr0 = tl.sum(scale * a0, axis=0)
        corr1 = tl.sum(scale * a1, axis=0)

        x0 = x0 + alpha * corr0
        x1 = x1 + alpha * corr1

        if do_clamp != 0:
            lo0 = tl.load(lo_ptr + row * s_ln + 0 * s_ld)
            lo1 = tl.load(lo_ptr + row * s_ln + 1 * s_ld)
            hi0 = tl.load(hi_ptr + row * s_hn + 0 * s_hd)
            hi1 = tl.load(hi_ptr + row * s_hn + 1 * s_hd)
            x0 = tl.minimum(tl.maximum(x0, lo0), hi0)
            x1 = tl.minimum(tl.maximum(x1, lo1), hi1)

        tl.store(p_ptr + row * s_pn + 0 * s_pd, x0)
        tl.store(p_ptr + row * s_pn + 1 * s_pd, x1)

    @triton.jit
    def _topk_linear_build_kernel(
        viol_ptr,              # [N, C]
        edge_ptr,              # [N, C]
        pconst_ptr,            # [N, P, 2]
        gtab_ptr,              # [P]
        ntab_ptr,              # [O, E, 2]
        otab_ptr,              # [O, E]
        em_ptr,                # [O, E] int8
        c_per_obs,             # C = P*O
        n_rows,                # N
        n_obs,                 # O
        n_p,                   # P
        n_e,                   # E
        topk,                  # K
        out_gain_ptr,          # [N, K]
        out_norm_ptr,          # [N, K, 2]
        out_off_ptr,           # [N, K]
        out_pconst_ptr,        # [N, K, 2]
        out_valid_ptr,         # [N, K] int8
        sv_n, sv_c,
        se_n, se_c,
        sp_n, sp_p, sp_d,
        sg_p,
        sn_o, sn_e, sn_d,
        so_o, so_e,
        sm_o, sm_e,
        sog_n, sog_k,
        son_n, son_k, son_d,
        soo_n, soo_k,
        sop_n, sop_k, sop_d,
        sov_n, sov_k,
        BLOCK_C: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        """按行融合：top-k(带索引) + obstacle/edge gather + 线性约束构建输入。"""
        row = tl.program_id(0)
        if row >= n_rows:
            return

        offs = tl.arange(0, BLOCK_C)
        m = offs < c_per_obs
        cur = tl.load(viol_ptr + row * sv_n + offs * sv_c, mask=m, other=-1.0e9)
        for kk in tl.static_range(0, TOPK):
            vmax, imax = tl.max(cur, axis=0, return_indices=True)
            one = tl.where(offs == imax, 1, 0)
            cid = tl.sum(offs * one, axis=0)
            ck = cid // n_obs
            oi = cid - ck * n_obs
            ei = tl.load(edge_ptr + row * se_n + cid * se_c)
            gd = tl.load(gtab_ptr + ck * sg_p)
            n0 = tl.load(ntab_ptr + oi * sn_o + ei * sn_e + 0 * sn_d)
            n1 = tl.load(ntab_ptr + oi * sn_o + ei * sn_e + 1 * sn_d)
            of = tl.load(otab_ptr + oi * so_o + ei * so_e)
            pc0 = tl.load(pconst_ptr + row * sp_n + ck * sp_p + 0 * sp_d)
            pc1 = tl.load(pconst_ptr + row * sp_n + ck * sp_p + 1 * sp_d)
            em = tl.load(em_ptr + oi * sm_o + ei * sm_e)

            # candidate_violation 已经 clamp>=0，这里用 vmax>0 判断是否有效约束。
            vv = (em != 0) & (vmax > 0.0)

            tl.store(out_gain_ptr + row * sog_n + kk * sog_k, gd)
            tl.store(out_norm_ptr + row * son_n + kk * son_k + 0 * son_d, n0)
            tl.store(out_norm_ptr + row * son_n + kk * son_k + 1 * son_d, n1)
            tl.store(out_off_ptr + row * soo_n + kk * soo_k, of)
            tl.store(out_pconst_ptr + row * sop_n + kk * sop_k + 0 * sop_d, pc0)
            tl.store(out_pconst_ptr + row * sop_n + kk * sop_k + 1 * sop_d, pc1)
            tl.store(out_valid_ptr + row * sov_n + kk * sov_k, vv)

            cur = tl.where(offs == cid, -1.0e9, cur)


def jacobi_update_fused_inplace(
    projected_flat: torch.Tensor,
    a_vec_flat: torch.Tensor,
    rhs_flat: torch.Tensor,
    denom_flat: torch.Tensor,
    valid_flat_i8: torch.Tensor,
    jacobi_relax: float,
    lower_flat: torch.Tensor | None = None,
    upper_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """对 ``projected_flat[...,2]`` 执行一次 Jacobi 融合更新（原地）。

    形状约定：
        projected_flat: ``(N, 2)``
        a_vec_flat: ``(N, K, 2)``
        rhs_flat / denom_flat / valid_flat_i8: ``(N, K)``
    """
    if not triton_available():
        raise RuntimeError("Triton is unavailable in current runtime")
    if projected_flat.ndim != 2 or projected_flat.shape[-1] != 2:
        raise ValueError(f"projected_flat must be (N, 2), got {tuple(projected_flat.shape)}")
    if a_vec_flat.ndim != 3 or a_vec_flat.shape[-1] != 2:
        raise ValueError(f"a_vec_flat must be (N, K, 2), got {tuple(a_vec_flat.shape)}")
    if rhs_flat.shape != denom_flat.shape or rhs_flat.shape != valid_flat_i8.shape:
        raise ValueError("rhs/denom/valid_flat_i8 shape mismatch")
    if rhs_flat.shape[0] != projected_flat.shape[0] or rhs_flat.shape[0] != a_vec_flat.shape[0]:
        raise ValueError("N dimension mismatch between projected/a_vec/rhs")
    if rhs_flat.shape[1] != a_vec_flat.shape[1]:
        raise ValueError("K dimension mismatch between a_vec/rhs")
    if projected_flat.device.type != "cuda":
        raise ValueError("jacobi_update_fused_inplace expects CUDA tensors")
    if projected_flat.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"unsupported projected dtype: {projected_flat.dtype}")
    if a_vec_flat.dtype != projected_flat.dtype or rhs_flat.dtype != projected_flat.dtype or denom_flat.dtype != projected_flat.dtype:
        raise ValueError("projected/a_vec/rhs/denom must share the same dtype")
    if valid_flat_i8.dtype != torch.int8:
        raise ValueError(f"valid_flat_i8 must be int8, got {valid_flat_i8.dtype}")
    use_clamp = (lower_flat is not None) or (upper_flat is not None)
    if use_clamp:
        if lower_flat is None or upper_flat is None:
            raise ValueError("lower_flat and upper_flat must be both None or both provided")
        if lower_flat.shape != projected_flat.shape or upper_flat.shape != projected_flat.shape:
            raise ValueError("lower_flat/upper_flat must have shape (N, 2)")
        if lower_flat.dtype != projected_flat.dtype or upper_flat.dtype != projected_flat.dtype:
            raise ValueError("lower_flat/upper_flat dtype must match projected_flat")

    n_rows = int(projected_flat.shape[0])
    k_cols = int(rhs_flat.shape[1])
    if n_rows == 0 or k_cols == 0:
        return projected_flat

    block_k = int(min(128, triton.next_power_of_2(k_cols)))
    num_warps = 1 if block_k <= 32 else 2 if block_k <= 64 else 4
    grid = (n_rows,)
    if use_clamp:
        _jacobi_update_clamp_kernel[grid](
            projected_flat,
            a_vec_flat,
            rhs_flat,
            denom_flat,
            valid_flat_i8,
            lower_flat,
            upper_flat,
            n_rows,
            k_cols,
            projected_flat.stride(0),
            projected_flat.stride(1),
            a_vec_flat.stride(0),
            a_vec_flat.stride(1),
            a_vec_flat.stride(2),
            rhs_flat.stride(0),
            rhs_flat.stride(1),
            denom_flat.stride(0),
            denom_flat.stride(1),
            valid_flat_i8.stride(0),
            valid_flat_i8.stride(1),
            lower_flat.stride(0),
            lower_flat.stride(1),
            upper_flat.stride(0),
            upper_flat.stride(1),
            float(jacobi_relax),
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _jacobi_update_kernel[grid](
            projected_flat,
            a_vec_flat,
            rhs_flat,
            denom_flat,
            valid_flat_i8,
            n_rows,
            k_cols,
            projected_flat.stride(0),
            projected_flat.stride(1),
            a_vec_flat.stride(0),
            a_vec_flat.stride(1),
            a_vec_flat.stride(2),
            rhs_flat.stride(0),
            rhs_flat.stride(1),
            denom_flat.stride(0),
            denom_flat.stride(1),
            valid_flat_i8.stride(0),
            valid_flat_i8.stride(1),
            float(jacobi_relax),
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return projected_flat


def jacobi_update_megakernel_inplace(
    projected_flat: torch.Tensor,
    selected_gain_flat: torch.Tensor,
    selected_normals_flat: torch.Tensor,
    selected_offsets_flat: torch.Tensor,
    selected_pconst_flat: torch.Tensor,
    valid_flat_i8: torch.Tensor,
    margin: float,
    jacobi_relax: float,
    lower_flat: torch.Tensor | None = None,
    upper_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Megakernel 版本：在内核内构造约束并完成一次 Jacobi 更新（原地）。"""
    if not triton_available():
        raise RuntimeError("Triton is unavailable in current runtime")
    if projected_flat.ndim != 2 or projected_flat.shape[-1] != 2:
        raise ValueError(f"projected_flat must be (N, 2), got {tuple(projected_flat.shape)}")
    if selected_gain_flat.ndim != 2:
        raise ValueError("selected_gain_flat must be (N, K)")
    if selected_offsets_flat.shape != selected_gain_flat.shape:
        raise ValueError("selected_offsets_flat must have shape (N, K)")
    if valid_flat_i8.shape != selected_gain_flat.shape or valid_flat_i8.dtype != torch.int8:
        raise ValueError("valid_flat_i8 must be int8 and match (N, K)")
    if selected_normals_flat.shape != (*selected_gain_flat.shape, 2):
        raise ValueError("selected_normals_flat must be (N, K, 2)")
    if selected_pconst_flat.shape != (*selected_gain_flat.shape, 2):
        raise ValueError("selected_pconst_flat must be (N, K, 2)")
    if projected_flat.device.type != "cuda":
        raise ValueError("jacobi_update_megakernel_inplace expects CUDA tensors")
    if projected_flat.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"unsupported projected dtype: {projected_flat.dtype}")

    for t in (selected_gain_flat, selected_offsets_flat, selected_normals_flat, selected_pconst_flat):
        if t.dtype != projected_flat.dtype or t.device != projected_flat.device:
            raise ValueError("all floating tensors must share dtype/device with projected_flat")

    use_clamp = (lower_flat is not None) or (upper_flat is not None)
    if use_clamp:
        if lower_flat is None or upper_flat is None:
            raise ValueError("lower_flat and upper_flat must be both None or both provided")
        if lower_flat.shape != projected_flat.shape or upper_flat.shape != projected_flat.shape:
            raise ValueError("lower_flat/upper_flat must be (N, 2)")
        if lower_flat.dtype != projected_flat.dtype or upper_flat.dtype != projected_flat.dtype:
            raise ValueError("lower_flat/upper_flat dtype mismatch")

    if projected_flat.is_inference():
        # 避免编译图中原地突变导致 cudagraph 被跳过。
        projected_flat = projected_flat.clone()

    n_rows = int(projected_flat.shape[0])
    k_cols = int(selected_gain_flat.shape[1])
    if n_rows == 0 or k_cols == 0:
        return projected_flat

    block_k = int(min(128, triton.next_power_of_2(k_cols)))
    num_warps = 1 if block_k <= 32 else 2 if block_k <= 64 else 4
    grid = (n_rows,)
    _jacobi_update_megakernel[grid](
        projected_flat,
        selected_gain_flat,
        selected_normals_flat,
        selected_offsets_flat,
        selected_pconst_flat,
        valid_flat_i8,
        lower_flat if use_clamp else projected_flat,  # dummy ptr if no clamp
        upper_flat if use_clamp else projected_flat,
        n_rows,
        k_cols,
        float(margin),
        float(jacobi_relax),
        int(1 if use_clamp else 0),
        projected_flat.stride(0),
        projected_flat.stride(1),
        selected_gain_flat.stride(0),
        selected_gain_flat.stride(1),
        selected_normals_flat.stride(0),
        selected_normals_flat.stride(1),
        selected_normals_flat.stride(2),
        selected_offsets_flat.stride(0),
        selected_offsets_flat.stride(1),
        selected_pconst_flat.stride(0),
        selected_pconst_flat.stride(1),
        selected_pconst_flat.stride(2),
        valid_flat_i8.stride(0),
        valid_flat_i8.stride(1),
        0 if not use_clamp else lower_flat.stride(0),
        0 if not use_clamp else lower_flat.stride(1),
        0 if not use_clamp else upper_flat.stride(0),
        0 if not use_clamp else upper_flat.stride(1),
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return projected_flat


def build_topk_linear_inputs(
    candidate_violation_flat: torch.Tensor,
    edge_idx_flat: torch.Tensor,
    p_const_flat: torch.Tensor,
    pos_gain: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    edge_mask: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 Triton 内融合 top-k 选择与 gather，输出 Jacobi 线性输入。"""
    if not triton_available():
        raise RuntimeError("Triton is unavailable in current runtime")
    if candidate_violation_flat.ndim != 2:
        raise ValueError("candidate_violation_flat must be (N, C)")
    if edge_idx_flat.shape != candidate_violation_flat.shape:
        raise ValueError("edge_idx_flat shape mismatch")
    if p_const_flat.ndim != 3 or p_const_flat.shape[-1] != 2:
        raise ValueError("p_const_flat must be (N, P, 2)")
    if pos_gain.ndim != 1:
        raise ValueError("pos_gain must be (P,)")
    if normals.ndim != 3 or normals.shape[-1] != 2:
        raise ValueError("normals must be (O, E, 2)")
    if offsets.shape != edge_mask.shape or offsets.ndim != 2:
        raise ValueError("offsets/edge_mask must be (O, E)")
    if candidate_violation_flat.device.type != "cuda":
        raise ValueError("build_topk_linear_inputs expects CUDA tensors")
    if candidate_violation_flat.dtype not in (torch.float16, torch.float32):
        raise ValueError("candidate_violation_flat must be fp16/fp32")

    n_rows = int(candidate_violation_flat.shape[0])
    c_per_obs = int(candidate_violation_flat.shape[1])
    n_p = int(p_const_flat.shape[1])
    n_obs = int(normals.shape[0])
    n_e = int(normals.shape[1])
    if c_per_obs != n_p * n_obs:
        raise ValueError("candidate count C must equal P*O")
    k = int(topk)
    if k <= 0:
        raise ValueError("topk must be positive")
    if (k & (k - 1)) != 0:
        raise ValueError("topk must be power-of-two for Triton static top-k")

    dev = candidate_violation_flat.device
    dt = candidate_violation_flat.dtype
    out_gain = torch.empty((n_rows, k), device=dev, dtype=dt)
    out_norm = torch.empty((n_rows, k, 2), device=dev, dtype=dt)
    out_off = torch.empty((n_rows, k), device=dev, dtype=dt)
    out_pconst = torch.empty((n_rows, k, 2), device=dev, dtype=dt)
    out_valid = torch.empty((n_rows, k), device=dev, dtype=torch.int8)

    edge_i32 = edge_idx_flat.to(torch.int32, copy=False)
    em_i8 = edge_mask.to(torch.int8, copy=False)

    block_c = int(min(256, triton.next_power_of_2(c_per_obs)))
    grid = (n_rows,)
    _topk_linear_build_kernel[grid](
        candidate_violation_flat,
        edge_i32,
        p_const_flat,
        pos_gain,
        normals,
        offsets,
        em_i8,
        c_per_obs,
        n_rows,
        n_obs,
        n_p,
        n_e,
        k,
        out_gain,
        out_norm,
        out_off,
        out_pconst,
        out_valid,
        candidate_violation_flat.stride(0),
        candidate_violation_flat.stride(1),
        edge_i32.stride(0),
        edge_i32.stride(1),
        p_const_flat.stride(0),
        p_const_flat.stride(1),
        p_const_flat.stride(2),
        pos_gain.stride(0),
        normals.stride(0),
        normals.stride(1),
        normals.stride(2),
        offsets.stride(0),
        offsets.stride(1),
        em_i8.stride(0),
        em_i8.stride(1),
        out_gain.stride(0),
        out_gain.stride(1),
        out_norm.stride(0),
        out_norm.stride(1),
        out_norm.stride(2),
        out_off.stride(0),
        out_off.stride(1),
        out_pconst.stride(0),
        out_pconst.stride(1),
        out_pconst.stride(2),
        out_valid.stride(0),
        out_valid.stride(1),
        BLOCK_C=block_c,
        TOPK=k,
        num_warps=4,
        num_stages=1,
    )
    return out_gain, out_norm, out_off, out_pconst, out_valid
