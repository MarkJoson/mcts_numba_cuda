"""One-time algebra and basis construction for MINCO rollout."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .constants import NCOFF, S

def construct_beta_t(
    t: float | torch.Tensor,
    rank: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return beta(t, rank) with shape ``(NCOFF=6, 1)``.

    Inputs:
        t: scalar time.
        rank: derivative order in ``[0, 5]``.
    Output:
        beta: ``(6, 1)`` polynomial basis column.

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
    """Return ``integral beta beta^T dt`` used by the MINCO/LQR objective.

    Inputs:
        piece_t: scalar segment duration.
        rank: derivative order, usually ``S=3`` for jerk/snap-like cost.
    Output:
        bbint: ``(6, 6)`` quadratic cost matrix.
    """

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
    """Return the modified MINCO boundary matrix.

    Input:
        piece_t: scalar planner segment duration.
    Output:
        mat_m: ``(6, 6)`` boundary/continuity matrix.
    """

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
    """Return rows ``beta(piece_t, rank)^T`` for ranks 0..5.

    Input:
        piece_t: scalar execution step duration.
    Output:
        mat_r: ``(6, 6)`` state-shift/evaluation matrix.
    """

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

    Inputs:
        a_mat: ``(6, 6)`` linear dynamics matrix.
        b_mat: ``(6, 1)`` control/input matrix.
        q_mat: ``(6, 6)`` quadratic state cost.
        r_mat: ``(1, 1)`` quadratic input cost.
    Output:
        gain: ``(1, 6)`` feedback matrix ``K``.

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
