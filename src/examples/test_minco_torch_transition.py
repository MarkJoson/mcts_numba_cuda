"""Numerical checks for ``minco_torch_transition.py``.

Run with a PyTorch environment, for example:

    PYTHONPATH=src conda run -n py310_torch280 python src/examples/test_minco_torch_transition.py
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    print("  [SKIP]  minco torch transition tests  (torch is not installed)")
    sys.exit(0)

from examples.minco_torch_transition import (  # noqa: E402
    NCOFF,
    S,
    MincoPointEnvTransition,
    MincoTorchTransition,
)


PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"


def record(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS_MARK if ok else FAIL_MARK
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        raise AssertionError(name)


def np_beta(t: float, rank: int) -> np.ndarray:
    beta = np.zeros((NCOFF, 1), dtype=np.float64)
    for i in range(rank, NCOFF):
        if float(t) != 0.0 or i - rank == 0:
            beta[i, 0] = math.factorial(i) / math.factorial(i - rank) * t ** (i - rank)
    return beta


def np_bbint(piece_t: float, rank: int) -> np.ndarray:
    beta = np_beta(piece_t, rank)
    out = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF):
        for j in range(NCOFF):
            denom = i + j - 2 * rank + 1
            if denom > 0:
                out[i, j] = beta[i, 0] * beta[j, 0] * piece_t / denom
    return out


def np_minco_m2(piece_t: float) -> np.ndarray:
    mat_m = np.zeros((NCOFF, NCOFF), dtype=np.float64)
    for i in range(NCOFF - 2):
        mat_m[i, :] = np_beta(0.0, i).T
    mat_m[-2, :] = np_beta(piece_t, 0).T
    mat_m[-1, :] = np_beta(piece_t, 1).T
    mat_m_inv = np.linalg.inv(mat_m)
    mat_supp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) @ mat_m_inv @ np_bbint(piece_t, S)
    mat_m[-1, :] = mat_supp[-1, :]
    return mat_m


def np_mat_r(piece_t: float) -> np.ndarray:
    return np.vstack([np_beta(piece_t, i).T for i in range(NCOFF)])


def np_lqr_gain(a_mat: np.ndarray, b_mat: np.ndarray, q_mat: np.ndarray,
                r_mat: np.ndarray, *, max_iter: int = 10_000,
                tol: float = 1e-12) -> np.ndarray:
    p_mat = q_mat.copy()
    for _ in range(max_iter):
        bt_p = b_mat.T @ p_mat
        gain = np.linalg.solve(r_mat + bt_p @ b_mat, bt_p @ a_mat)
        next_p = q_mat + a_mat.T @ p_mat @ a_mat - a_mat.T @ p_mat @ b_mat @ gain
        delta = np.linalg.norm(next_p - p_mat)
        scale = max(1.0, np.linalg.norm(p_mat))
        p_mat = next_p
        if delta / scale <= tol:
            break
    bt_p = b_mat.T @ p_mat
    return np.linalg.solve(r_mat + bt_p @ b_mat, bt_p @ a_mat)


class NumpyExperiment2Planner:
    def __init__(self, piece_t: float, ratio: float = 0.1, n_checkpoints: int = 20):
        self.piece_t = float(piece_t) / float(ratio)
        self.n_checkpoints = int(n_checkpoints)
        mat_m = np_minco_m2(self.piece_t)
        mat_m_inv = np.linalg.inv(mat_m)
        mat_s = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        mat_u = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).T
        mat_r = np_mat_r(piece_t)
        self.mat_f = mat_m_inv @ mat_s @ mat_r
        self.mat_g = (mat_m_inv @ mat_u).reshape((-1, 1))
        q_mat = np_bbint(piece_t, S)
        k_mat = np_lqr_gain(self.mat_f, self.mat_g, q_mat, np.array([[1.0]]))
        kpp = np.linalg.pinv(self.mat_g) @ (np.eye(NCOFF) - self.mat_f) + k_mat
        self.mat_f_stab = self.mat_f - self.mat_g @ k_mat
        self.mat_g_stab = self.mat_g @ kpp @ np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

    def iter_func_new(self, tgt_pos: np.ndarray, last_coff: np.ndarray) -> np.ndarray:
        return self.mat_f_stab @ last_coff + self.mat_g_stab @ tgt_pos.reshape(1, -1)

    def calc_bound(self, t: float, rank: int, val: float,
                   coff: np.ndarray) -> np.ndarray:
        const_coeff = np_beta(t, rank).T @ self.mat_f_stab
        bound_coeff = float((np_beta(t, rank).T @ self.mat_g_stab).reshape(()))
        val = -val if bound_coeff < 0.0 else val
        return np.array([
            (-val - const_coeff @ coff) / bound_coeff,
            (val - const_coeff @ coff) / bound_coeff,
        ])

    def bound_all(self, coff: np.ndarray) -> np.ndarray:
        ts = [(i + 1) / (self.n_checkpoints + 1) * self.piece_t
              for i in range(self.n_checkpoints)]
        vel_bounds = np.concatenate([self.calc_bound(t, 1, 2.0, coff) for t in ts], axis=1)
        acc_bounds = np.concatenate([self.calc_bound(t, 2, 3.0, coff) for t in ts], axis=1)
        bounds = np.concatenate([vel_bounds, acc_bounds], axis=1)
        return np.array([np.max(bounds[0, :], axis=0), np.min(bounds[1, :], axis=0)])


def test_matrices_and_one_step() -> None:
    torch.set_default_dtype(torch.float64)
    transition = MincoTorchTransition(0.1, dtype=torch.float64)
    ref = NumpyExperiment2Planner(0.1)

    coeff = np.array([
        [0.2, -0.3],
        [0.1, 0.05],
        [-0.02, 0.03],
        [0.0, 0.01],
        [0.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float64)
    target = np.array([1.5, -0.7], dtype=np.float64)

    torch_next = transition.step(
        torch.from_numpy(coeff),
        torch.from_numpy(target),
        clamp_target=False,
    ).detach().numpy()
    np_next = ref.iter_func_new(target, coeff)

    record("stable F matrix matches numpy reference",
           np.allclose(transition.mat_f_stab.detach().numpy(), ref.mat_f_stab, atol=1e-10))
    record("stable G matrix matches numpy reference",
           np.allclose(transition.mat_g_stab.detach().numpy(), ref.mat_g_stab, atol=1e-10))
    record("single transition matches numpy reference",
           np.allclose(torch_next, np_next, atol=1e-10))


def test_bounds_batch_and_flat_state() -> None:
    transition = MincoTorchTransition(0.1, dtype=torch.float64)
    ref = NumpyExperiment2Planner(0.1)

    coeff0 = np.zeros((NCOFF, 2), dtype=np.float64)
    coeff0[0] = [0.0, 0.0]
    coeff1 = ref.iter_func_new(np.array([0.8, -0.4], dtype=np.float64), coeff0)
    coeff_batch = np.stack([coeff0, coeff1], axis=0)
    target_batch = np.array([[100.0, -100.0], [0.3, 0.6]], dtype=np.float64)

    coeff_t = torch.from_numpy(coeff_batch)
    lower_t, upper_t = transition.bounds(coeff_t)
    projected_t = transition.project_target(coeff_t, torch.from_numpy(target_batch))
    next_flat_t, projected_flat_t = transition.step_flat(
        transition.flatten_coefficients(coeff_t),
        torch.from_numpy(target_batch),
        return_projected=True,
    )

    ref_bounds = np.stack([ref.bound_all(coeff_batch[i]) for i in range(2)], axis=0)
    ref_projected = np.clip(target_batch, ref_bounds[:, 0, :], ref_bounds[:, 1, :])
    ref_next = np.stack([
        ref.iter_func_new(ref_projected[i], coeff_batch[i])
        for i in range(2)
    ], axis=0).reshape(2, -1)

    record("batched bounds match numpy reference",
           np.allclose(lower_t.detach().numpy(), ref_bounds[:, 0, :], atol=1e-10)
           and np.allclose(upper_t.detach().numpy(), ref_bounds[:, 1, :], atol=1e-10))
    record("target projection matches numpy clip",
           np.allclose(projected_t.detach().numpy(), ref_projected, atol=1e-10)
           and np.allclose(projected_flat_t.detach().numpy(), ref_projected, atol=1e-10))
    record("flat batched step matches numpy reference",
           np.allclose(next_flat_t.detach().numpy(), ref_next, atol=1e-10))


def test_pool_gather_and_rollout() -> None:
    transition = MincoTorchTransition(0.1, dtype=torch.float64)
    ref = NumpyExperiment2Planner(0.1)

    root = transition.initial_coefficients([0.0, 0.0], batch_shape=(2, 3))
    root_flat = transition.flatten_coefficients(root)
    tree_ids = torch.tensor([0, 1, 1], dtype=torch.long)
    node_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    targets = torch.tensor([[0.2, 0.1], [0.5, -0.1], [-0.4, 0.3]], dtype=torch.float64)

    gathered_next = transition.gather_step_flat(
        root_flat,
        tree_ids,
        node_ids,
        targets,
        clamp_target=False,
    )
    direct_next = transition.step_flat(
        root_flat[tree_ids, node_ids],
        targets,
        clamp_target=False,
    )
    record("flat pool gather step matches direct indexed step",
           torch.allclose(gathered_next, direct_next, atol=1e-12))

    init = np.zeros((NCOFF, 2), dtype=np.float64)
    seq = np.array([[0.2, 0.0], [0.4, 0.1], [0.5, -0.2]], dtype=np.float64)
    expected = init
    for target in seq:
        expected = ref.iter_func_new(target, expected)

    rollout_out = transition.rollout_flat(
        transition.flatten_coefficients(torch.from_numpy(init)),
        torch.from_numpy(seq),
        clamp_target=False,
    )
    record("flat recurrent rollout matches numpy loop",
           np.allclose(rollout_out.detach().numpy(), expected.reshape(-1), atol=1e-10))


def test_point_env_time_pack_and_gather() -> None:
    env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        tf=0.3,
        dtype=torch.float64,
        position_bounds=((-10.0, 10.0), (-10.0, 10.0)),
    )
    positions = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
    ], dtype=torch.float64)
    flat = env.initial_flat_state(positions)
    target = positions + torch.tensor([0.2, 0.0], dtype=torch.float64)

    next_flat, done, info = env.step_flat(flat, target, return_info=True)
    coeff, active, time = env.unpack_state(next_flat)
    record("point env flat state packs time and active mask",
           next_flat.shape == (2, env.state_dim)
           and torch.allclose(time, torch.full((2,), 0.1, dtype=torch.float64))
           and torch.all(active == 1.0)
           and not done.any()
           and info.projected_targets.shape == (2, 4, 2)
           and coeff.shape == (2, 4, NCOFF, 2))

    pool = flat.reshape(2, 1, env.state_dim).repeat(1, 2, 1)
    gathered_flat, gathered_done = env.gather_step_flat(
        pool,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        target,
    )
    direct_flat, direct_done = env.step_flat(flat, target)
    record("point env tree-pool gather step matches direct step",
           torch.allclose(gathered_flat, direct_flat, atol=1e-12)
           and torch.equal(gathered_done, direct_done))

    late_flat = env.pack_state(env.initial_coefficients(positions), time=torch.tensor([0.2, 0.25]))
    _, late_done = env.step_flat(late_flat, target)
    record("point env emits done when time horizon is reached",
           torch.equal(late_done, torch.tensor([True, True])))


def test_point_env_collision_and_bounds_done() -> None:
    env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        collision_radius=1.0,
        tf=80.0,
        dtype=torch.float64,
        position_bounds=((-2.0, 2.0), (-2.0, 2.0)),
    )
    collision_pos = torch.tensor([
        [0.0, 0.0],
        [0.0, 2.0],
        [0.5, 0.0],
        [1.8, 1.8],
    ], dtype=torch.float64)
    collision_flat = env.initial_flat_state(collision_pos)
    _, collision_done, collision_info = env.step_flat(
        collision_flat,
        collision_pos,
        clamp_target=False,
        return_info=True,
    )
    record("point env deactivates cross-team point collisions",
           torch.equal(collision_info.collision_mask, torch.tensor([True, False, True, False]))
           and torch.equal(collision_info.active.bool(), torch.tensor([False, True, False, True]))
           and not bool(collision_done))

    one_to_one_pos = torch.tensor([
        [0.0, 0.0],
        [1.8, 1.8],
        [0.2, 0.0],
        [0.3, 0.0],
    ], dtype=torch.float64)
    one_to_one_flat = env.initial_flat_state(one_to_one_pos)
    _, _, one_to_one_info = env.step_flat(
        one_to_one_flat,
        one_to_one_pos,
        clamp_target=False,
        return_info=True,
    )
    record("point env collision matching is one-to-one by nearest pair",
           int(one_to_one_info.collision_mask.sum()) == 2)

    eliminate_pos = torch.tensor([
        [0.0, 0.0],
        [0.2, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
    ], dtype=torch.float64)
    eliminate_flat = env.initial_flat_state(eliminate_pos)
    _, eliminate_done, eliminate_info = env.step_flat(
        eliminate_flat,
        eliminate_pos,
        clamp_target=False,
        return_info=True,
    )
    record("point env done when a team is eliminated by collisions",
           bool(eliminate_done)
           and not eliminate_info.active.bool()[:2].any()
           and not eliminate_info.active.bool()[2:].any())

    bounds_pos = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 1.0],
    ], dtype=torch.float64)
    bounds_flat = env.initial_flat_state(bounds_pos)
    _, bounds_done, bounds_info = env.step_flat(
        bounds_flat,
        bounds_pos,
        clamp_target=False,
        return_info=True,
    )
    record("point env done on out-of-bounds point",
           bool(bounds_done)
           and not bool(bounds_info.valid)
           and torch.equal(bounds_info.out_of_bounds_mask, torch.tensor([False, False, True, False])))


def test_point_env_convex_obstacle_geometry() -> None:
    square_ccw = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
    square_cw = list(reversed(square_ccw))
    triangle = [
        [2.0, 0.0],
        [3.0, 0.0],
        [2.5, 1.0],
    ]
    env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        collision_radius=0.0,
        obstacle_vertices=[square_ccw, square_cw, triangle],
        obstacle_collision_margin=0.0,
        tf=80.0,
        dtype=torch.float64,
        position_bounds=((-10.0, 10.0), (-10.0, 10.0)),
    )

    points = torch.tensor([
        [0.5, 0.5],    # inside both square orientations
        [1.5, 0.5],    # outside square
        [2.5, 0.25],   # inside triangle
        [3.5, 0.5],    # outside all obstacles
    ], dtype=torch.float64)
    flat = env.initial_flat_state(points)
    _, done, info = env.step_flat(
        flat,
        points,
        clamp_target=False,
        return_info=True,
    )
    clearance = env.obstacle_clearance(points.reshape(1, 4, 2))[0]

    record("convex obstacle half-plane clearance handles cw and ccw polygons",
           clearance[0, 0] <= 0.0
           and clearance[0, 1] <= 0.0
           and clearance[1, 0] > 0.0
           and clearance[2, 2] <= 0.0
           and clearance[3].min() > 0.0)
    record("point env emits obstacle collision and done masks",
           torch.equal(info.obstacle_collision_mask, torch.tensor([True, False, True, False]))
           and torch.equal(info.collision_mask, info.obstacle_collision_mask)
           and bool(done))

    margin_env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        collision_radius=0.0,
        obstacle_vertices=[square_ccw],
        obstacle_collision_margin=0.15,
        dtype=torch.float64,
        position_bounds=((-10.0, 10.0), (-10.0, 10.0)),
    )
    near_edge = torch.tensor([
        [1.10, 0.50],
        [1.25, 0.50],
        [2.00, 2.00],
        [-1.00, -1.00],
    ], dtype=torch.float64)
    near_mask = margin_env.obstacle_collision_mask(
        near_edge.reshape(1, 4, 2),
        torch.ones((1, 4), dtype=torch.float64),
    )[0]
    record("convex obstacle margin catches near-edge points",
           torch.equal(near_mask, torch.tensor([True, False, False, False])))


def test_point_env_obstacle_batch_performance() -> None:
    env = MincoPointEnvTransition(
        num_agents=4,
        piece_t=0.1,
        dt=0.1,
        collision_radius=0.0,
        obstacle_vertices=[
            [[-2.0, -2.0], [-1.0, -2.0], [-1.0, -1.0], [-2.0, -1.0]],
            [[1.0, -2.0], [2.0, -2.0], [2.0, -1.0], [1.0, -1.0]],
            [[-2.0, 1.0], [-1.0, 1.0], [-1.0, 2.0], [-2.0, 2.0]],
            [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
        ],
        obstacle_collision_margin=0.05,
        dtype=torch.float32,
        position_bounds=((-5.0, 5.0), (-5.0, 5.0)),
        done_on_team_eliminated=False,
    )
    batch = 4096
    base = torch.linspace(-4.0, 4.0, batch, dtype=torch.float32)
    positions = torch.stack([
        torch.stack((base, torch.zeros_like(base)), dim=-1),
        torch.stack((base, torch.ones_like(base) * 0.5), dim=-1),
        torch.stack((base, torch.ones_like(base) * -0.5), dim=-1),
        torch.stack((base, torch.ones_like(base)), dim=-1),
    ], dim=1)
    flat = env.initial_flat_state(positions)
    target = positions + torch.tensor([0.01, 0.0], dtype=torch.float32)

    # Warmup keeps one-time dispatch overhead out of the measured smoke result.
    env.step_flat(flat, target)
    t0 = time.perf_counter()
    next_flat, done, info = env.step_flat(flat, target, return_info=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    transitions_per_s = batch / max(elapsed_ms / 1000.0, 1e-12)

    ok = (
        next_flat.shape == (batch, env.state_dim)
        and done.shape == (batch,)
        and info.obstacle_collision_mask.shape == (batch, 4)
        and transitions_per_s > 1.0
    )
    record("batched obstacle step performance smoke",
           ok,
           f"{elapsed_ms:.3f} ms for {batch} states ({transitions_per_s:.0f}/s)")


def main() -> None:
    test_matrices_and_one_step()
    test_bounds_batch_and_flat_state()
    test_pool_gather_and_rollout()
    test_point_env_time_pack_and_gather()
    test_point_env_collision_and_bounds_done()
    test_point_env_convex_obstacle_geometry()
    test_point_env_obstacle_batch_performance()
    print("================================================================")
    print("  ALL PASSED: 20 checks")
    print("================================================================")


if __name__ == "__main__":
    main()
