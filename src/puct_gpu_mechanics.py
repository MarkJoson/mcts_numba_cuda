"""
puct_gpu_mechanics.py
---------------------
Device functions that define the mechanics of a PUCT problem for GPU kernels.

This mirrors the role of ``mctsnc_game_mechanics.py`` but targets
continuous-state, continuous-action problems.  Currently implements a
**double integrator** as a canonical demo problem:

    state  = [position, velocity]          (state_dim = 2)
    action = [acceleration]                (action_dim = 1)
    dynamics:
        v' = v + a * dt
        p' = p + v * dt + 0.5 * a * dt²
    terminal: |position| > 10  or  steps > max_steps
    reward  : 1 - |position| / 10          (normalised, per robot — single)

For real problems, replace ``step``, ``is_terminal``, and
``normalized_reward`` with problem-specific logic, keeping the same
signatures.

Notes
-----
All functions are decorated with ``@cuda.jit(device=True)`` so that they can
be called **from inside** other ``@cuda.jit`` kernels, but are NOT themselves
launchable as kernels.
"""

import math
from numba import cuda, float32, int8, int16, int32, boolean

# ── double integrator parameters ────────────────────────────────────────────
DI_DT = 0.05          # timestep [s]
DI_POS_LIMIT = 10.0   # |position| beyond this → terminal
DI_MAX_STEPS = 200    # depth limit used by caller; not enforced here


@cuda.jit(device=True)
def step(state, action_idx, action_params, dt, next_state):
    """
    Compute ``next_state = f(state, action, dt)`` for the double integrator.

    Parameters
    ----------
    state : device array, shape (state_dim,) float32
        Current state [position, velocity].
    action_idx : int32
        Discrete action index (unused for continuous problems; provided for
        interface consistency).
    action_params : device array, shape (action_dim,) float32
        Continuous action parameters.  action_params[0] = acceleration.
    dt : float32
        Timestep.
    next_state : device array, shape (state_dim,) float32
        Output — next state (written in-place).
    """
    pos = state[0]
    vel = state[1]
    acc = action_params[0]
    next_state[1] = vel + acc * dt
    next_state[0] = pos + vel * dt + float32(0.5) * acc * dt * dt


@cuda.jit(device=True)
def is_terminal(state):
    """
    Return ``True`` if *state* is a terminal state.

    Parameters
    ----------
    state : device array, shape (state_dim,) float32
    """
    pos = state[0]
    return pos > DI_POS_LIMIT or pos < -DI_POS_LIMIT


@cuda.jit(device=True)
def normalized_reward(state, action_idx, action_params, reward_out):
    """
    Compute per-robot normalised reward vector for a (state, action) pair.

    For the double integrator (single robot), the reward is
    ``1 - |position| / POS_LIMIT``, clipped to [0, 1].

    Parameters
    ----------
    state : device array, shape (state_dim,) float32
    action_idx : int32
        Discrete action index (for interface consistency).
    action_params : device array, shape (action_dim,) float32
    reward_out : device array, shape (num_robots,) float32
        Output reward per robot (written in-place).
    """
    pos = state[0]
    r = float32(1.0) - math.fabs(pos) / float32(DI_POS_LIMIT)
    if r < float32(0.0):
        r = float32(0.0)
    reward_out[0] = r
