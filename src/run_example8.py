"""
run_example8.py
---------------
Self-contained GPU PUCT runner for the example8 pursuit-evasion scenario.

This script has ZERO dependency on decision_making. Everything is resolved
within the mcts_numba_cuda project:
  - examples/example8_problem.py  : pursuit-evasion game logic
  - examples/example8_adapter.py  : GPU PUCT ↔ problem adapter
  - puct_gpu.py / puct_gpu_kernels.py : GPU PUCT engine

Usage
-----
    cd mcts_numba_cuda/src
    python run_example8.py                         # 1 trial, 200 sims
    python run_example8.py --trials 5              # 5 random episodes
    python run_example8.py --simulations 1000 --movie
    python run_example8.py --diagnostics --movie   # full analysis
"""

import sys
import os
import time
import argparse
import numpy as np

# Force headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import animation, cm

# ── Path setup (only mcts_numba_cuda/src) ────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from examples.example8_problem import Example8
from examples.example8_adapter import GPU_PUCT_Adapter


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GPU PUCT — Example8 Pursuit-Evasion (self-contained)"
    )
    # Solver parameters
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--n-trees", type=int, default=8)
    p.add_argument("--max-actions", type=int, default=32)
    p.add_argument("--search-depth", type=int, default=100)
    p.add_argument("--device-memory", type=float, default=1.0)
    # PUCT hyperparams
    p.add_argument("--C-pw", type=float, default=2.0)
    p.add_argument("--alpha-pw", type=float, default=0.5)
    p.add_argument("--C-exp", type=float, default=1.0)
    p.add_argument("--alpha-exp", type=float, default=0.25)
    # Simulation
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    # Output
    p.add_argument("--output-dir", default=None)
    p.add_argument("--movie", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting (self-contained, no decision_making plotter dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_n_colors(n, cmap=None):
    cm_subsection = np.linspace(0, 1, n)
    if cmap is None:
        cmap = cm.tab20
    return [cmap(x) for x in cm_subsection]


def plot_sim_result(sim_result, problem):
    """Plot states, actions, and rewards vs time (matches decision_making plotter)."""
    times = sim_result["times"]
    states = sim_result["states"]
    actions = sim_result["actions"]
    rewards = sim_result["rewards"]
    num_robots = problem.num_robots
    robot_state_dims = [len(sidx) for sidx in problem.state_idxs]
    action_dim = actions.shape[1] if actions.ndim >= 2 else problem.action_dim

    ncols = max(max(robot_state_dims), action_dim, 2, num_robots + 1)
    fig, axs = plt.subplots(
        nrows=num_robots + 2, ncols=ncols, figsize=(4 * ncols, 3 * (num_robots + 2))
    )

    for i_robot in range(num_robots):
        robot_state_idx = problem.state_idxs[i_robot]
        for i_ax, i_state in enumerate(robot_state_idx):
            axs[i_robot, i_ax].plot(times, states[:, i_state])
            axs[i_robot, i_ax].set_ylim(
                problem.state_lims[i_state, 0], problem.state_lims[i_state, 1]
            )
        axs[i_robot, 0].set_ylabel(f"Robot State {i_robot}")

    for i_action in range(action_dim):
        axs[num_robots, i_action].plot(times[1:], actions[:, i_action])
        axs[num_robots, i_action].set_ylim(
            problem.action_lims[i_action, 0], problem.action_lims[i_action, 1]
        )
    axs[num_robots, 0].set_ylabel("Actions")

    for i_robot in range(num_robots):
        axs[num_robots + 1, 0].plot(times[1:], rewards[:, i_robot])
        axs[num_robots + 1, 1].plot(times[1:], np.cumsum(rewards[:, i_robot]))
    axs[num_robots + 1, 0].set_ylabel("Rewards")


def save_figs(filename):
    """Save all open matplotlib figures to PDF and close them."""
    file_dir = os.path.dirname(filename)
    if file_dir and not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    pp = PdfPages(filename)
    for i in plt.get_fignums():
        pp.savefig(plt.figure(i))
        plt.close(plt.figure(i))
    pp.close()


def make_movie(sim_result, problem, filename):
    """Generate animated GIF of an episode."""
    states = np.asarray(sim_result["states"])
    actions = np.asarray(sim_result["actions"])
    if hasattr(problem, "time_idx"):
        times = states[:, int(problem.time_idx)].astype(float)
    else:
        times = np.asarray(sim_result["times"]).ravel()

    fig, ax = plt.subplots()

    target_fps = 30
    total_time = times[-1] - times[0]
    if total_time <= 1e-12:
        num_frames = 2
        effective_fps = target_fps
    else:
        effective_fps = float(target_fps)
        num_frames = max(2, int(total_time * effective_fps) + 1)
    new_times = np.linspace(times[0], times[-1], num_frames)
    print(f"  making movie: {num_frames} frames at {effective_fps:.0f} fps")

    def frame_render_history(frame_time):
        if frame_time <= times[0] + 1e-12:
            return states[:1], actions[:1]
        if frame_time >= times[-1] - 1e-12:
            return states, actions

        idx = int(np.searchsorted(times, frame_time, side='right') - 1)
        idx = max(0, min(idx, len(times) - 2))
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        dt = max(t1 - t0, 1e-8)
        local_t = float(frame_time - t0)
        history = states[:idx + 1]
        action_history = actions[:min(idx + 1, len(actions))]

        if local_t <= 1e-12 or idx >= len(actions):
            return history, action_history

        state0 = states[idx].reshape((-1, 1))
        action0 = actions[idx].reshape((-1, 1))
        if hasattr(problem, "interpolate_state"):
            state_t = problem.interpolate_state(state0, action0, local_t, dt)
        else:
            alpha = local_t / dt
            state1 = states[idx + 1].reshape((-1, 1))
            state_t = (1.0 - alpha) * state0 + alpha * state1
        return np.vstack((history, state_t.reshape((1, -1)))), action_history

    def animate(i_t):
        ax.clear()
        ax.grid(True)
        frame_time = float(new_times[i_t])
        states_i, actions_i = frame_render_history(frame_time)
        problem.render(states=states_i, actions=actions_i, fig=fig, ax=ax)
        ax.text(0.02, 0.98, f"t = {frame_time:.2f}s",
                transform=ax.transAxes, va='top')

    interval = 1000.0 / effective_fps
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval)
    writer = animation.PillowWriter(fps=effective_fps)
    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".gif"
    tmp_filename = root + ".tmp" + ext

    last_report = {"frame": -1}
    def progress_callback(frame_idx, _total):
        if frame_idx == 0 or frame_idx == num_frames - 1 or frame_idx - last_report["frame"] >= 50:
            print(f"  rendered frame {frame_idx + 1}/{num_frames}")
            last_report["frame"] = frame_idx

    anim.save(tmp_filename, writer=writer, dpi=80, progress_callback=progress_callback)
    os.replace(tmp_filename, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(problem, solver, verbose=True):
    """Run one full episode. Returns sim_result dict."""
    state = problem.initialize()
    times, states, actions, rewards, step_durations = [], [], [], [], []

    states.append(state)
    times.append(float(state[problem.time_idx, 0]))

    for step, _ in enumerate(problem.times[1:]):
        t0 = time.time()
        action = solver.policy(problem, state)
        dt_policy = time.time() - t0

        reward = problem.reward(state, action)
        next_state = problem.step(state, action, problem.dt)
        done = problem.is_terminal(next_state)

        times.append(float(next_state[problem.time_idx, 0]))
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        step_durations.append(dt_policy)

        if verbose:
            r_str = "  ".join(f"{float(v):>+.4f}" for v in reward.flatten())
            a_str = "  ".join(f"{x:+.3f}" for x in action.flatten()[:4])
            tag = " ← TERMINAL" if done else ""
            print(f"  {step+1:>4}  [{r_str}]  [{a_str}]{tag}  ({dt_policy:.3f}s)")

        if done:
            break
        state = next_state

    # Reshape for consistency with plotter
    states_arr = np.array([s.flatten() for s in states])   # (T+1, state_dim)
    actions_arr = np.array([a.flatten() for a in actions])  # (T, action_dim)
    rewards_arr = np.array([r.flatten() for r in rewards])  # (T, num_robots)

    return {
        "times": times,
        "states": states_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "step_durations": step_durations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    output_dir = args.output_dir or os.path.join(_SCRIPT_DIR, "..", "outputs", "example8")
    os.makedirs(output_dir, exist_ok=True)

    # ── Problem (self-contained) ─────────────────────────────────────────────
    problem = Example8()
    problem.use_minco_rollout = True
    if args.diagnostics:
        problem.set_visualization_detail(True)

    print(f"\n{'='*65}")
    print(f"  GPU PUCT — Example8 Pursuit-Evasion (self-contained)")
    print(f"{'='*65}")
    print(f"  state_dim={problem.state_dim}, action_dim={problem.action_dim}, "
          f"num_robots={problem.num_robots}")
    print(f"  dt={problem.dt}, tf={problem.tf}, timesteps={len(problem.times)}")

    # ── Solver (self-contained) ──────────────────────────────────────────────
    solver = GPU_PUCT_Adapter(
        policy_oracle=[None],
        value_oracle=None,
        search_depth=args.search_depth,
        number_simulations=args.simulations,
        C_pw=args.C_pw,
        alpha_pw=args.alpha_pw,
        C_exp=args.C_exp,
        alpha_exp=args.alpha_exp,
        beta_policy=0.0,
        beta_value=0.0,
        max_actions=args.max_actions,
        n_trees=args.n_trees,
        device_memory=args.device_memory,
    )

    print(f"  Solver: {solver}")
    print(f"  Trials: {args.trials}")
    print(f"{'='*65}\n")

    # ── Run episodes ─────────────────────────────────────────────────────────
    sim_results = []
    for trial in range(args.trials):
        print(f"\n{'─'*50}")
        print(f"  Trial {trial+1}/{args.trials}")
        print(f"{'─'*50}")

        sim_result = run_episode(problem, solver, verbose=True)

        n_steps = len(sim_result["actions"])
        labels = {0: "evader-0", 1: "evader-1", 2: "pursuer-0", 3: "pursuer-1"}
        print(f"\n  Trial {trial+1} — {n_steps} steps")
        for r in range(problem.num_robots):
            total_r = float(np.sum(sim_result["rewards"][:, r]))
            print(f"    {labels.get(r, f'robot-{r}'):>10}: reward = {total_r:+.4f}")
        if sim_result["step_durations"]:
            print(f"    avg policy time = {np.mean(sim_result['step_durations']):.3f}s/step")

        sim_results.append(sim_result)

    # ── Plotting ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        print(f"\nGenerating plots → {output_dir}/")

        for sim_result in sim_results:
            plot_sim_result(sim_result, problem)
            problem.render(
                states=sim_result["states"],
                actions=sim_result["actions"],
            )
            if args.diagnostics and hasattr(problem, "plot_run_diagnostics"):
                problem.plot_run_diagnostics(sim_result)

        pdf_path = os.path.join(output_dir, "run.pdf")
        save_figs(pdf_path)
        print(f"  Saved: {pdf_path}")

    # ── Movie ────────────────────────────────────────────────────────────────
    if args.movie and sim_results:
        gif_path = os.path.join(output_dir, "episode.gif")
        print(f"\nGenerating movie → {gif_path}")
        make_movie(sim_results[-1], problem, gif_path)
        print(f"  Saved: {gif_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Summary across {len(sim_results)} trial(s)")
    print(f"{'='*65}")
    labels = {0: "evader-0", 1: "evader-1", 2: "pursuer-0", 3: "pursuer-1"}
    all_rewards = np.array([
        [float(np.sum(sr["rewards"][:, r])) for r in range(problem.num_robots)]
        for sr in sim_results
    ])
    steps = [len(sr["actions"]) for sr in sim_results]
    print(f"  Steps: mean={np.mean(steps):.0f}, min={min(steps)}, max={max(steps)}")
    for r in range(problem.num_robots):
        label = labels.get(r, f"robot-{r}")
        print(f"  {label:>10}: reward = {np.mean(all_rewards[:, r]):+.4f} "
              f"± {np.std(all_rewards[:, r]):.4f}")
    if sim_results[0]["step_durations"]:
        all_dt = [d for sr in sim_results for d in sr["step_durations"]]
        print(f"  Policy time: {np.mean(all_dt):.3f}s ± {np.std(all_dt):.3f}s per step")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
