"""Show individual MPC plans, their four-step reuse, and applied actions.

Example:
    source .venv/bin/activate && source env.sh
    python plots/plot_receding_horizon.py --seed 0 --module 0

By default the script selects a 48-minute interval with the greatest command
movement. A full episode has around 100 replans, which cannot be read as
overlaid lines; use --start to examine a specific interval instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import xdt.dt.envs  # noqa: F401 - registers LhcbCooling-v0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpc.model import Agent, N_MODULES  # noqa: E402


def collect(seed: int, horizon: int, scenario: str, module: int):
    env = gym.make("LhcbCooling-v0", scenario=scenario, horizon=horizon, ramp_max=25.0)
    agent = Agent(ROOT / "mpc")
    observation, _ = env.reset(seed=seed)
    applied, plans = [], []
    for minute in range(horizon):
        agent.act(observation)
        assert agent.last_trace is not None
        trace = agent.last_trace
        applied.append(trace["applied_action"][3 * module : 3 * module + 3])
        plan = trace["new_plans"][module]
        if plan is not None:
            plans.append((minute, np.asarray(plan)))
        observation, _, terminated, truncated, _ = env.step(trace["desired_action"])
        if terminated or truncated:
            break
    return np.asarray(applied), plans


def busiest_window(applied: np.ndarray, duration: int) -> int:
    """Choose a legible interval where the controller is actively changing."""
    if len(applied) <= duration:
        return 0
    movement = np.abs(np.diff(applied[:, 0], prepend=applied[0, 0]))
    movement += np.abs(np.diff(applied[:, 2], prepend=applied[0, 2]))
    score = np.convolve(movement, np.ones(duration), mode="valid")
    return int(np.argmax(score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--module", type=int, default=0, choices=range(N_MODULES))
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--start", type=int, default=None,
                        help="First minute to display (default: busiest interval)")
    parser.add_argument("--duration", type=int, default=48,
                        help="Number of minutes to display")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mpc-receding-horizon.png"))
    args = parser.parse_args()
    applied, plans = collect(args.seed, args.horizon, args.scenario, args.module)
    duration = min(args.duration, len(applied))
    start = busiest_window(applied, duration) if args.start is None else args.start
    if start < 0 or start >= len(applied):
        parser.error(f"--start must be between 0 and {len(applied) - 1}")
    end = min(start + duration, len(applied))
    minute = np.arange(start, end)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    labels = ((0, "Outside fan [%]"), (2, "Water [%]"))
    for axis, (column, ylabel) in zip(axes, labels):
        action_column = 0 if column == 0 else 2
        # Draw this first so retained-plan markers remain visible even where
        # execution exactly follows the plan (which is the usual case).
        axis.plot(minute, applied[start:end, action_column], color="black", lw=2.4, label="Applied action")
        visible_plans = [item for item in plans if item[0] < end and item[0] + len(item[1]) > start]
        for index, (plan_start, plan) in enumerate(visible_plans):
            x = plan_start + np.arange(len(plan))
            values = plan[:, 0 if column == 0 else 1]
            in_window = (x >= start) & (x < end)
            retained = in_window & (np.arange(len(plan)) < 4)
            color = plt.cm.tab20(index % 20)
            axis.plot(x[in_window], values[in_window], color=color, alpha=0.7, lw=1.4, ls="--")
            axis.plot(
                x[retained], values[retained], color=color, alpha=1.0, lw=3.2,
                marker="o", ms=7, markeredgecolor="white", markeredgewidth=0.8,
            )
        # Legend handles explain encoding without assigning a noisy label to
        # each individual replan.
        axis.plot([], [], color="0.45", ls="--", lw=1.4, label="Forecast tail (per replan)")
        axis.plot([], [], color="0.45", lw=3.2, marker="o", ms=7, label="Four retained actions")
        axis.set(ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper left", ncol=3, fontsize=9)
    axes[0].set_title(
        f"Receding-horizon MPC — module {args.module + 1}, seed {args.seed}, minutes {start}–{end - 1}"
    )
    axes[1].set(xlabel="Minute", ylim=(-3, 83))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output} ({len(plans)} replans captured)")


if __name__ == "__main__":
    main()
