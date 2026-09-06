"""Generate the single-episode MPC operating overview described in the README.

Example:
    source .venv/bin/activate && source env.sh
    python plots/plot_episode_overview.py --seed 0 --output artifacts/mpc-overview.png
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
from mpc.model import Agent, N_MODULES, PER_MODULE  # noqa: E402


def module_values(observation: np.ndarray, offset: int, center: float, scale: float) -> np.ndarray:
    return np.asarray(
        [observation[module * PER_MODULE + offset] * scale + center for module in range(N_MODULES)]
    )


def run_episode(seed: int, horizon: int, scenario: str) -> dict[str, np.ndarray]:
    env = gym.make("LhcbCooling-v0", scenario=scenario, horizon=horizon, ramp_max=25.0)
    agent = Agent(ROOT / "mpc")
    observation, _ = env.reset(seed=seed)
    rows: dict[str, list[np.ndarray | float | bool]] = {
        "load": [], "outdoor": [], "wet_bulb": [], "fan": [], "water": [],
        "energy": [], "inlet": [], "max_inlet": [], "adiabatic": [],
    }
    for _ in range(horizon):
        rows["load"].append(module_values(observation, 3, 400.0, 150.0))
        rows["outdoor"].append(float(observation[55] * 10.0 + 15.0))
        rows["wet_bulb"].append(float(observation[57] * 8.0 + 12.0))
        agent.act(observation)
        assert agent.last_trace is not None
        applied = agent.last_trace["applied_action"]
        rows["fan"].append(applied[0::3])
        rows["water"].append(applied[2::3])
        rows["adiabatic"].append(bool(agent.last_trace["adiabatic"]))
        observation, _, terminated, truncated, info = env.step(agent.last_trace["desired_action"])
        rows["energy"].append(float(info["energy_kwh_per_min"]))
        # These post-step readings are the realized rack-inlet temperatures
        # resulting from the action just applied above.
        rows["inlet"].append(module_values(observation, 0, 27.0, 4.0))
        rows["max_inlet"].append(float(info["max_inlet_c"]))
        if terminated or truncated:
            break
    return {key: np.asarray(value) for key, value in rows.items()}


def shade_adiabatic(ax: plt.Axes, adiabatic: np.ndarray) -> None:
    start: int | None = None
    for minute, enabled in enumerate(np.r_[adiabatic, False]):
        if enabled and start is None:
            start = minute
        elif not enabled and start is not None:
            ax.axvspan(start, minute, color="#50b8a0", alpha=0.12, lw=0)
            start = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--dispatch-output", type=Path, default=Path("artifacts/mpc-dispatch.png"))
    parser.add_argument("--outcomes-output", type=Path, default=Path("artifacts/mpc-outcomes.png"))
    args = parser.parse_args()
    data = run_episode(args.seed, args.horizon, args.scenario)
    minute = np.arange(len(data["energy"]))
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)

    for module in range(N_MODULES):
        axes[0].plot(minute, data["load"][:, module], color=colors[module], label=f"Module {module + 1}")
    axes[0].set(ylabel="IT load [kW]", title=f"MPC cooling dispatch — seed {args.seed}")
    axes[0].legend(ncol=5, fontsize=9, loc="upper left")
    axes[1].plot(minute, data["outdoor"], color="#d55e00", label="Outdoor")
    axes[1].plot(minute, data["wet_bulb"], color="#0072b2", label="Wet bulb")
    axes[1].axhline(20, color="0.35", ls="--", lw=1, label="Adiabatic threshold")
    axes[1].set(ylabel="Temperature [°C]")
    axes[1].legend(ncol=3, fontsize=9, loc="upper left")
    for module in range(N_MODULES):
        axes[2].plot(minute, data["fan"][:, module], color=colors[module], label=f"Module {module + 1}")
        axes[3].plot(minute, data["water"][:, module], color=colors[module])
    axes[2].set(ylabel="Outside fan [%]", ylim=(15, 105))
    axes[3].set(ylabel="Water [%]", ylim=(-3, 83))
    for axis in axes:
        shade_adiabatic(axis, data["adiabatic"])
        axis.grid(axis="y", alpha=0.25)
    axes[3].set_xlabel("Minute")
    axes[1].text(0.995, 0.08, "green shading: adiabatic mode", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=9, color="#267c69")
    args.dispatch_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.dispatch_output, dpi=180)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    for module in range(N_MODULES):
        axes[0].plot(minute, data["inlet"][:, module], color=colors[module], label=f"Module {module + 1}")
    axes[0].plot(minute, data["max_inlet"], color="black", lw=1.6, ls="--", label="Maximum")
    axes[0].axhline(27.0, color="0.35", lw=1, ls=":", label="27 °C target")
    axes[0].axhline(35.0, color="#d55e00", lw=1, ls=":", label="35 °C ceiling")
    # This episode's inlet temperatures are tightly regulated at the 27 °C
    # target. A fixed micro-range makes those small module differences legible
    # while still using absolute Celsius labels.
    axes[0].set(
        ylabel="Rack inlet [°C]",
        ylim=(27.0, 27.0003),
        title=f"MPC thermal and energy outcomes — seed {args.seed}",
    )
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend(ncol=4, fontsize=8, loc="upper left")
    axes[1].plot(minute, data["energy"], color="#4d4d4d")
    axes[1].set(ylabel="Energy [kWh/min]", xlabel="Minute")
    for axis in axes:
        shade_adiabatic(axis, data["adiabatic"])
        axis.grid(axis="y", alpha=0.25)
    args.outcomes_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outcomes_output, dpi=180)
    print(f"Wrote {args.dispatch_output}\nWrote {args.outcomes_output}")


if __name__ == "__main__":
    main()
