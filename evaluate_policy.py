"""
Evaluate an LHCb cooling submission or trained policy locally.
IMPORTANT: This script is what i assume the organizers will use to evaluate submissions.
Written accordingly to what it says in the challenge description, and to what i observed in the starting kit.


Examples:
    source env.sh
    .venv/bin/python evaluate_policy.py --policy starting_kit/policy.pt
    .venv/bin/python evaluate_policy.py --agent-dir my_submission
    .venv/bin/python evaluate_policy.py --policy starting_kit/policy.pt --compare-pid
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm

import xdt.dt.envs  # noqa: F401  (registers LhcbCooling-v0)
from xdt.dt.envs.baseline import PIDBaselineController
from xdt.dt.envs.policy import MlpPolicyNet
from xdt.dt.envs.spaces import PER_MODULE_OBS, SHARED_OBS


PUBLIC_SEEDS = 20


@dataclass
class EpisodeResult:
    seed: int
    reward: float
    energy_kwh: float
    peak_inlet_c: float
    steps: int


@dataclass
class Summary:
    total_return: float
    mean_daily_return: float
    total_energy_kwh: float
    mean_daily_energy_kwh: float
    mean_peak_inlet_c: float
    worst_peak_inlet_c: float


def run_episode(env, agent, seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    total_energy = 0.0
    peak_inlet = -np.inf
    steps = 0

    while not done:
        action = np.asarray(agent.act(obs), dtype=np.float32)
        if action.shape != env.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(
                f"invalid action for seed {seed}: expected shape "
                f"{env.action_space.shape}, got {action.shape}, finite="
                f"{np.all(np.isfinite(action))}"
            )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        # The docs define daily energy as this per-minute value accumulated.
        total_energy += float(info["energy_kwh_per_min"])
        peak_inlet = max(peak_inlet, float(info.get("max_inlet_c", -np.inf)))
        steps += 1

    return EpisodeResult(seed, total_reward, total_energy, peak_inlet, steps)


def summarize(name: str, results: list[EpisodeResult]) -> Summary:
    rewards = np.asarray([r.reward for r in results], dtype=np.float64)
    energy = np.asarray([r.energy_kwh for r in results], dtype=np.float64)
    peaks = np.asarray([r.peak_inlet_c for r in results], dtype=np.float64)
    summary = Summary(
        total_return=float(rewards.sum()),
        mean_daily_return=float(rewards.mean()),
        total_energy_kwh=float(energy.sum()),
        mean_daily_energy_kwh=float(energy.mean()),
        mean_peak_inlet_c=float(peaks.mean()),
        worst_peak_inlet_c=float(peaks.max()),
    )

    print(f"\n{name}")
    print("-" * len(name))
    for r in results:
        print(
            f"seed={r.seed:2d} reward={r.reward:10.2f} "
            f"energy={r.energy_kwh:8.2f} peak_inlet={r.peak_inlet_c:6.2f}C "
            f"steps={r.steps}"
        )
    print(
        f"total_return={summary.total_return:.2f} "
        f"mean_daily_return={summary.mean_daily_return:.2f} "
        f"total_energy_kwh={summary.total_energy_kwh:.2f} "
        f"mean_daily_energy_kwh={summary.mean_daily_energy_kwh:.2f} "
        f"mean_peak_inlet={summary.mean_peak_inlet_c:.2f}C "
        f"worst_peak_inlet={summary.worst_peak_inlet_c:.2f}C"
    )
    return summary


def load_policy(path: str) -> MlpPolicyNet:
    spec = torch.load(path, map_location="cpu")
    return MlpPolicyNet.from_spec(spec)


def load_agent(agent_dir: str):
    model_path = os.path.join(agent_dir, "model.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"no model.py found in {agent_dir!r}")

    module_name = "_lhcb_submission_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {model_path}")

    old_path = list(sys.path)
    sys.path.insert(0, agent_dir)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path = old_path

    return module.Agent(agent_dir)


def make_pid_agent(obs_dim: int) -> PIDBaselineController:
    n_modules = (obs_dim - len(SHARED_OBS)) // len(PER_MODULE_OBS)
    return PIDBaselineController(n_modules=n_modules)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--policy", default=None)
    source.add_argument("--agent-dir", default=None)
    parser.add_argument("--seeds", type=int, default=PUBLIC_SEEDS)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--compare-pid", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        "LhcbCooling-v0",
        scenario=args.scenario,
        horizon=args.horizon,
        ramp_max=25.0,
    )

    if args.agent_dir is not None:
        agent_name = f"Agent: {args.agent_dir}"
        agent = load_agent(args.agent_dir)
    else:
        policy_path = args.policy or "starting_kit/policy.pt"
        agent_name = f"Policy: {policy_path}"
        agent = load_policy(policy_path)

    policy_results = [
        run_episode(env, agent, seed)
        for seed in tqdm(range(args.seeds), desc=agent_name, unit="episode")
    ]
    policy_summary = summarize(agent_name, policy_results)

    if args.compare_pid:
        pid = make_pid_agent(env.observation_space.shape[0])
        pid_results = [
            run_episode(env, pid, seed)
            for seed in tqdm(range(args.seeds), desc="PID baseline", unit="episode")
        ]
        pid_summary = summarize("PID baseline", pid_results)

        print("\nPolicy vs PID")
        print("-------------")
        print(
            f"mean_daily_return_delta="
            f"{policy_summary.mean_daily_return - pid_summary.mean_daily_return:.2f}"
        )
        print(
            f"energy_saved_vs_pid_kwh="
            f"{pid_summary.total_energy_kwh - policy_summary.total_energy_kwh:.2f}"
        )
        print(
            f"mean_peak_inlet_delta_c="
            f"{policy_summary.mean_peak_inlet_c - pid_summary.mean_peak_inlet_c:.2f}"
        )
        print(
            f"beats_pid={policy_summary.mean_daily_return > pid_summary.mean_daily_return}"
        )


if __name__ == "__main__":
    main()
