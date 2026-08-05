"""Train SAC to make small, safe corrections to ``mpc.model.Agent``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import SAC

from mpc.model import Agent as DispatchAgent
from residual_sac.residual_env import ResidualVecEnv
from xdt.dt.envs.vec_cooling_env import BatchedCoolingVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--max-correction", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("residual_sac/weights/sac"))
    args = parser.parse_args()

    base_env = BatchedCoolingVecEnv(
        num_envs=args.envs,
        scenario="mixed",
        horizon=args.horizon,
        ramp_max=25.0,
        device=args.device,
        seed=args.seed,
    )
    env = ResidualVecEnv(
        base_env, lambda: DispatchAgent(ROOT / "mpc"), args.max_correction
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        buffer_size=500_000,
        learning_starts=args.learning_starts,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto_0.001",
        policy_kwargs={"net_arch": [128, 128]},
        device=args.device,
        seed=args.seed,
        verbose=1,
    )
    # Start from exact MPC behavior, with only small stochastic exploration.
    torch.nn.init.zeros_(model.policy.actor.mu.weight)
    torch.nn.init.zeros_(model.policy.actor.mu.bias)
    torch.nn.init.zeros_(model.policy.actor.log_std.weight)
    torch.nn.init.constant_(model.policy.actor.log_std.bias, -2.5)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved Stable-Baselines3 SAC checkpoint to {args.out}.zip")


if __name__ == "__main__":
    main()
