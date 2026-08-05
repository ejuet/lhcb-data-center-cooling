"""Residual-SAC agent around the repository MPC policy."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import SAC

# ``evaluate_policy.py`` imports this file directly, so make the repository
# root discoverable when it is loaded as ``residual_sac/model.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.model import Agent as DispatchAgent
from residual_sac.residual_env import (
    _apply_constraints,
    policy_observation,
    project_action,
)


class Agent:
    """Run a trained five-action SAC residual as a normal 15-action agent."""

    def __init__(self, model_dir: str | Path, weights: str = "weights/sac.zip"):
        model_dir = Path(model_dir)
        self.policy = SAC.load(model_dir / weights, device="cpu")
        self.dispatcher = DispatchAgent(ROOT / "mpc")

    def act(self, observation: np.ndarray) -> np.ndarray:
        # Preserve the state before the dispatcher records its base action.
        previous = (
            None if self.dispatcher.steps >= 480 else self.dispatcher.previous_action
        )
        base = self.dispatcher.act(observation)
        context = policy_observation(
            observation, base, previous, self.dispatcher.steps - 1
        )
        correction, _ = self.policy.predict(context, deterministic=True)
        correction = np.where(np.abs(correction) < 0.05, 0.0, correction)
        action = project_action(observation, base, correction, previous).astype(
            np.float32
        )

        # Plan future MPC actions around what the simulator will actually use,
        # not the uncorrected base action it just produced.
        outdoor = float(observation[55] * 10.0 + 15.0)
        self.dispatcher.previous_action = _apply_constraints(
            action, previous, outdoor >= 20.0
        )
        return action
