"""Stable-Baselines3 wrapper for safe residual control of the MPC policy."""

from __future__ import annotations

from typing import Callable

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

# Keep the dispatcher in one place.  This experiment deliberately builds on
# the production MPC policy instead of carrying a second copy of it.
from mpc.model import (
    FAN_FLOOR_FRAC,
    HEAD_MAX_FACTOR,
    HEAD_MIN_FACTOR,
    HEAD_REF_C,
    HIGH,
    LOW,
    N_MODULES,
    PER_MODULE,
    Q_MAX_KW,
    Agent as DispatchAgent,
    _load_fits,
    _modules,
    _required_fan,
)

RAW_OBS_DIM = 58
CONTEXT_OBS_DIM = RAW_OBS_DIM + 3 * (3 * N_MODULES) + 1


def _apply_constraints(
    action: np.ndarray, previous: np.ndarray | None, adiabatic: bool
) -> np.ndarray:
    """Match the simulator's action bounds, ramp limit, and water interlock."""
    applied = np.clip(action, LOW, HIGH)
    if previous is not None:
        applied = np.clip(applied, previous - 25.0, previous + 25.0)
        applied = np.clip(applied, LOW, HIGH)
    applied = applied.copy()
    if not adiabatic:
        applied[2::3] = 0.0
    return applied


def policy_observation(
    observation: np.ndarray,
    base_action: np.ndarray,
    previous_action: np.ndarray | None,
    step: int,
) -> np.ndarray:
    """Expose the MPC action and ramp state to the residual policy."""
    base = np.asarray(base_action, dtype=np.float32)
    previous = (
        base
        if previous_action is None
        else np.asarray(previous_action, dtype=np.float32)
    )
    midpoint = ((LOW + HIGH) / 2.0).astype(np.float32)
    half_range = ((HIGH - LOW) / 2.0).astype(np.float32)
    return np.concatenate(
        (
            np.asarray(observation, dtype=np.float32),
            (base - midpoint) / half_range,
            (previous - midpoint) / half_range,
            (base - previous) / 25.0,
            np.asarray([np.clip(step / 479.0, 0.0, 1.0)], dtype=np.float32),
        )
    )


def _capacity(fan: float, water: float, outdoor: float, wet_bulb: float) -> float:
    ambient = outdoor - water / 80.0 * (outdoor - wet_bulb)
    head = np.clip((27.0 - ambient) / HEAD_REF_C, HEAD_MIN_FACTOR, HEAD_MAX_FACTOR)
    airflow = FAN_FLOOR_FRAC + fan / 100.0 * (1.0 - FAN_FLOOR_FRAC)
    return float(Q_MAX_KW * head * airflow)


def _economic_objective(
    observation: np.ndarray, action: np.ndarray, module: int
) -> float:
    """The dispatcher's immediate cooling-power plus water objective."""
    obs = np.asarray(observation, dtype=np.float64)
    i = 3 * module
    outdoor = float(obs[55] * 10.0 + 15.0)
    humidity = float(obs[56] * 20.0 + 65.0)
    wet_bulb = float(obs[57] * 8.0 + 12.0)
    load_kw = float(obs[module * PER_MODULE + 3] * 150.0 + 400.0)
    return_c = float(obs[module * PER_MODULE + 1] * 8.0 + 45.0)
    features = np.asarray(
        [
            [
                load_kw * 1000.0,
                return_c,
                outdoor,
                humidity,
                wet_bulb,
                action[i],
                action[i + 1],
                action[i + 2],
            ]
        ],
        dtype=np.float64,
    )
    power_w = max(float(_load_fits()[module]["adiabatic"].predict(features)[0]), 0.0)
    return power_w / 1_000_000.0 + 0.0004 * float(action[i + 2])


def project_action(
    observation: np.ndarray,
    base_action: np.ndarray,
    correction: np.ndarray,
    previous_action: np.ndarray | None,
) -> np.ndarray:
    """Convert five water corrections into capacity-preserving MPC actions."""
    base = np.asarray(base_action, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if np.all(correction == 0.0):
        return base.copy()

    obs = np.asarray(observation, dtype=np.float64)
    outdoor = float(obs[55] * 10.0 + 15.0)
    if outdoor < 20.0:
        return base.copy()
    wet_bulb = float(obs[57] * 8.0 + 12.0)
    loads = _modules(obs, 3, 400.0, 150.0)
    base_applied = _apply_constraints(base, previous_action, adiabatic=True)
    projected = base.copy()

    for module in range(N_MODULES):
        i = 3 * module
        water = float(np.clip(base[i + 2] + correction[module], 0.0, 80.0))
        if previous_action is not None:
            water = float(
                np.clip(
                    water, previous_action[i + 2] - 25.0, previous_action[i + 2] + 25.0
                )
            )
        base_capacity = _capacity(
            base_applied[i], base_applied[i + 2], outdoor, wet_bulb
        )
        fan = max(
            float(
                _required_fan(
                    np.array([loads[module]]), np.array([water]), outdoor, wet_bulb
                )[0]
            ),
            float(
                _required_fan(
                    np.array([base_capacity]), np.array([water]), outdoor, wet_bulb
                )[0]
            ),
        )
        projected[i] = np.clip(fan, 20.0, 100.0)
        projected[i + 2] = water
        candidate = _apply_constraints(projected, previous_action, adiabatic=True)
        if (
            _capacity(candidate[i], candidate[i + 2], outdoor, wet_bulb) + 1e-6
            < base_capacity
        ):
            projected[i : i + 3] = base[i : i + 3]
            continue
        # Avoid trading a known immediate loss for a speculative RL gain.  The
        # residual is still free to exploit delayed effects in the simulator.
        if (
            _economic_objective(obs, candidate, module)
            > _economic_objective(obs, base_applied, module) + 1e-10
        ):
            projected[i : i + 3] = base[i : i + 3]
    return projected


class ResidualVecEnv(VecEnvWrapper):
    """Reduce the 15-control environment to five safe residual controls."""

    def __init__(
        self,
        env: VecEnv,
        dispatcher_factory: Callable[[], DispatchAgent],
        max_correction: float = 1.0,
    ) -> None:
        super().__init__(
            env,
            observation_space=spaces.Box(
                -np.inf, np.inf, (CONTEXT_OBS_DIM,), np.float32
            ),
            action_space=spaces.Box(
                -max_correction, max_correction, (N_MODULES,), np.float32
            ),
        )
        self.dispatcher_factory = dispatcher_factory
        self.dispatchers = [dispatcher_factory() for _ in range(env.num_envs)]
        self.raw_observation: np.ndarray | None = None
        self.base_actions: np.ndarray | None = None
        self.pending_applied: np.ndarray | None = None

    def _policy_observations(
        self, raw_observation: np.ndarray, previous: np.ndarray, rampable: np.ndarray
    ) -> np.ndarray:
        bases, observations = [], []
        for i, dispatcher in enumerate(self.dispatchers):
            base = dispatcher.act(raw_observation[i])
            bases.append(base)
            observations.append(
                policy_observation(
                    raw_observation[i],
                    base,
                    previous[i] if rampable[i] else None,
                    dispatcher.steps - 1,
                )
            )
        self.base_actions = np.asarray(bases, dtype=np.float32)
        return np.asarray(observations, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.dispatchers = [self.dispatcher_factory() for _ in range(self.num_envs)]
        self.raw_observation = np.asarray(self.venv.reset(), dtype=np.float32)
        return self._policy_observations(
            self.raw_observation,
            np.asarray(self.venv._prev_action, dtype=np.float64),
            np.asarray(self.venv._rampable, dtype=bool),
        )

    def step_async(self, corrections: np.ndarray) -> None:
        assert self.raw_observation is not None and self.base_actions is not None
        previous = np.asarray(self.venv._prev_action, dtype=np.float64)
        rampable = np.asarray(self.venv._rampable, dtype=bool)
        actions = [
            project_action(
                self.raw_observation[i],
                self.base_actions[i],
                corrections[i],
                previous[i] if rampable[i] else None,
            )
            for i in range(self.num_envs)
        ]
        self.pending_applied = np.asarray(
            [
                _apply_constraints(
                    action,
                    previous[i] if rampable[i] else None,
                    float(self.raw_observation[i, 55] * 10.0 + 15.0) >= 20.0,
                )
                for i, action in enumerate(actions)
            ],
            dtype=np.float64,
        )
        self.venv.step_async(np.asarray(actions, dtype=np.float32))

    def step_wait(self):
        raw_observation, rewards, dones, infos = self.venv.step_wait()
        self.raw_observation = np.asarray(raw_observation, dtype=np.float32)
        applied = np.asarray(self.venv._prev_action, dtype=np.float64)
        assert self.pending_applied is not None
        shaped_rewards = np.asarray(rewards, dtype=np.float32).copy()
        for i, done in enumerate(dones):
            # The base reward includes IT load, which the controls cannot
            # influence.  Removing it makes SAC focus its value function on
            # cooling cost, without changing the optimal policy.
            reward_observation = infos[i].get(
                "terminal_observation", self.raw_observation[i]
            )
            cooling_kw = sum(
                float(reward_observation[module * PER_MODULE + 4] * 10.0 + 15.0)
                for module in range(N_MODULES)
            )
            shaped_rewards[i] += (
                max(float(infos[i]["total_power_kw"]) - cooling_kw, 0.0) / 1000.0
            )
            if done:
                terminal_raw = np.asarray(
                    infos[i]["terminal_observation"], dtype=np.float32
                )
                dispatcher = self.dispatchers[i]
                dispatcher.previous_action = self.pending_applied[i].copy()
                terminal_base = dispatcher.act(terminal_raw)
                infos[i]["terminal_observation"] = policy_observation(
                    terminal_raw,
                    terminal_base,
                    self.pending_applied[i],
                    dispatcher.steps - 1,
                )
                self.dispatchers[i] = self.dispatcher_factory()
            else:
                self.dispatchers[i].previous_action = applied[i].copy()
        return (
            self._policy_observations(
                self.raw_observation,
                applied,
                np.asarray(self.venv._rampable, dtype=bool),
            ),
            shaped_rewards,
            dones,
            infos,
        )
