"""
Receding-horizon capacity dispatcher for the LHCb cooling challenge.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Importing the fit's defining module is required by joblib unpickling. The xdt
# package is part of the official evaluation image (the reference agent uses it).
import xdt.dt.models.cooling_learned_nl  # noqa: F401
from xdt.dt.data.ozone import DEFAULT_ARTIFACTS


N_MODULES = 5
PER_MODULE = 11
LOW = np.tile([20.0, 40.0, 0.0], N_MODULES)
HIGH = np.tile([100.0, 100.0, 80.0], N_MODULES)

Q_MAX_KW = 1500.0
FAN_FLOOR_FRAC = 0.35
HEAD_REF_C = 5.5
HEAD_MIN_FACTOR = 0.18
HEAD_MAX_FACTOR = 1.4
CONTAINERS = (2, 3, 4, 5, 6)
_FITS_CACHE: list[dict[str, Any]] | None = None


def _wet_bulb_c(outdoor: float, humidity: float) -> float:
    """Stull wet-bulb approximation used by the evaluation environment."""
    rh = float(np.clip(humidity, 0.0, 100.0))
    return float(
        outdoor * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(outdoor + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def _modules(obs: np.ndarray, offset: int, center: float, scale: float) -> np.ndarray:
    return np.asarray(
        [obs[m * PER_MODULE + offset] * scale + center for m in range(N_MODULES)],
        dtype=np.float64,
    )


def _required_fan(load_kw: np.ndarray, water: np.ndarray, outdoor: float, wet_bulb: float) -> np.ndarray:
    effective_ambient = outdoor - np.clip(water / 80.0, 0.0, 1.0) * (outdoor - wet_bulb)
    head_factor = np.clip((27.0 - effective_ambient) / HEAD_REF_C, HEAD_MIN_FACTOR, HEAD_MAX_FACTOR)
    airflow = load_kw / (Q_MAX_KW * head_factor)
    return 100.0 * (airflow - FAN_FLOOR_FRAC) / (1.0 - FAN_FLOOR_FRAC)


def _load_fits():
    global _FITS_CACHE
    if _FITS_CACHE is not None:
        return _FITS_CACHE
    artifacts = Path(os.environ.get("DT_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS)))
    _FITS_CACHE = []
    for container in CONTAINERS:
        spec = joblib.load(artifacts / f"cooling_it{container}_fullyear.joblib")
        _FITS_CACHE.append({"dry": spec.dry, "adiabatic": spec.adiabatic})
    return _FITS_CACHE


class Agent:
    def __init__(self, model_dir):
        self.previous_action: np.ndarray | None = None
        self.steps = 0
        self.fits = _load_fits()
        self.inside_step = 5.0
        self.ramp_limit = 25.0
        # Keep only distinct disturbance samples. reset() and the first step()
        # expose the same disturbance, so blindly retaining every observation
        # corrupts the first second-order extrapolation.
        self.load_history: list[np.ndarray] = []
        self.return_history: list[np.ndarray] = []
        self.weather_history: list[tuple[float, float]] = []
        self.humidity_history: list[float] = []
        # Reuse a few actions from each 12-step solution.  Re-solving the same
        # slowly changing path every minute dominates CPU time and provides
        # little benefit; current capacity is still checked analytically in
        # act() before an action is returned.
        self.plan_queues: list[list[tuple[float, float]]] = [
            [] for _ in range(N_MODULES)
        ]

    def _economic_action(
        self,
        module: int,
        load_kw: float,
        return_c: float,
        outdoor: float,
        humidity: float,
        wet_bulb: float,
        adiabatic: bool,
    ) -> tuple[float, float, float]:
        """Globally select water and inside fan on the safe-capacity boundary."""
        if self.previous_action is None:
            fan_lo, fan_hi = 20.0, 100.0
            inside_lo, inside_hi = 40.0, 100.0
            water_lo, water_hi = (0.0, 80.0) if adiabatic else (0.0, 0.0)
        else:
            prev = self.previous_action[3 * module : 3 * module + 3]
            fan_lo, fan_hi = max(20.0, prev[0] - self.ramp_limit), min(100.0, prev[0] + self.ramp_limit)
            inside_lo, inside_hi = max(40.0, prev[1] - self.ramp_limit), min(100.0, prev[1] + self.ramp_limit)
            if adiabatic:
                water_lo, water_hi = max(0.0, prev[2] - self.ramp_limit), min(80.0, prev[2] + self.ramp_limit)
            else:
                water_lo = water_hi = 0.0

        waters = np.arange(np.ceil(water_lo), water_hi + 0.1, 1.0)
        if waters.size == 0:
            waters = np.asarray([water_lo])
        fans = _required_fan(np.full(waters.shape, load_kw), waters, outdoor, wet_bulb)
        safe = fans <= fan_hi + 1e-9
        if not np.any(safe):
            # Ramp reachability can make the exact boundary temporarily
            # impossible at a mode transition. Maximise available capacity.
            return fan_hi, inside_lo, water_hi
        waters = waters[safe]
        minimum_fans = np.clip(fans[safe], fan_lo, fan_hi)

        fans = minimum_fans

        inside_values = np.arange(
            np.ceil(inside_lo / self.inside_step) * self.inside_step,
            inside_hi + 0.1,
            self.inside_step,
        )
        if inside_values.size == 0:
            inside_values = np.asarray([inside_lo])
        water_grid = np.repeat(waters, inside_values.size)
        fan_grid = np.repeat(fans, inside_values.size)
        inside_grid = np.tile(inside_values, waters.size)
        n = water_grid.size
        features = np.column_stack(
            [
                np.full(n, load_kw * 1000.0),
                np.full(n, return_c),
                np.full(n, outdoor),
                np.full(n, humidity),
                np.full(n, wet_bulb),
                fan_grid,
                inside_grid,
                water_grid,
            ]
        )
        regime = "adiabatic" if adiabatic else "dry"
        model = self.fits[module][regime]
        power_w = np.maximum(model.predict(features), 0.0)
        # Reward charges electrical power in MW and water at 0.0004 per point.
        objective = power_w / 1_000_000.0 + 0.0004 * water_grid
        best = int(np.argmin(objective))
        return float(fan_grid[best]), float(inside_grid[best]), float(water_grid[best])

    @staticmethod
    def _forecast_from_history(history: list[np.ndarray], offset: int) -> np.ndarray:
        """Extrapolate ``offset`` genuine disturbance samples past the latest.

        ``offset == 0`` returns the latest sample. With three samples the
        formula extends the local quadratic whose one-step form is the original
        ``3*x[t] - 3*x[t-1] + x[t-2]`` dispatcher forecast.
        """
        current = history[-1]
        if offset <= 0 or len(history) == 1:
            return current.copy()
        first = current - history[-2]
        if len(history) < 3:
            return current + float(offset) * first
        second = current - 2.0 * history[-2] + history[-3]
        return current + float(offset) * first + 0.5 * offset * (offset + 1) * second

    def _append_disturbance(
        self,
        load: np.ndarray,
        return_c: np.ndarray,
        outdoor: float,
        wet_bulb: float,
        humidity: float,
    ) -> None:
        """Retain a compact history while suppressing the reset duplicate."""
        duplicate = False
        if self.load_history and self.weather_history and self.humidity_history:
            old_outdoor, old_wet_bulb = self.weather_history[-1]
            duplicate = (
                np.max(np.abs(load - self.load_history[-1])) < 1e-7
                and abs(outdoor - old_outdoor) < 1e-7
                and abs(wet_bulb - old_wet_bulb) < 1e-7
                and abs(humidity - self.humidity_history[-1]) < 1e-7
            )
        if duplicate:
            return
        self.load_history.append(load.copy())
        self.return_history.append(return_c.copy())
        self.weather_history.append((outdoor, wet_bulb))
        self.humidity_history.append(humidity)
        del self.load_history[:-3]
        del self.return_history[:-3]
        del self.weather_history[:-3]
        del self.humidity_history[:-3]

    def _forecast_trajectory(self, count: int) -> list[dict[str, Any]]:
        """Forecast disturbances for the actions about to be applied.

        On the first call, reset's observation describes the same minute as the
        first action, so the first offset is zero. On later calls the observation
        describes the preceding transition and the first offset is one.
        """
        first_offset = 0 if self.steps == 0 else 1
        trajectory: list[dict[str, Any]] = []
        weather_arrays = [np.asarray(v, dtype=np.float64) for v in self.weather_history]
        humidity_arrays = [np.asarray([v], dtype=np.float64) for v in self.humidity_history]
        for i in range(count):
            offset = first_offset + i
            load = self._forecast_from_history(self.load_history, offset)
            return_c = self._forecast_from_history(self.return_history, offset)
            weather = self._forecast_from_history(weather_arrays, offset)
            humidity = float(
                self._forecast_from_history(humidity_arrays, offset)[0]
            )
            outdoor = float(weather[0])
            humidity = float(np.clip(humidity, 0.0, 100.0))
            wet_bulb = _wet_bulb_c(outdoor, humidity)
            trajectory.append(
                {
                    "load": load,
                    "return": return_c,
                    "outdoor": outdoor,
                    "humidity": humidity,
                    "wet_bulb": wet_bulb,
                    "adiabatic": outdoor >= 20.0,
                }
            )
        return trajectory

    def _plan_fan_water(
        self,
        module: int,
        trajectory: list[dict[str, Any]],
        inside: float,
    ) -> list[tuple[float, float]] | None:
        """Solve a short safe fan/water path under both actuator ramps.

        Inside fan is held at the current greedy economic choice. The offline
        oracle found its anticipatory gain negligible, while fixing it here
        keeps the important fan/water search small enough to run every minute.
        Cooling-model predictions are vectorized into one call per module.
        """
        horizon = len(trajectory)
        water_step = 1.0
        stage_waters: list[np.ndarray] = []
        stage_fans: list[np.ndarray] = []
        feature_blocks: list[np.ndarray] = []

        for forecast in trajectory:
            if forecast["adiabatic"]:
                waters = np.arange(0.0, 80.0 + 0.1 * water_step, water_step)
            else:
                waters = np.asarray([0.0])
            fans = _required_fan(
                np.full(waters.size, float(forecast["load"][module])),
                waters,
                float(forecast["outdoor"]),
                float(forecast["wet_bulb"]),
            )
            feasible = fans <= 100.0 + 1e-9
            waters = waters[feasible]
            fans = np.clip(fans[feasible], 20.0, 100.0)
            if waters.size == 0:
                return None
            count = waters.size
            feature_blocks.append(
                np.column_stack(
                    [
                        np.full(count, float(forecast["load"][module]) * 1000.0),
                        np.full(count, float(forecast["return"][module])),
                        np.full(count, float(forecast["outdoor"])),
                        np.full(count, float(forecast["humidity"])),
                        np.full(count, float(forecast["wet_bulb"])),
                        fans,
                        np.full(count, inside),
                        waters,
                    ]
                )
            )
            stage_waters.append(waters)
            stage_fans.append(fans)

        # A horizon can cross the dry/adiabatic model boundary, so predict the
        # two regimes in batches rather than issuing one tiny call per minute.
        stage_costs: list[np.ndarray | None] = [None] * horizon
        for adiabatic in (False, True):
            indices = [i for i, row in enumerate(trajectory) if row["adiabatic"] == adiabatic]
            if not indices:
                continue
            sizes = [feature_blocks[i].shape[0] for i in indices]
            features = np.concatenate([feature_blocks[i] for i in indices], axis=0)
            regime = "adiabatic" if adiabatic else "dry"
            power_w = np.maximum(self.fits[module][regime].predict(features), 0.0)
            cursor = 0
            for i, size in zip(indices, sizes):
                power = power_w[cursor : cursor + size]
                stage_costs[i] = power / 1_000_000.0 + 0.0004 * stage_waters[i]
                cursor += size

        dp: list[np.ndarray] = []
        parents: list[np.ndarray] = []
        first = np.asarray(stage_costs[0], dtype=np.float64).copy()
        if self.previous_action is not None:
            prev = self.previous_action[3 * module : 3 * module + 3]
            fan_ok = np.abs(stage_fans[0] - prev[0]) <= self.ramp_limit + 1e-9
            # Entering dry mode applies the water interlock after ramping and
            # immediately records zero as the new previous action.
            water_ok = np.ones_like(fan_ok) if not trajectory[0]["adiabatic"] else (
                np.abs(stage_waters[0] - prev[2]) <= self.ramp_limit + 1e-9
            )
            first[~(fan_ok & water_ok)] = np.inf
        if not np.any(np.isfinite(first)):
            return None
        dp.append(first)
        parents.append(np.full(first.size, -1, dtype=np.int16))

        for t in range(1, horizon):
            current_cost = np.asarray(stage_costs[t], dtype=np.float64)
            fan_ok = (
                np.abs(stage_fans[t][:, None] - stage_fans[t - 1][None, :])
                <= self.ramp_limit + 1e-9
            )
            if trajectory[t]["adiabatic"]:
                water_ok = (
                    np.abs(stage_waters[t][:, None] - stage_waters[t - 1][None, :])
                    <= self.ramp_limit + 1e-9
                )
            else:
                water_ok = np.ones_like(fan_ok)
            candidate = np.where(fan_ok & water_ok, dp[t - 1][None, :], np.inf)
            parent = np.argmin(candidate, axis=1).astype(np.int16)
            best = candidate[np.arange(current_cost.size), parent]
            value = current_cost + best
            dp.append(value)
            parents.append(parent)
        if not np.any(np.isfinite(dp[-1])):
            return None

        index = int(np.argmin(dp[-1]))
        path = [index]
        for t in range(horizon - 1, 0, -1):
            index = int(parents[t][index])
            path.append(index)
        indices = path[::-1]
        return [
            (float(stage_fans[t][i]), float(stage_waters[t][i]))
            for t, i in enumerate(indices)
        ]

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float64)
        load = _modules(obs, 3, 400.0, 150.0)
        return_c = _modules(obs, 1, 45.0, 8.0)
        outdoor = float(obs[55] * 10.0 + 15.0)
        wet_bulb = float(obs[57] * 8.0 + 12.0)
        humidity = float(obs[56] * 20.0 + 65.0)

        # The official evaluator reuses one Agent across 480-minute episodes.
        # Clear temporal state at that boundary so seed N+1 is not extrapolated
        # from the final two samples of seed N.
        if self.steps >= 480:
            self.previous_action = None
            self.load_history.clear()
            self.return_history.clear()
            self.weather_history.clear()
            self.humidity_history.clear()
            for queue in self.plan_queues:
                queue.clear()
            self.steps = 0
        self._append_disturbance(load, return_c, outdoor, wet_bulb, humidity)
        trajectory = self._forecast_trajectory(12)
        current = trajectory[0]
        forecast_load = current["load"]
        forecast_outdoor = float(current["outdoor"])
        forecast_wet_bulb = float(current["wet_bulb"])
        forecast_return = current["return"]
        forecast_humidity = float(current["humidity"])
        predicted_adiabatic = bool(current["adiabatic"])
        desired = np.empty(3 * N_MODULES, dtype=np.float64)
        for module in range(N_MODULES):
            # The fitted models choose the minimum inside-fan setting in
            # virtually every state (and its measured oracle gain is
            # negligible).  Keeping it at 40 avoids a second, expensive
            # HistGradientBoosting prediction for every module and minute.
            inside = 40.0

            # In dry mode water has exactly one state and the required fan is
            # fixed by the analytical capacity boundary.  There is therefore
            # no economic search to perform.  Dry-weather fan trajectories are
            # smooth enough to be reached within the actuator ramp; important
            # mode transitions are already visible to the adiabatic planner
            # before dry mode begins.
            fan = float(
                _required_fan(
                    np.asarray([float(forecast_load[module])]),
                    np.asarray([0.0]),
                    forecast_outdoor,
                    forecast_wet_bulb,
                )[0]
            )
            fan = float(np.clip(fan, 20.0, 100.0))
            water = 0.0
            if predicted_adiabatic:
                queue = self.plan_queues[module]
                if not queue:
                    planned = self._plan_fan_water(module, trajectory, inside)
                    if planned is not None:
                        # Four-minute execution retains most of the benefit of
                        # receding-horizon planning while reducing tree-model
                        # inference by approximately a factor of four.
                        queue.extend(planned[:4])
                if queue:
                    planned_fan, water = queue.pop(0)
                    current_fan = float(
                        _required_fan(
                            np.asarray([float(forecast_load[module])]),
                            np.asarray([water]),
                            forecast_outdoor,
                            forecast_wet_bulb,
                        )[0]
                    )
                    fan = float(np.clip(max(planned_fan, current_fan), 20.0, 100.0))
                else:
                    # A temporarily unreachable capacity boundary is safest at
                    # maximum available heat-rejection capacity.
                    fan, water = 100.0, 80.0
            else:
                self.plan_queues[module].clear()
            desired[3 * module : 3 * module + 3] = (fan, inside, water)
        desired = np.clip(desired, LOW, HIGH)

        # Track the action the wrapper will apply, including its ramp and water
        # interlock, so future extensions can plan around reachability.
        if self.previous_action is None:
            applied = desired.copy()
        else:
            applied = np.clip(
                desired,
                self.previous_action - self.ramp_limit,
                self.previous_action + self.ramp_limit,
            )
        if not predicted_adiabatic:
            applied[2::3] = 0.0

        self.previous_action = applied
        self.steps += 1
        return desired.astype(np.float32)
