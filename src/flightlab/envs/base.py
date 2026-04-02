"""Base Gymnasium environment for flightlab tasks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flightlab.core.seed import seeded_rng
from flightlab.core.types import AircraftState, ControlCommand, TaskEvaluation
from flightlab.dynamics.base import DynamicsConfig
from flightlab.dynamics.kinematic import KinematicDynamics
from flightlab.render.replay import EpisodeRecorder
from flightlab.sensors.observation import ObservationBuilder


class BaseFlightEnv(gym.Env[np.ndarray, np.ndarray], ABC):
    """Shared Gymnasium implementation for all task-specific environments."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        *,
        seed: int | None = None,
        dynamics_config: DynamicsConfig | None = None,
        max_steps: int,
    ) -> None:
        super().__init__()
        self._base_seed = 0 if seed is None else int(seed)
        self._rng = seeded_rng(self._base_seed)
        self._dynamics = KinematicDynamics(dynamics_config or DynamicsConfig())
        self._observation_builder = ObservationBuilder()
        self._recorder = EpisodeRecorder()
        self._max_steps = max_steps
        self._step_count = 0
        self._last_action = np.zeros(4, dtype=np.float32)
        self._last_info: dict[str, Any] = {}
        self._episode_return = 0.0
        self._cross_track_errors: list[float] = []
        self._altitude_errors: list[float] = []
        self._action_deltas: list[float] = []
        self._latest_observation = np.zeros(
            len(self._observation_builder.feature_names), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._observation_builder.feature_names),),
            dtype=np.float32,
        )

    @property
    def state(self) -> AircraftState:
        """Return the current aircraft state."""
        return self._dynamics.state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment with deterministic seeding."""
        super().reset(seed=seed)
        effective_seed = self._base_seed if seed is None else int(seed)
        self._rng = seeded_rng(effective_seed)
        self._step_count = 0
        self._episode_return = 0.0
        self._cross_track_errors = []
        self._altitude_errors = []
        self._action_deltas = []
        self._last_action = np.zeros(4, dtype=np.float32)
        self._last_info = {}
        self._recorder.reset()
        self._on_reset(options or {})
        state = self._dynamics.reset(self._initial_state())
        observation = self._observe(state)
        info = self._build_info(self._reset_evaluation(state))
        info["seed"] = effective_seed
        self._last_info = info
        self._recorder.record_reset(state, info)
        self._latest_observation = observation
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one environment step."""
        self._step_count += 1
        action_array = np.asarray(action, dtype=np.float32)
        command = ControlCommand.from_array(action_array)
        previous_action = self._last_action.copy()
        state = self._dynamics.step(command)
        evaluation = self._evaluate(state)
        truncated = self._step_count >= self._max_steps
        observation = self._observe(state)
        self._latest_observation = observation
        self._episode_return += evaluation.reward
        self._cross_track_errors.append(
            abs(float(evaluation.metrics.get("cross_track_error_m", 0.0)))
        )
        altitude_error = float(evaluation.metrics.get("altitude_error_m", 0.0))
        self._altitude_errors.append(altitude_error)
        self._action_deltas.append(float(np.mean(np.abs(action_array - previous_action))))
        info = self._build_info(evaluation)
        if evaluation.terminated or truncated:
            info["episode_summary"] = self._episode_summary(info=info, success=evaluation.success)
        self._last_info = info
        self._last_action = action_array
        self._recorder.record_step(state, action_array.tolist(), evaluation.reward, info)
        return observation, evaluation.reward, evaluation.terminated, truncated, info

    def render(self) -> str:
        """Render a compact ANSI-like state summary."""
        phase = self._last_info.get("task_phase", "RESET")
        reward = float(self._last_info.get("reward", 0.0))
        return (
            f"time={self.state.time_s:.1f}s "
            f"task={self.task_name} "
            f"phase={phase} "
            f"pos=({self.state.position_x_m:.1f},"
            f"{self.state.position_y_m:.1f},"
            f"{self.state.altitude_m:.1f}) "
            f"speed={self.state.airspeed_mps:.1f} reward={reward:.3f}"
        )

    def export_replay(self, path: str) -> str:
        """Export the current episode replay to JSON."""
        return str(self._recorder.export_json(path))

    def episode_summary(self, *, success: bool | None = None) -> dict[str, float | bool]:
        """Return a benchmark-friendly summary for the current episode state."""
        effective_success = (
            bool(self._last_info.get("success", False)) if success is None else success
        )
        return self._episode_summary(info=self._last_info, success=effective_success)

    def _episode_summary(self, *, info: dict[str, Any], success: bool) -> dict[str, float | bool]:
        """Build a benchmark-friendly per-episode summary."""
        altitude_rmse_m = math.sqrt(
            sum(error * error for error in self._altitude_errors)
            / max(len(self._altitude_errors), 1)
        )
        return {
            "success": success,
            "crash": bool(info.get("safety_flags", {}).get("crash", False)),
            "stall": bool(info.get("safety_flags", {}).get("stall", False)),
            "runway_excursion": bool(info.get("safety_flags", {}).get("runway_excursion", False)),
            "average_cross_track_error_m": float(np.mean(self._cross_track_errors))
            if self._cross_track_errors
            else 0.0,
            "altitude_rmse_m": altitude_rmse_m,
            "action_smoothness": float(np.mean(self._action_deltas))
            if self._action_deltas
            else 0.0,
            "episode_return": self._episode_return,
            "completion_time_s": self.state.time_s,
        }

    def _build_info(self, evaluation: TaskEvaluation) -> dict[str, Any]:
        """Build the standardized info dictionary."""
        info = {
            "reward": evaluation.reward,
            "reward_breakdown": dict(evaluation.reward_breakdown),
            "task_phase": evaluation.phase,
            "safety_flags": dict(evaluation.safety_flags),
            "stall_risk": float(evaluation.metrics.get("stall_risk", 0.0)),
            "cross_track_error_m": float(evaluation.metrics.get("cross_track_error_m", 0.0)),
            "altitude_error_m": float(evaluation.metrics.get("altitude_error_m", 0.0)),
            "touchdown_metrics": {
                key: value
                for key, value in evaluation.metrics.items()
                if key.startswith("touchdown_")
            },
            "success": evaluation.success,
        }
        return info

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Return the environment task name."""

    @abstractmethod
    def _initial_state(self) -> AircraftState:
        """Return the initial state for the next episode."""

    @abstractmethod
    def _observe(self, state: AircraftState) -> np.ndarray:
        """Build the task-specific observation."""

    @abstractmethod
    def _evaluate(self, state: AircraftState) -> TaskEvaluation:
        """Evaluate the current state."""

    def _on_reset(self, options: dict[str, Any]) -> None:
        """Hook for task-specific reset state."""

    def _reset_evaluation(self, state: AircraftState) -> TaskEvaluation:
        """Return the initial info payload after reset."""
        return TaskEvaluation(reward=0.0, phase="RESET")
