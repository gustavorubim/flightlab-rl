"""Landing environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flightlab.core.types import AircraftState, TaskEvaluation
from flightlab.dynamics.base import DynamicsConfig
from flightlab.envs.base import BaseFlightEnv
from flightlab.guidance.approach import GlideslopeReference
from flightlab.tasks.landing import LandingTaskConfig, evaluate_landing
from flightlab.world.runway import Runway


def default_landing_runway() -> Runway:
    """Return the default landing runway."""
    return Runway(name="27", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)


@dataclass(frozen=True)
class LandingEnvConfig:
    """Environment configuration for landing."""

    runway: Runway = field(default_factory=default_landing_runway)
    task: LandingTaskConfig = field(default_factory=LandingTaskConfig)
    dynamics: DynamicsConfig = field(
        default_factory=lambda: DynamicsConfig(runway_elevation_m=120.0)
    )
    lateral_jitter_m: float = 8.0
    altitude_jitter_m: float = 8.0
    heading_jitter_rad: float = 0.05


class LandingEnv(BaseFlightEnv):
    """Runway landing benchmark environment."""

    def __init__(self, *, seed: int | None = None, config: LandingEnvConfig | None = None) -> None:
        self.config = config or LandingEnvConfig()
        self._glideslope = GlideslopeReference(
            runway=self.config.runway,
            glide_angle_deg=self.config.task.glide_angle_deg,
        )
        self._touchdown_step = False
        self._touchdown_sink_rate_mps = 0.0
        self._previous_vertical_speed_mps = 0.0
        self._previous_on_ground = False
        super().__init__(
            seed=seed,
            dynamics_config=self.config.dynamics,
            max_steps=self.config.task.max_steps,
        )

    @property
    def task_name(self) -> str:
        return "landing"

    def _on_reset(self, options: dict[str, object]) -> None:
        self._touchdown_step = False
        self._touchdown_sink_rate_mps = 0.0
        self._previous_vertical_speed_mps = 0.0
        self._previous_on_ground = False

    def _initial_state(self) -> AircraftState:
        runway = self.config.runway
        along_m = -650.0
        altitude_m = self._glideslope.target_altitude_m(along_m)
        return AircraftState(
            position_x_m=along_m + float(self._rng.uniform(-10.0, 10.0)),
            position_y_m=float(
                self._rng.uniform(-self.config.lateral_jitter_m, self.config.lateral_jitter_m)
            ),
            altitude_m=altitude_m
            + float(
                self._rng.uniform(-self.config.altitude_jitter_m, self.config.altitude_jitter_m)
            ),
            roll_rad=0.0,
            pitch_rad=-0.04,
            heading_rad=runway.heading_rad
            + float(
                self._rng.uniform(-self.config.heading_jitter_rad, self.config.heading_jitter_rad)
            ),
            u_mps=26.0,
            v_mps=0.0,
            w_mps=-1.5,
            p_radps=0.0,
            q_radps=0.0,
            r_radps=0.0,
            airspeed_mps=26.0,
            groundspeed_mps=26.0,
            vertical_speed_mps=-1.5,
            angle_of_attack_rad=0.02,
            sideslip_rad=0.0,
            throttle=0.45,
            elevator=0.0,
            aileron=0.0,
            rudder=0.0,
            on_ground=False,
        )

    def _observe(self, state: AircraftState) -> np.ndarray:
        along_m, lateral_m = self.config.runway.local_coordinates(
            state.position_x_m, state.position_y_m
        )
        return self._observation_builder.build(
            state,
            task_delta_x_m=along_m,
            task_delta_y_m=lateral_m,
            task_delta_altitude_m=self._glideslope.altitude_error_m(along_m, state.altitude_m),
            task_target_speed_error_mps=20.0 - state.airspeed_mps,
            altitude_agl_m=state.altitude_m - self.config.runway.elevation_m,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Capture sink rate before touchdown clamping occurs in the dynamics model."""
        self._previous_vertical_speed_mps = self.state.vertical_speed_mps
        return super().step(action)

    def _evaluate(self, state: AircraftState) -> TaskEvaluation:
        self._touchdown_step = not self._previous_on_ground and state.on_ground
        self._touchdown_sink_rate_mps = (
            abs(self._previous_vertical_speed_mps) if self._touchdown_step else 0.0
        )
        evaluation = evaluate_landing(
            state,
            self.config.runway,
            self.config.task,
            touchdown_step=self._touchdown_step,
            touchdown_sink_rate_mps=self._touchdown_sink_rate_mps,
        )
        self._previous_on_ground = state.on_ground
        return evaluation
