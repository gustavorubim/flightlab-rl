"""Takeoff environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flightlab.core.types import AircraftState, TaskEvaluation
from flightlab.dynamics.base import DynamicsConfig
from flightlab.envs.base import BaseFlightEnv
from flightlab.tasks.takeoff import TakeoffTaskConfig, evaluate_takeoff
from flightlab.world.runway import Runway


def default_takeoff_runway() -> Runway:
    """Return the default takeoff runway."""
    return Runway(name="09", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)


@dataclass(frozen=True)
class TakeoffEnvConfig:
    """Environment configuration for takeoff."""

    runway: Runway = field(default_factory=default_takeoff_runway)
    task: TakeoffTaskConfig = field(default_factory=TakeoffTaskConfig)
    dynamics: DynamicsConfig = field(
        default_factory=lambda: DynamicsConfig(runway_elevation_m=120.0, lift_off_speed_mps=24.0)
    )
    lateral_jitter_m: float = 1.5
    heading_jitter_rad: float = 0.04


class TakeoffEnv(BaseFlightEnv):
    """Runway takeoff benchmark environment."""

    def __init__(self, *, seed: int | None = None, config: TakeoffEnvConfig | None = None) -> None:
        self.config = config or TakeoffEnvConfig()
        super().__init__(
            seed=seed,
            dynamics_config=self.config.dynamics,
            max_steps=self.config.task.max_steps,
        )

    @property
    def task_name(self) -> str:
        return "takeoff"

    def _initial_state(self) -> AircraftState:
        runway = self.config.runway
        return AircraftState(
            position_x_m=runway.threshold_x_m - 10.0 + float(self._rng.uniform(-1.0, 1.0)),
            position_y_m=runway.threshold_y_m
            + float(self._rng.uniform(-self.config.lateral_jitter_m, self.config.lateral_jitter_m)),
            altitude_m=runway.elevation_m,
            roll_rad=0.0,
            pitch_rad=0.0,
            heading_rad=runway.heading_rad
            + float(
                self._rng.uniform(-self.config.heading_jitter_rad, self.config.heading_jitter_rad)
            ),
            u_mps=1.0,
            v_mps=0.0,
            w_mps=0.0,
            p_radps=0.0,
            q_radps=0.0,
            r_radps=0.0,
            airspeed_mps=1.0,
            groundspeed_mps=1.0,
            vertical_speed_mps=0.0,
            angle_of_attack_rad=0.0,
            sideslip_rad=0.0,
            throttle=0.2,
            elevator=0.0,
            aileron=0.0,
            rudder=0.0,
            on_ground=True,
        )

    def _observe(self, state: AircraftState) -> np.ndarray:
        along_m, lateral_m = self.config.runway.local_coordinates(
            state.position_x_m, state.position_y_m
        )
        return self._observation_builder.build(
            state,
            task_delta_x_m=along_m,
            task_delta_y_m=lateral_m,
            task_delta_altitude_m=self.config.task.success_altitude_agl_m
            - (state.altitude_m - self.config.runway.elevation_m),
            task_target_speed_error_mps=self.config.task.rotation_speed_mps - state.airspeed_mps,
            altitude_agl_m=state.altitude_m - self.config.runway.elevation_m,
        )

    def _evaluate(self, state: AircraftState) -> TaskEvaluation:
        return evaluate_takeoff(state, self.config.runway, self.config.task)
