"""Flight-plan-following environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flightlab.core.types import AircraftState, TaskEvaluation
from flightlab.dynamics.base import DynamicsConfig
from flightlab.envs.base import BaseFlightEnv
from flightlab.guidance.route import RouteManager
from flightlab.tasks.flight_plan import FlightPlanTaskConfig, evaluate_flight_plan
from flightlab.world.mission import Mission, Waypoint


def default_mission() -> Mission:
    """Return the default benchmark mission."""
    return Mission(
        name="pattern",
        waypoints=(
            Waypoint("wp1", 200.0, 0.0, 140.0, 26.0, 40.0),
            Waypoint("wp2", 350.0, 220.0, 155.0, 27.0, 40.0),
            Waypoint("wp3", 120.0, 320.0, 145.0, 25.0, 40.0),
        ),
    )


@dataclass(frozen=True)
class FlightPlanEnvConfig:
    """Environment configuration for the flight-plan task."""

    mission: Mission = field(default_factory=default_mission)
    task: FlightPlanTaskConfig = field(default_factory=FlightPlanTaskConfig)
    dynamics: DynamicsConfig = field(
        default_factory=lambda: DynamicsConfig(runway_elevation_m=120.0)
    )
    position_jitter_m: float = 20.0
    altitude_jitter_m: float = 10.0
    heading_jitter_rad: float = 0.15
    speed_jitter_mps: float = 2.0


class FlightPlanEnv(BaseFlightEnv):
    """Waypoint-following benchmark environment."""

    def __init__(
        self, *, seed: int | None = None, config: FlightPlanEnvConfig | None = None
    ) -> None:
        self.config = config or FlightPlanEnvConfig()
        self._route_manager = RouteManager(self.config.mission)
        self._latest_progress = None
        super().__init__(
            seed=seed,
            dynamics_config=self.config.dynamics,
            max_steps=self.config.task.max_steps,
        )

    @property
    def task_name(self) -> str:
        return "flight_plan"

    def _on_reset(self, options: dict[str, object]) -> None:
        self._route_manager.reset()
        self._latest_progress = None

    def _initial_state(self) -> AircraftState:
        first_waypoint = self.config.mission.waypoints[0]
        return AircraftState(
            position_x_m=first_waypoint.x_m
            - 140.0
            + float(
                self._rng.uniform(-self.config.position_jitter_m, self.config.position_jitter_m)
            ),
            position_y_m=first_waypoint.y_m
            - 60.0
            + float(
                self._rng.uniform(-self.config.position_jitter_m, self.config.position_jitter_m)
            ),
            roll_rad=0.0,
            pitch_rad=0.02,
            heading_rad=float(
                self._rng.uniform(-self.config.heading_jitter_rad, self.config.heading_jitter_rad)
            ),
            u_mps=24.0,
            v_mps=0.0,
            w_mps=0.0,
            p_radps=0.0,
            q_radps=0.0,
            r_radps=0.0,
            airspeed_mps=25.0
            + float(self._rng.uniform(-self.config.speed_jitter_mps, self.config.speed_jitter_mps)),
            groundspeed_mps=25.0,
            vertical_speed_mps=0.0,
            angle_of_attack_rad=0.03,
            sideslip_rad=0.0,
            throttle=0.55,
            elevator=0.0,
            aileron=0.0,
            rudder=0.0,
            on_ground=False,
            altitude_m=float(
                first_waypoint.altitude_m
                + self._rng.uniform(-self.config.altitude_jitter_m, self.config.altitude_jitter_m)
            ),
        )

    def _observe(self, state: AircraftState) -> np.ndarray:
        if self._latest_progress is None:
            progress = self._route_manager.progress(
                state.position_x_m,
                state.position_y_m,
                state.altitude_m,
                state.airspeed_mps,
            )
        else:
            progress = self._latest_progress
        dx_m = progress.current_waypoint.x_m - state.position_x_m
        dy_m = progress.current_waypoint.y_m - state.position_y_m
        return self._observation_builder.build(
            state,
            task_delta_x_m=dx_m,
            task_delta_y_m=dy_m,
            task_delta_altitude_m=progress.altitude_error_m,
            task_target_speed_error_mps=progress.speed_error_mps,
            altitude_agl_m=state.altitude_m - self.config.dynamics.runway_elevation_m,
        )

    def _evaluate(self, state: AircraftState) -> TaskEvaluation:
        progress = self._route_manager.progress(
            state.position_x_m,
            state.position_y_m,
            state.altitude_m,
            state.airspeed_mps,
        )
        self._latest_progress = progress
        return evaluate_flight_plan(state, progress, self.config.task)
