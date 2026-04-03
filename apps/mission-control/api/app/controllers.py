"""Live controller adapters for mission-control runtime sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import RLPhaseSwitchedConfig
from flightlab.controllers import PIDAutopilot
from flightlab.core.geometry import clamp, signed_smallest_angle
from flightlab.core.types import AircraftState, ControlCommand
from flightlab.guidance.route import RouteProgress
from flightlab.rl import load_model_class
from flightlab.sensors.observation import ObservationBuilder
from flightlab.tasks.takeoff import TakeoffTaskConfig
from flightlab.world.runway import Runway


class MissionPilot(Protocol):
    """Protocol implemented by live mission-control pilots."""

    def reset(self) -> None:
        """Reset any internal controller state."""

    def command(
        self,
        state: AircraftState,
        *,
        phase: str,
        runway: Runway,
        route_progress: RouteProgress | None,
        takeoff_config: TakeoffTaskConfig,
        dt_s: float,
    ) -> ControlCommand:
        """Produce the next live control command."""


@dataclass
class PIDMissionPilot:
    """Deterministic live mission pilot built on the repo's PID autopilot."""

    autopilot: PIDAutopilot

    def __init__(self) -> None:
        self.autopilot = PIDAutopilot()

    def reset(self) -> None:
        self.autopilot.reset()

    def command(
        self,
        state: AircraftState,
        *,
        phase: str,
        runway: Runway,
        route_progress: RouteProgress | None,
        takeoff_config: TakeoffTaskConfig,
        dt_s: float,
    ) -> ControlCommand:
        if phase == "standby":
            return ControlCommand(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.0)
        if phase in {"takeoff_roll", "climb_out"}:
            _along_m, lateral_m = runway.local_coordinates(state.position_x_m, state.position_y_m)
            heading_error = runway.heading_error_rad(state.heading_rad) + clamp(
                -lateral_m / 45.0,
                -0.22,
                0.22,
            )
            altitude_agl_m = max(state.altitude_m - runway.elevation_m, 0.0)
            aileron = self.autopilot.heading_controller.update(heading_error, dt_s)
            rudder = clamp(
                (0.6 * heading_error) + (0.15 * aileron) - (0.08 * state.sideslip_rad), -0.8, 0.8
            )
            rotation_active = (
                state.airspeed_mps
                >= takeoff_config.rotation_speed_mps * takeoff_config.rotation_window_speed_ratio
            ) or not state.on_ground
            if phase == "takeoff_roll":
                if rotation_active:
                    pitch_target_rad = takeoff_config.rotation_pitch_target_rad
                    pitch_error = pitch_target_rad - state.pitch_rad
                    elevator = clamp((1.05 * pitch_error) - (0.18 * state.q_radps), -0.08, 0.18)
                else:
                    elevator = clamp(
                        (-0.35 * state.pitch_rad) - (0.12 * state.q_radps), -0.05, 0.05
                    )
                throttle = 1.0
            else:
                pitch_target_rad = 0.12 if altitude_agl_m < 18.0 else 0.08
                pitch_error = pitch_target_rad - state.pitch_rad
                vertical_speed_error = (
                    takeoff_config.climb_target_vertical_speed_mps - state.vertical_speed_mps
                )
                elevator = clamp(
                    (0.95 * pitch_error) + (0.03 * vertical_speed_error) - (0.16 * state.q_radps),
                    -0.1,
                    0.16,
                )
                throttle = (
                    1.0 if state.airspeed_mps < takeoff_config.rotation_speed_mps + 4.0 else 0.9
                )
            return ControlCommand(
                elevator=elevator,
                aileron=aileron,
                rudder=rudder,
                throttle=throttle,
            )
        if route_progress is None:
            return ControlCommand(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.45)
        return self.autopilot.command(
            state,
            heading_error_rad=signed_smallest_angle(
                state.heading_rad,
                route_progress.desired_track_rad,
            ),
            altitude_error_m=route_progress.altitude_error_m,
            speed_error_mps=route_progress.speed_error_mps,
            dt_s=dt_s,
        )


class RLPhaseSwitchedPilot:
    """Live mission pilot that switches between task-specific checkpoints by phase."""

    def __init__(self, config: RLPhaseSwitchedConfig) -> None:
        self._config = config
        takeoff_cls = load_model_class(config.takeoff.algorithm)
        flight_plan_cls = load_model_class(config.flight_plan.algorithm)
        self._takeoff_model = takeoff_cls.load(str(config.takeoff.load_path))
        self._flight_plan_model = flight_plan_cls.load(str(config.flight_plan.load_path))
        self._observation_builder = ObservationBuilder()

    def reset(self) -> None:
        """Reset no-op for stateless inference models."""

    def command(
        self,
        state: AircraftState,
        *,
        phase: str,
        runway: Runway,
        route_progress: RouteProgress | None,
        takeoff_config: TakeoffTaskConfig,
        dt_s: float,
    ) -> ControlCommand:
        del dt_s
        if phase == "standby":
            return ControlCommand(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.0)
        if phase in {"takeoff_roll", "climb_out"}:
            observation = self._build_takeoff_observation(state, runway, takeoff_config)
            action, _state = self._takeoff_model.predict(observation, deterministic=True)
            return ControlCommand.from_array(action)
        observation = self._build_flight_plan_observation(state, runway, route_progress)
        action, _state = self._flight_plan_model.predict(observation, deterministic=True)
        return ControlCommand.from_array(action)

    def _build_takeoff_observation(
        self,
        state: AircraftState,
        runway: Runway,
        takeoff_config: TakeoffTaskConfig,
    ) -> object:
        along_m, lateral_m = runway.local_coordinates(state.position_x_m, state.position_y_m)
        return self._observation_builder.build(
            state,
            task_delta_x_m=along_m,
            task_delta_y_m=lateral_m,
            task_delta_altitude_m=takeoff_config.success_altitude_agl_m
            - (state.altitude_m - runway.elevation_m),
            task_target_speed_error_mps=takeoff_config.rotation_speed_mps - state.airspeed_mps,
            altitude_agl_m=state.altitude_m - runway.elevation_m,
        )

    def _build_flight_plan_observation(
        self,
        state: AircraftState,
        runway: Runway,
        route_progress: RouteProgress | None,
    ) -> object:
        if route_progress is None:
            raise RuntimeError("Route progress is required for enroute RL control.")
        dx_m = route_progress.current_waypoint.x_m - state.position_x_m
        dy_m = route_progress.current_waypoint.y_m - state.position_y_m
        return self._observation_builder.build(
            state,
            task_delta_x_m=dx_m,
            task_delta_y_m=dy_m,
            task_delta_altitude_m=route_progress.altitude_error_m,
            task_target_speed_error_mps=route_progress.speed_error_mps,
            altitude_agl_m=state.altitude_m - runway.elevation_m,
        )
