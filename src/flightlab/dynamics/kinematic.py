"""Deterministic lightweight fixed-wing dynamics backend."""

from __future__ import annotations

import math
from dataclasses import replace

from flightlab.core.geometry import clamp, wrap_angle_rad
from flightlab.core.types import AircraftState, ControlCommand
from flightlab.dynamics.base import DynamicsConfig


class KinematicDynamics:
    """A compact deterministic backend for fast headless experiments and tests."""

    def __init__(self, config: DynamicsConfig | None = None) -> None:
        self.config = config or DynamicsConfig()
        self._state: AircraftState | None = None

    @property
    def state(self) -> AircraftState:
        """Return the current state."""
        if self._state is None:
            raise RuntimeError("Dynamics backend has not been reset.")
        return self._state

    def reset(self, initial_state: AircraftState) -> AircraftState:
        """Reset the internal state."""
        self._state = replace(initial_state)
        return self.state

    def step(self, command: ControlCommand) -> AircraftState:
        """Advance the dynamics by one fixed step."""
        if self._state is None:
            raise RuntimeError("Dynamics backend has not been reset.")

        state = replace(self._state)
        dt_s = self.config.dt_s
        clipped = command.clipped()
        actuator_blend = clamp(dt_s / max(self.config.actuator_tau_s, dt_s), 0.0, 1.0)

        state.elevator += actuator_blend * (clipped.elevator - state.elevator)
        state.aileron += actuator_blend * (clipped.aileron - state.aileron)
        state.rudder += actuator_blend * (clipped.rudder - state.rudder)
        state.throttle += actuator_blend * (clipped.throttle - state.throttle)

        mass_factor = max(state.mass_kg / self.config.nominal_mass_kg, 0.5)
        thrust_accel = (18.0 * state.throttle) / mass_factor
        drag_accel = 0.02 * state.airspeed_mps**2
        speed_dot = thrust_accel - drag_accel - 7.0 * math.sin(state.pitch_rad)
        airspeed_mps = max(state.airspeed_mps + dt_s * speed_dot, 0.0)

        p_dot = -1.8 * state.p_radps + 2.4 * state.aileron + 0.2 * state.rudder
        q_dot = -2.1 * state.q_radps + 2.8 * state.elevator - 0.2 * state.pitch_rad
        r_dot = -1.6 * state.r_radps + 1.8 * state.rudder + 0.1 * state.aileron
        state.p_radps += dt_s * p_dot
        state.q_radps += dt_s * q_dot
        state.r_radps += dt_s * r_dot

        state.roll_rad = clamp(state.roll_rad + dt_s * state.p_radps, -1.1, 1.1)
        state.pitch_rad = clamp(state.pitch_rad + dt_s * state.q_radps, -0.5, 0.7)

        heading_rate = (
            9.81 * math.tan(clamp(state.roll_rad, -0.9, 0.9)) / max(airspeed_mps, 5.0)
        ) + (0.1 * state.r_radps)
        state.heading_rad = wrap_angle_rad(state.heading_rad + dt_s * heading_rate)

        climb_command = 0.35 * airspeed_mps * math.sin(state.pitch_rad) + 0.04 * (
            airspeed_mps - 20.0
        )
        vertical_accel = climb_command - 0.45 * state.vertical_speed_mps
        vertical_speed_mps = state.vertical_speed_mps + dt_s * vertical_accel

        on_ground = state.on_ground
        if on_ground:
            if airspeed_mps > self.config.lift_off_speed_mps and state.pitch_rad > 0.05:
                on_ground = False
            else:
                vertical_speed_mps = 0.0
                state.altitude_m = self.config.runway_elevation_m
                state.roll_rad *= 0.5
                state.pitch_rad = clamp(state.pitch_rad, -0.1, 0.25)

        altitude_m = state.altitude_m + dt_s * vertical_speed_mps
        if not on_ground and altitude_m <= self.config.runway_elevation_m:
            altitude_m = self.config.runway_elevation_m
            vertical_speed_mps = 0.0
            on_ground = True
            state.pitch_rad = 0.0
            state.roll_rad *= 0.6

        horizontal_airspeed_mps = math.sqrt(max(airspeed_mps**2 - vertical_speed_mps**2, 0.0))
        ground_vx_mps = (
            horizontal_airspeed_mps * math.cos(state.heading_rad) + self.config.wind_east_mps
        )
        ground_vy_mps = (
            horizontal_airspeed_mps * math.sin(state.heading_rad) + self.config.wind_north_mps
        )
        groundspeed_mps = math.hypot(ground_vx_mps, ground_vy_mps)
        state.position_x_m += dt_s * ground_vx_mps
        state.position_y_m += dt_s * ground_vy_mps

        flight_path_angle = math.atan2(vertical_speed_mps, max(horizontal_airspeed_mps, 1.0))
        angle_of_attack_rad = clamp(
            state.pitch_rad - flight_path_angle + 0.05 * state.elevator, -0.5, 0.5
        )
        sideslip_rad = clamp(
            0.2 * state.rudder
            + 0.02 * self.config.wind_north_mps
            - 0.02 * self.config.wind_east_mps,
            -0.3,
            0.3,
        )

        state.u_mps = airspeed_mps * math.cos(angle_of_attack_rad)
        state.v_mps = airspeed_mps * math.sin(sideslip_rad)
        state.w_mps = airspeed_mps * math.sin(angle_of_attack_rad)
        state.airspeed_mps = airspeed_mps
        state.groundspeed_mps = groundspeed_mps
        state.vertical_speed_mps = vertical_speed_mps
        state.altitude_m = altitude_m
        state.angle_of_attack_rad = angle_of_attack_rad
        state.sideslip_rad = sideslip_rad
        state.on_ground = on_ground
        state.time_s += dt_s
        self._state = state
        return self.state
