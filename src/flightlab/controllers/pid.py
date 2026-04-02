"""PID baseline controllers."""

from __future__ import annotations

from dataclasses import dataclass, field

from flightlab.core.geometry import clamp
from flightlab.core.types import AircraftState, ControlCommand


@dataclass
class PIDController:
    """Minimal PID controller with deterministic internal state."""

    kp: float
    ki: float
    kd: float
    output_limits: tuple[float, float] = (-1.0, 1.0)
    integral: float = 0.0
    previous_error: float = 0.0

    def reset(self) -> None:
        """Reset the controller integrator and derivative state."""
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error: float, dt_s: float) -> float:
        """Update the controller and return the clipped output."""
        derivative = 0.0 if dt_s <= 0.0 else (error - self.previous_error) / dt_s
        self.integral += error * dt_s
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return clamp(output, *self.output_limits)


@dataclass
class PIDAutopilot:
    """Simple classical baseline for waypoint and runway tasks."""

    heading_controller: PIDController = field(
        default_factory=lambda: PIDController(1.2, 0.0, 0.15, (-1.0, 1.0))
    )
    altitude_controller: PIDController = field(
        default_factory=lambda: PIDController(0.04, 0.001, 0.01, (-1.0, 1.0))
    )
    speed_controller: PIDController = field(
        default_factory=lambda: PIDController(0.08, 0.002, 0.01, (0.0, 1.0))
    )

    def reset(self) -> None:
        """Reset all low-level loops."""
        self.heading_controller.reset()
        self.altitude_controller.reset()
        self.speed_controller.reset()

    def command(
        self,
        state: AircraftState,
        *,
        heading_error_rad: float,
        altitude_error_m: float,
        speed_error_mps: float,
        dt_s: float,
    ) -> ControlCommand:
        """Produce a normalized four-axis control command."""
        aileron = self.heading_controller.update(heading_error_rad, dt_s)
        elevator = self.altitude_controller.update(altitude_error_m, dt_s)
        throttle = self.speed_controller.update(speed_error_mps, dt_s)
        rudder = 0.2 * aileron - 0.1 * state.sideslip_rad
        return ControlCommand(
            elevator=elevator,
            aileron=aileron,
            rudder=clamp(rudder, -1.0, 1.0),
            throttle=throttle,
        )
