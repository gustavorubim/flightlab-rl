"""Shared dataclasses used throughout the package."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flightlab.core.geometry import clamp


@dataclass
class AircraftState:
    """Aircraft state in a local East-North-Up frame."""

    position_x_m: float
    position_y_m: float
    altitude_m: float
    roll_rad: float
    pitch_rad: float
    heading_rad: float
    u_mps: float
    v_mps: float
    w_mps: float
    p_radps: float
    q_radps: float
    r_radps: float
    airspeed_mps: float
    groundspeed_mps: float
    vertical_speed_mps: float
    angle_of_attack_rad: float
    sideslip_rad: float
    throttle: float
    elevator: float
    aileron: float
    rudder: float
    mass_kg: float = 1200.0
    cg_offset_m: float = 0.0
    on_ground: bool = False
    time_s: float = 0.0

    @property
    def position_xy_m(self) -> tuple[float, float]:
        """Return the horizontal position tuple."""
        return (self.position_x_m, self.position_y_m)

    def as_observation_vector(self) -> np.ndarray:
        """Return the state as a base observation vector."""
        return np.asarray(
            [
                self.u_mps,
                self.v_mps,
                self.w_mps,
                self.p_radps,
                self.q_radps,
                self.r_radps,
                self.roll_rad,
                self.pitch_rad,
                np.sin(self.heading_rad),
                np.cos(self.heading_rad),
                self.altitude_m,
                self.vertical_speed_mps,
                self.airspeed_mps,
                self.groundspeed_mps,
                self.angle_of_attack_rad,
                self.sideslip_rad,
                self.throttle,
                self.elevator,
                self.aileron,
                self.rudder,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class ControlCommand:
    """Normalized control commands."""

    elevator: float
    aileron: float
    rudder: float
    throttle: float

    def clipped(self) -> ControlCommand:
        """Return the command clipped to normalized actuator ranges."""
        return ControlCommand(
            elevator=clamp(self.elevator, -1.0, 1.0),
            aileron=clamp(self.aileron, -1.0, 1.0),
            rudder=clamp(self.rudder, -1.0, 1.0),
            throttle=clamp(self.throttle, 0.0, 1.0),
        )

    def as_array(self) -> np.ndarray:
        """Convert the command to a float32 array."""
        return np.asarray(
            [self.elevator, self.aileron, self.rudder, self.throttle], dtype=np.float32
        )

    @classmethod
    def from_array(cls, value: np.ndarray | list[float] | tuple[float, ...]) -> ControlCommand:
        """Construct a command from a sequence."""
        elevator, aileron, rudder, throttle = [float(part) for part in value]
        return cls(elevator=elevator, aileron=aileron, rudder=rudder, throttle=throttle)


@dataclass
class TaskEvaluation:
    """Output returned by task evaluators."""

    reward: float
    phase: str
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    success: bool = False
    safety_flags: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
