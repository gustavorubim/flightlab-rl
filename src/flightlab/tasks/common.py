"""Common task helpers."""

from __future__ import annotations

import math

from flightlab.core.geometry import clamp
from flightlab.core.types import AircraftState


def stall_risk(state: AircraftState, stall_speed_mps: float) -> float:
    """Estimate stall risk from airspeed and angle of attack."""
    speed_term = clamp((stall_speed_mps + 4.0 - state.airspeed_mps) / 6.0, 0.0, 1.0)
    alpha_term = clamp(abs(state.angle_of_attack_rad) / 0.35, 0.0, 1.0)
    return clamp(0.6 * speed_term + 0.4 * alpha_term, 0.0, 1.0)


def envelope_violation(state: AircraftState) -> bool:
    """Return whether the state violates a conservative fixed-wing envelope."""
    return abs(state.roll_rad) > 1.1 or abs(state.pitch_rad) > 0.8 or math.isnan(state.altitude_m)


def safe_reward_component(error: float, tolerance: float) -> float:
    """Convert an absolute error to a bounded reward-like score."""
    return clamp(1.0 - abs(error) / max(tolerance, 1e-6), -1.0, 1.0)
