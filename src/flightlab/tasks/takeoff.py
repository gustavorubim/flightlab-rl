"""Takeoff task phases, rewards, and terminations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from flightlab.core.types import AircraftState, TaskEvaluation
from flightlab.tasks.common import envelope_violation, safe_reward_component, stall_risk
from flightlab.world.runway import Runway


class TakeoffPhase(StrEnum):
    """Explicit takeoff phases."""

    TAXI_ALIGN = "TAXI_ALIGN"
    TAKEOFF_ROLL = "TAKEOFF_ROLL"
    ROTATE = "ROTATE"
    INITIAL_CLIMB = "INITIAL_CLIMB"


@dataclass(frozen=True)
class TakeoffTaskConfig:
    """Configuration for the takeoff task."""

    rotation_speed_mps: float = 24.0
    success_altitude_agl_m: float = 30.0
    max_centerline_error_m: float = 20.0
    max_heading_error_rad: float = 0.4
    max_steps: int = 400
    stall_speed_mps: float = 18.0


def classify_takeoff_phase(
    state: AircraftState, runway: Runway, config: TakeoffTaskConfig
) -> TakeoffPhase:
    """Classify the current takeoff phase."""
    altitude_agl_m = max(state.altitude_m - runway.elevation_m, 0.0)
    if altitude_agl_m > 1.5 and not state.on_ground:
        return TakeoffPhase.INITIAL_CLIMB
    if state.airspeed_mps <= 3.0:
        return TakeoffPhase.TAXI_ALIGN
    if state.airspeed_mps >= config.rotation_speed_mps:
        return TakeoffPhase.ROTATE
    return TakeoffPhase.TAKEOFF_ROLL


def evaluate_takeoff(
    state: AircraftState,
    runway: Runway,
    config: TakeoffTaskConfig,
) -> TaskEvaluation:
    """Evaluate reward, phase, and termination for takeoff."""
    phase = classify_takeoff_phase(state, runway, config)
    along_m, lateral_m = runway.local_coordinates(state.position_x_m, state.position_y_m)
    heading_error_rad = runway.heading_error_rad(state.heading_rad)
    raw_altitude_agl_m = state.altitude_m - runway.elevation_m
    altitude_agl_m = max(state.altitude_m - runway.elevation_m, 0.0)
    stall_risk_value = stall_risk(state, config.stall_speed_mps)
    excursion = abs(lateral_m) > runway.width_m / 2.0 + 3.0
    over_rotation = state.pitch_rad > 0.45 and altitude_agl_m < 5.0
    envelope = envelope_violation(state)
    crash = raw_altitude_agl_m < -1.0

    speed_component = min(state.airspeed_mps / config.rotation_speed_mps, 1.0)
    rotation_component = (
        1.0 if phase is TakeoffPhase.ROTATE and 0.08 <= state.pitch_rad <= 0.25 else 0.0
    )
    climb_component = safe_reward_component(
        config.success_altitude_agl_m - altitude_agl_m, config.success_altitude_agl_m
    )
    safety_penalty = (
        -1.5 * stall_risk_value - float(excursion) - float(over_rotation) - float(envelope)
    )

    reward_breakdown = {
        "centerline": safe_reward_component(lateral_m, config.max_centerline_error_m),
        "heading": safe_reward_component(heading_error_rad, config.max_heading_error_rad),
        "speed_buildup": speed_component,
        "rotation": rotation_component,
        "climb": climb_component,
        "safety": safety_penalty,
    }
    reward = (
        0.25 * reward_breakdown["centerline"]
        + 0.2 * reward_breakdown["heading"]
        + 0.2 * reward_breakdown["speed_buildup"]
        + 0.15 * reward_breakdown["rotation"]
        + 0.2 * reward_breakdown["climb"]
        + reward_breakdown["safety"]
    )
    success = (
        phase is TakeoffPhase.INITIAL_CLIMB
        and altitude_agl_m >= config.success_altitude_agl_m
        and abs(lateral_m) <= 10.0
        and abs(heading_error_rad) <= 0.25
    )

    return TaskEvaluation(
        reward=reward,
        phase=phase.value,
        reward_breakdown=reward_breakdown,
        terminated=success or excursion or over_rotation or envelope or crash,
        success=success,
        safety_flags={
            "stall": stall_risk_value >= 0.9,
            "runway_excursion": excursion,
            "over_rotation": over_rotation,
            "envelope_violation": envelope,
            "crash": crash,
        },
        metrics={
            "cross_track_error_m": abs(lateral_m),
            "altitude_error_m": config.success_altitude_agl_m - altitude_agl_m,
            "along_track_m": along_m,
            "stall_risk": stall_risk_value,
        },
    )
