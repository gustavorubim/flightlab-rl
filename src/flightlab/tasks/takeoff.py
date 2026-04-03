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
    rotation_pitch_target_rad: float = 0.14
    rotation_pitch_tolerance_rad: float = 0.08
    rotation_window_speed_ratio: float = 0.9
    climb_target_vertical_speed_mps: float = 3.5
    overspeed_abort_margin_mps: float = 4.0
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
    failed_liftoff = state.on_ground and (
        state.airspeed_mps >= config.rotation_speed_mps + config.overspeed_abort_margin_mps
    )
    envelope = envelope_violation(state)
    crash = raw_altitude_agl_m < -1.0

    speed_component = min(state.airspeed_mps / config.rotation_speed_mps, 1.0)
    in_rotation_window = state.airspeed_mps >= (
        config.rotation_speed_mps * config.rotation_window_speed_ratio
    )
    rotation_component = 0.0
    climb_component = 0.0
    vertical_climb_component = 0.0
    delayed_rotation_penalty = 0.0
    if in_rotation_window:
        rotation_component = safe_reward_component(
            state.pitch_rad - config.rotation_pitch_target_rad,
            config.rotation_pitch_tolerance_rad,
        )
        climb_component = max(
            -1.0,
            min((2.0 * altitude_agl_m / config.success_altitude_agl_m) - 1.0, 1.0),
        )
        vertical_climb_component = max(
            -1.0,
            min(
                state.vertical_speed_mps / max(config.climb_target_vertical_speed_mps, 1e-6),
                1.0,
            ),
        )
        if state.on_ground and altitude_agl_m < 0.5:
            delayed_rotation_penalty = -max(
                0.0,
                min(
                    (
                        state.airspeed_mps
                        - (config.rotation_speed_mps * config.rotation_window_speed_ratio)
                    )
                    / max(
                        config.rotation_speed_mps
                        * (1.0 - config.rotation_window_speed_ratio)
                        + config.overspeed_abort_margin_mps,
                        1e-6,
                    ),
                    1.0,
                ),
            )
    elif phase is TakeoffPhase.TAXI_ALIGN:
        rotation_component = 0.0
    else:
        rotation_component = max(
            -1.0,
            min(state.pitch_rad / max(config.rotation_pitch_target_rad, 1e-6), 1.0),
        )

    climb_component = max(
        climb_component,
        safe_reward_component(
            config.success_altitude_agl_m - altitude_agl_m,
            config.success_altitude_agl_m,
        ),
    )
    safety_penalty = (
        -1.5 * stall_risk_value - float(excursion) - float(over_rotation) - float(envelope)
        - float(failed_liftoff)
    )

    reward_breakdown = {
        "centerline": safe_reward_component(lateral_m, config.max_centerline_error_m),
        "heading": safe_reward_component(heading_error_rad, config.max_heading_error_rad),
        "speed_buildup": speed_component,
        "rotation": rotation_component,
        "climb": climb_component,
        "vertical_climb": vertical_climb_component,
        "delayed_rotation": delayed_rotation_penalty,
        "safety": safety_penalty,
    }
    reward = (
        0.2 * reward_breakdown["centerline"]
        + 0.15 * reward_breakdown["heading"]
        + 0.15 * reward_breakdown["speed_buildup"]
        + 0.15 * reward_breakdown["rotation"]
        + 0.2 * reward_breakdown["climb"]
        + 0.15 * reward_breakdown["vertical_climb"]
        + reward_breakdown["delayed_rotation"]
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
        terminated=success or excursion or over_rotation or failed_liftoff or envelope or crash,
        success=success,
        safety_flags={
            "stall": stall_risk_value >= 0.9,
            "runway_excursion": excursion,
            "over_rotation": over_rotation,
            "failed_liftoff": failed_liftoff,
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
