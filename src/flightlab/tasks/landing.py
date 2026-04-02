"""Landing task phases, rewards, and terminations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from flightlab.core.types import AircraftState, TaskEvaluation
from flightlab.guidance.approach import GlideslopeReference
from flightlab.tasks.common import envelope_violation, safe_reward_component, stall_risk
from flightlab.world.runway import Runway


class LandingPhase(StrEnum):
    """Explicit landing phases."""

    APPROACH = "APPROACH"
    FINAL = "FINAL"
    FLARE = "FLARE"
    TOUCHDOWN = "TOUCHDOWN"
    ROLLOUT = "ROLLOUT"


@dataclass(frozen=True)
class LandingTaskConfig:
    """Configuration for the landing task."""

    final_distance_m: float = 250.0
    flare_height_m: float = 6.0
    stop_speed_mps: float = 4.0
    touchdown_sink_limit_mps: float = 2.5
    max_centerline_error_m: float = 18.0
    max_steps: int = 600
    stall_speed_mps: float = 18.0
    glide_angle_deg: float = 3.0


def classify_landing_phase(
    state: AircraftState,
    runway: Runway,
    config: LandingTaskConfig,
    *,
    touchdown_step: bool,
) -> LandingPhase:
    """Classify the current landing phase."""
    along_m, _lateral_m = runway.local_coordinates(state.position_x_m, state.position_y_m)
    altitude_agl_m = max(state.altitude_m - runway.elevation_m, 0.0)
    if touchdown_step:
        return LandingPhase.TOUCHDOWN
    if state.on_ground:
        return LandingPhase.ROLLOUT
    if altitude_agl_m <= config.flare_height_m:
        return LandingPhase.FLARE
    if along_m >= -config.final_distance_m:
        return LandingPhase.FINAL
    return LandingPhase.APPROACH


def evaluate_landing(
    state: AircraftState,
    runway: Runway,
    config: LandingTaskConfig,
    *,
    touchdown_step: bool = False,
    touchdown_sink_rate_mps: float = 0.0,
) -> TaskEvaluation:
    """Evaluate reward, phase, and termination for landing."""
    glideslope = GlideslopeReference(runway=runway, glide_angle_deg=config.glide_angle_deg)
    phase = classify_landing_phase(state, runway, config, touchdown_step=touchdown_step)
    along_m, lateral_m = runway.local_coordinates(state.position_x_m, state.position_y_m)
    altitude_agl_m = max(state.altitude_m - runway.elevation_m, 0.0)
    glide_error_m = glideslope.altitude_error_m(along_m, state.altitude_m)
    heading_error_rad = runway.heading_error_rad(state.heading_rad)
    stall_risk_value = stall_risk(state, config.stall_speed_mps)
    runway_bounds = 0.0 <= along_m <= runway.length_m
    excursion = state.on_ground and (
        abs(lateral_m) > runway.width_m / 2.0 + 3.0 or not runway_bounds
    )
    hard_landing = touchdown_step and abs(touchdown_sink_rate_mps) > config.touchdown_sink_limit_mps
    envelope = envelope_violation(state)
    crash = hard_landing or (
        state.on_ground and not runway_bounds and state.groundspeed_mps > config.stop_speed_mps
    )

    flare_quality = 1.0 if phase is LandingPhase.FLARE and -0.1 <= state.pitch_rad <= 0.2 else 0.0
    touchdown_quality = (
        safe_reward_component(abs(touchdown_sink_rate_mps), config.touchdown_sink_limit_mps)
        if touchdown_step
        else 0.0
    )
    rollout_quality = 1.0 if phase is LandingPhase.ROLLOUT and abs(lateral_m) <= 3.0 else 0.0
    safety_penalty = (
        -1.5 * stall_risk_value - float(excursion) - float(hard_landing) - float(envelope)
    )

    reward_breakdown = {
        "alignment": safe_reward_component(lateral_m, config.max_centerline_error_m),
        "glideslope": safe_reward_component(glide_error_m, 25.0),
        "stability": safe_reward_component(heading_error_rad, 0.3),
        "flare": flare_quality,
        "touchdown": touchdown_quality,
        "rollout": rollout_quality,
        "safety": safety_penalty,
    }
    reward = (
        0.22 * reward_breakdown["alignment"]
        + 0.22 * reward_breakdown["glideslope"]
        + 0.16 * reward_breakdown["stability"]
        + 0.12 * reward_breakdown["flare"]
        + 0.16 * reward_breakdown["touchdown"]
        + 0.12 * reward_breakdown["rollout"]
        + reward_breakdown["safety"]
    )
    success = (
        phase is LandingPhase.ROLLOUT
        and state.groundspeed_mps <= config.stop_speed_mps
        and runway_bounds
        and abs(lateral_m) <= 4.0
    )
    return TaskEvaluation(
        reward=reward,
        phase=phase.value,
        reward_breakdown=reward_breakdown,
        terminated=success or excursion or hard_landing or envelope or crash,
        success=success,
        safety_flags={
            "stall": stall_risk_value >= 0.9,
            "runway_excursion": excursion,
            "hard_landing": hard_landing,
            "envelope_violation": envelope,
            "crash": crash,
        },
        metrics={
            "cross_track_error_m": abs(lateral_m),
            "altitude_error_m": glide_error_m,
            "touchdown_sink_rate_mps": abs(touchdown_sink_rate_mps),
            "stall_risk": stall_risk_value,
            "altitude_agl_m": altitude_agl_m,
        },
    )
