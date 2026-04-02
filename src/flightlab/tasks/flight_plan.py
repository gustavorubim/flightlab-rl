"""Flight-plan following rewards and terminations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from flightlab.core.geometry import signed_smallest_angle
from flightlab.core.types import AircraftState, TaskEvaluation
from flightlab.guidance.route import RouteProgress
from flightlab.tasks.common import envelope_violation, safe_reward_component, stall_risk


class FlightPlanPhase(StrEnum):
    """High-level flight-plan phase."""

    ENROUTE = "ENROUTE"
    WAYPOINT_CAPTURE = "WAYPOINT_CAPTURE"
    MISSION_COMPLETE = "MISSION_COMPLETE"


@dataclass(frozen=True)
class FlightPlanTaskConfig:
    """Configuration for the waypoint-following task."""

    max_cross_track_error_m: float = 120.0
    max_altitude_error_m: float = 80.0
    max_heading_error_rad: float = 0.6
    completion_bonus: float = 2.0
    waypoint_bonus: float = 0.5
    stall_speed_mps: float = 18.0
    max_steps: int = 500


def evaluate_flight_plan(
    state: AircraftState,
    progress: RouteProgress,
    config: FlightPlanTaskConfig,
) -> TaskEvaluation:
    """Evaluate reward, progress, and terminations for waypoint tracking."""
    heading_error_rad = signed_smallest_angle(progress.desired_track_rad, state.heading_rad)
    stall_risk_value = stall_risk(state, config.stall_speed_mps)
    envelope = envelope_violation(state)

    phase = FlightPlanPhase.ENROUTE
    if progress.completed_waypoint and not progress.mission_complete:
        phase = FlightPlanPhase.WAYPOINT_CAPTURE
    if progress.mission_complete:
        phase = FlightPlanPhase.MISSION_COMPLETE

    reward_breakdown = {
        "cross_track": safe_reward_component(
            progress.cross_track_error_m, config.max_cross_track_error_m
        ),
        "altitude": safe_reward_component(progress.altitude_error_m, config.max_altitude_error_m),
        "speed": safe_reward_component(progress.speed_error_mps, 12.0),
        "heading": safe_reward_component(heading_error_rad, config.max_heading_error_rad),
        "smoothness": 1.0
        - min(
            (abs(state.elevator) + abs(state.aileron) + abs(state.rudder)) / 3.0,
            1.0,
        ),
        "waypoint_bonus": config.waypoint_bonus if progress.completed_waypoint else 0.0,
        "safety": -1.5 * stall_risk_value - float(envelope),
    }
    reward = (
        0.24 * reward_breakdown["cross_track"]
        + 0.22 * reward_breakdown["altitude"]
        + 0.18 * reward_breakdown["speed"]
        + 0.16 * reward_breakdown["heading"]
        + 0.1 * reward_breakdown["smoothness"]
        + reward_breakdown["waypoint_bonus"]
        + reward_breakdown["safety"]
    )
    if progress.mission_complete:
        reward += config.completion_bonus

    return TaskEvaluation(
        reward=reward,
        phase=phase.value,
        reward_breakdown=reward_breakdown,
        terminated=progress.mission_complete or envelope,
        success=progress.mission_complete,
        safety_flags={
            "stall": stall_risk_value >= 0.9,
            "runway_excursion": False,
            "envelope_violation": envelope,
            "crash": False,
        },
        metrics={
            "cross_track_error_m": progress.cross_track_error_m,
            "altitude_error_m": progress.altitude_error_m,
            "speed_error_mps": progress.speed_error_mps,
            "heading_error_rad": heading_error_rad,
            "stall_risk": stall_risk_value,
            "waypoint_index": float(progress.waypoint_index),
        },
    )
