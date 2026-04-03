from __future__ import annotations

from flightlab.guidance.route import RouteProgress
from flightlab.tasks.flight_plan import FlightPlanPhase, FlightPlanTaskConfig, evaluate_flight_plan
from flightlab.tasks.landing import LandingPhase, LandingTaskConfig, evaluate_landing
from flightlab.tasks.takeoff import (
    TakeoffPhase,
    TakeoffTaskConfig,
    classify_takeoff_phase,
    evaluate_takeoff,
)
from flightlab.world.mission import Waypoint
from flightlab.world.runway import Runway


def test_takeoff_phase_transitions_and_success(make_state) -> None:
    runway = Runway(name="09", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)
    config = TakeoffTaskConfig()
    taxi_state = make_state(on_ground=True, airspeed_mps=0.0, groundspeed_mps=0.0, altitude_m=120.0)
    rotate_state = make_state(on_ground=True, airspeed_mps=25.0, altitude_m=120.0, pitch_rad=0.1)
    climb_state = make_state(
        on_ground=False,
        airspeed_mps=28.0,
        altitude_m=155.0,
        vertical_speed_mps=4.0,
        pitch_rad=0.15,
    )
    assert classify_takeoff_phase(taxi_state, runway, config) is TakeoffPhase.TAXI_ALIGN
    assert classify_takeoff_phase(rotate_state, runway, config) is TakeoffPhase.ROTATE
    evaluation = evaluate_takeoff(climb_state, runway, config)
    assert evaluation.phase == TakeoffPhase.INITIAL_CLIMB.value
    assert evaluation.success is True
    assert evaluation.reward_breakdown["climb"] > 0.0


def test_takeoff_detects_excursion_and_stall(make_state) -> None:
    runway = Runway(name="09", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)
    state = make_state(
        position_y_m=25.0,
        airspeed_mps=10.0,
        angle_of_attack_rad=0.4,
        on_ground=True,
        altitude_m=120.0,
    )
    evaluation = evaluate_takeoff(state, runway, TakeoffTaskConfig())
    assert evaluation.terminated is True
    assert evaluation.safety_flags["runway_excursion"] is True
    assert evaluation.safety_flags["stall"] is True


def test_takeoff_fails_grounded_overspeed_without_liftoff(make_state) -> None:
    runway = Runway(name="09", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)
    state = make_state(
        altitude_m=120.0,
        airspeed_mps=29.0,
        groundspeed_mps=29.0,
        pitch_rad=0.02,
        vertical_speed_mps=0.0,
        on_ground=True,
    )
    evaluation = evaluate_takeoff(state, runway, TakeoffTaskConfig())
    assert evaluation.terminated is True
    assert evaluation.safety_flags["failed_liftoff"] is True
    assert evaluation.reward_breakdown["delayed_rotation"] < 0.0
    assert evaluation.reward < 0.0


def test_landing_phase_rewards_and_hard_landing(make_state) -> None:
    runway = Runway(name="27", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)
    approach_state = make_state(position_x_m=-400.0, altitude_m=145.0, airspeed_mps=25.0)
    evaluation = evaluate_landing(approach_state, runway, LandingTaskConfig())
    assert evaluation.phase == LandingPhase.APPROACH.value

    final_state = make_state(
        position_x_m=-100.0,
        altitude_m=130.0,
        airspeed_mps=22.0,
        pitch_rad=-0.05,
    )
    final_eval = evaluate_landing(final_state, runway, LandingTaskConfig())
    assert final_eval.phase == LandingPhase.FINAL.value

    flare_state = make_state(
        position_x_m=-20.0,
        altitude_m=124.0,
        airspeed_mps=18.0,
        pitch_rad=0.02,
    )
    flare_eval = evaluate_landing(flare_state, runway, LandingTaskConfig())
    assert flare_eval.phase == LandingPhase.FLARE.value

    touchdown_state = make_state(
        position_x_m=100.0,
        altitude_m=120.0,
        airspeed_mps=18.0,
        groundspeed_mps=18.0,
        on_ground=True,
    )
    hard = evaluate_landing(
        touchdown_state,
        runway,
        LandingTaskConfig(),
        touchdown_step=True,
        touchdown_sink_rate_mps=3.5,
    )
    assert hard.phase == LandingPhase.TOUCHDOWN.value
    assert hard.safety_flags["hard_landing"] is True
    rollout = evaluate_landing(
        make_state(
            position_x_m=50.0,
            altitude_m=120.0,
            on_ground=True,
            groundspeed_mps=3.0,
            airspeed_mps=3.0,
        ),
        runway,
        LandingTaskConfig(),
    )
    assert rollout.phase == LandingPhase.ROLLOUT.value
    assert rollout.success is True


def test_flight_plan_evaluation_supports_waypoint_capture_and_completion(make_state) -> None:
    progress = RouteProgress(
        current_waypoint=Waypoint("b", 100.0, 0.0, 110.0, 26.0),
        waypoint_index=1,
        distance_to_waypoint_m=10.0,
        cross_track_error_m=5.0,
        altitude_error_m=2.0,
        speed_error_mps=1.0,
        completed_waypoint=True,
        mission_complete=False,
        desired_track_rad=0.0,
    )
    evaluation = evaluate_flight_plan(make_state(), progress, FlightPlanTaskConfig())
    assert evaluation.phase == FlightPlanPhase.WAYPOINT_CAPTURE.value
    assert evaluation.reward_breakdown["waypoint_bonus"] > 0.0

    completion_progress = RouteProgress(
        current_waypoint=Waypoint("c", 120.0, 0.0, 110.0, 26.0),
        waypoint_index=2,
        distance_to_waypoint_m=0.0,
        cross_track_error_m=0.0,
        altitude_error_m=0.0,
        speed_error_mps=0.0,
        completed_waypoint=True,
        mission_complete=True,
        desired_track_rad=0.0,
    )
    complete_eval = evaluate_flight_plan(make_state(), completion_progress, FlightPlanTaskConfig())
    assert complete_eval.phase == FlightPlanPhase.MISSION_COMPLETE.value
    assert complete_eval.success is True
