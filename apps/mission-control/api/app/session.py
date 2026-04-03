"""Real-time mission session runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import HTTPException

from app.config import ControllerRegistry
from app.controllers import PIDMissionPilot, RLPhaseSwitchedPilot
from app.schemas import (
    AircraftTelemetryModel,
    CommandResponseModel,
    ControllerOptionModel,
    MissionModel,
    MissionStateModel,
    RuntimeHealthModel,
    RunwayModel,
    SessionMetaModel,
    SessionSnapshotModel,
    SessionStartRequest,
    TrailPointModel,
)
from flightlab.core.types import AircraftState
from flightlab.dynamics import DynamicsConfig, KinematicDynamics
from flightlab.envs.flight_plan import default_mission
from flightlab.envs.takeoff import default_takeoff_runway
from flightlab.guidance.route import RouteManager
from flightlab.tasks.flight_plan import FlightPlanTaskConfig, evaluate_flight_plan
from flightlab.tasks.takeoff import TakeoffTaskConfig, evaluate_takeoff
from flightlab.world.mission import Mission
from flightlab.world.runway import Runway


class SessionPhase(StrEnum):
    """Mission-control session phase."""

    STANDBY = "standby"
    TAKEOFF_ROLL = "takeoff_roll"
    CLIMB_OUT = "climb_out"
    ENROUTE = "enroute"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MissionSession:
    """Single live mission session."""

    session_id: str
    controller_mode: str
    runway: Runway
    mission: Mission
    pilot: PIDMissionPilot | RLPhaseSwitchedPilot
    dynamics: KinematicDynamics
    phase: SessionPhase = SessionPhase.STANDBY
    paused: bool = False
    resume_phase: SessionPhase = SessionPhase.STANDBY
    takeoff_armed: bool = False
    route_manager: RouteManager = field(init=False)
    trail: deque[TrailPointModel] = field(default_factory=lambda: deque(maxlen=180))
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    last_command_status: str = "ready"
    last_error: str | None = None
    last_route_progress: object | None = None
    last_takeoff_evaluation: object | None = None

    def __post_init__(self) -> None:
        self.route_manager = RouteManager(self.mission)

    @property
    def state(self) -> AircraftState:
        return self.dynamics.state


class MissionRuntimeService:
    """Own and advance the single live mission session."""

    def __init__(
        self,
        *,
        controller_registry_path: str | Path,
        tick_hz: float = 20.0,
        trail_points: int = 180,
    ) -> None:
        self._registry_path = Path(controller_registry_path)
        self._registry = ControllerRegistry.from_path(self._registry_path)
        self._tick_interval_s = 1.0 / tick_hz
        self._trail_points = trail_points
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._session = self._build_session(
            controller_mode="pid",
            mission=default_mission(),
            runway=default_takeoff_runway(),
        )

    async def startup(self) -> None:
        """Start the background simulation loop."""
        if self._task is None:
            self._shutdown_event.clear()
            self._task = asyncio.create_task(self._run_loop(), name="mission-control-runtime")

    async def shutdown(self) -> None:
        """Stop the background simulation loop."""
        self._shutdown_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run_loop(self) -> None:
        """Advance the live session at the configured tick rate."""
        while not self._shutdown_event.is_set():
            started_at = perf_counter()
            async with self._lock:
                self._tick_session()
            elapsed_s = perf_counter() - started_at
            await asyncio.sleep(max(self._tick_interval_s - elapsed_s, 0.0))

    async def get_snapshot(self) -> SessionSnapshotModel:
        """Return the current session snapshot."""
        async with self._lock:
            return self._snapshot_locked()

    async def list_controllers(self) -> list[ControllerOptionModel]:
        """Return the controller modes visible to the frontend."""
        return [
            ControllerOptionModel(
                mode="pid",
                label=self._registry.pid.label,
                description=self._registry.pid.description,
                available=True,
            ),
            ControllerOptionModel(
                mode="rl_phase_switched",
                label=self._registry.rl_phase_switched.label,
                description=self._registry.rl_phase_switched.description,
                available=self._registry.rl_phase_switched.available,
                details={
                    "takeoff": self._registry.rl_phase_switched.takeoff.label,
                    "flight_plan": self._registry.rl_phase_switched.flight_plan.label,
                },
            ),
        ]

    async def start_session(self, request: SessionStartRequest) -> CommandResponseModel:
        """Create a fresh mission session."""
        mission = request.mission.to_mission() if request.mission is not None else default_mission()
        runway = (
            request.runway.to_runway() if request.runway is not None else default_takeoff_runway()
        )
        async with self._lock:
            self._session = self._build_session(
                controller_mode=request.controller_mode,
                mission=mission,
                runway=runway,
            )
            self._append_event("Session initialized and waiting on runway.")
            return self._command_response_locked("session started")

    async def reset_session(self) -> CommandResponseModel:
        """Reset the current session to runway-ready standby."""
        async with self._lock:
            current = self._session
            self._session = self._build_session(
                controller_mode=current.controller_mode,
                mission=current.mission,
                runway=current.runway,
            )
            self._append_event("Session reset to runway-ready standby.")
            return self._command_response_locked("session reset")

    async def arm_takeoff(self) -> CommandResponseModel:
        """Arm takeoff from the standby phase."""
        async with self._lock:
            if self._session.phase is not SessionPhase.STANDBY:
                self._fail_command("Takeoff can only be triggered from standby.")
            self._session.takeoff_armed = True
            self._session.last_command_status = "takeoff armed"
            self._append_event("Takeoff command armed.")
            return self._command_response_locked("takeoff armed")

    async def pause(self) -> CommandResponseModel:
        """Pause the live simulation."""
        async with self._lock:
            if self._session.phase in {
                SessionPhase.COMPLETED,
                SessionPhase.FAILED,
                SessionPhase.PAUSED,
            }:
                self._fail_command("Pause is only valid during an active session.")
            self._session.resume_phase = self._session.phase
            self._session.phase = SessionPhase.PAUSED
            self._session.paused = True
            self._session.last_command_status = "paused"
            self._append_event("Session paused.")
            return self._command_response_locked("paused")

    async def resume(self) -> CommandResponseModel:
        """Resume a paused live simulation."""
        async with self._lock:
            if self._session.phase is not SessionPhase.PAUSED:
                self._fail_command("Resume is only valid when the session is paused.")
            self._session.phase = self._session.resume_phase
            self._session.paused = False
            self._session.last_command_status = "running"
            self._append_event("Session resumed.")
            return self._command_response_locked("resumed")

    async def replace_mission(self, mission_model: MissionModel) -> CommandResponseModel:
        """Replace the active mission and immediately divert to the new route."""
        async with self._lock:
            active_phase = (
                self._session.resume_phase
                if self._session.phase is SessionPhase.PAUSED
                else self._session.phase
            )
            if self._session.controller_mode == "rl_phase_switched":
                if active_phase not in {SessionPhase.STANDBY, SessionPhase.ENROUTE}:
                    self._fail_command(
                        "RL mission updates are only supported in standby, "
                        "enroute, or paused from those phases."
                    )
            elif active_phase in {SessionPhase.COMPLETED, SessionPhase.FAILED}:
                self._fail_command("Mission updates are not supported after the session has ended.")
            mission = mission_model.to_mission()
            self._session.mission = mission
            self._session.route_manager = RouteManager(mission)
            self._session.last_route_progress = None
            self._session.last_command_status = "mission updated"
            self._append_event(f"Mission replanned to '{mission.name}'.")
            return self._command_response_locked("mission updated")

    async def tick_once_for_test(self) -> None:
        """Advance the live session one tick for direct unit tests."""
        async with self._lock:
            self._tick_session()

    def _build_session(
        self,
        *,
        controller_mode: str,
        mission: Mission,
        runway: Runway,
    ) -> MissionSession:
        if controller_mode == "pid":
            pilot = PIDMissionPilot()
        elif controller_mode == "rl_phase_switched":
            if not self._registry.rl_phase_switched.available:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "RL phase-switched mode is unavailable because "
                        "one or more checkpoints are missing."
                    ),
                )
            pilot = RLPhaseSwitchedPilot(self._registry.rl_phase_switched)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported controller mode '{controller_mode}'."
            )
        dynamics = KinematicDynamics(
            DynamicsConfig(
                dt_s=self._tick_interval_s,
                runway_elevation_m=runway.elevation_m,
                lift_off_speed_mps=TakeoffTaskConfig().rotation_speed_mps,
            )
        )
        dynamics.reset(self._initial_state(runway))
        session = MissionSession(
            session_id=uuid4().hex[:8],
            controller_mode=controller_mode,
            runway=runway,
            mission=mission,
            pilot=pilot,
            dynamics=dynamics,
        )
        session.trail = deque(maxlen=self._trail_points)
        session.pilot.reset()
        self._record_trail(session)
        return session

    def _initial_state(self, runway: Runway) -> AircraftState:
        """Return the deterministic runway-ready start state."""
        return AircraftState(
            position_x_m=runway.threshold_x_m - 10.0,
            position_y_m=runway.threshold_y_m,
            altitude_m=runway.elevation_m,
            roll_rad=0.0,
            pitch_rad=0.0,
            heading_rad=runway.heading_rad,
            u_mps=1.0,
            v_mps=0.0,
            w_mps=0.0,
            p_radps=0.0,
            q_radps=0.0,
            r_radps=0.0,
            airspeed_mps=1.0,
            groundspeed_mps=1.0,
            vertical_speed_mps=0.0,
            angle_of_attack_rad=0.0,
            sideslip_rad=0.0,
            throttle=0.0,
            elevator=0.0,
            aileron=0.0,
            rudder=0.0,
            on_ground=True,
        )

    def _tick_session(self) -> None:
        """Advance the live session by one tick."""
        session = self._session
        if session.phase in {SessionPhase.COMPLETED, SessionPhase.FAILED, SessionPhase.PAUSED}:
            return
        if session.phase is SessionPhase.STANDBY and not session.takeoff_armed:
            session.last_route_progress = self._route_progress(session)
            return
        if session.phase is SessionPhase.STANDBY and session.takeoff_armed:
            self._set_phase(session, SessionPhase.TAKEOFF_ROLL, "Takeoff roll started.")

        route_progress = self._route_progress(session)
        command = session.pilot.command(
            session.state,
            phase=session.phase.value,
            runway=session.runway,
            route_progress=route_progress,
            takeoff_config=TakeoffTaskConfig(),
            dt_s=session.dynamics.config.dt_s,
        )
        session.dynamics.step(command)
        self._record_trail(session)
        route_progress = self._route_progress(session)

        if session.phase in {SessionPhase.TAKEOFF_ROLL, SessionPhase.CLIMB_OUT}:
            evaluation = evaluate_takeoff(session.state, session.runway, TakeoffTaskConfig())
            session.last_takeoff_evaluation = evaluation
            if evaluation.success:
                self._set_phase(
                    session, SessionPhase.ENROUTE, "Climb-out complete; entering enroute mission."
                )
                session.last_command_status = "enroute"
                return
            if evaluation.terminated:
                self._session_failed(session, self._failure_reason(evaluation.safety_flags))
                return
            altitude_agl_m = max(session.state.altitude_m - session.runway.elevation_m, 0.0)
            next_phase = (
                SessionPhase.CLIMB_OUT
                if altitude_agl_m > 1.5 and not session.state.on_ground
                else SessionPhase.TAKEOFF_ROLL
            )
            self._set_phase(session, next_phase)
            session.last_command_status = "takeoff active"
            return

        evaluation = evaluate_flight_plan(session.state, route_progress, FlightPlanTaskConfig())
        if evaluation.success:
            self._set_phase(session, SessionPhase.COMPLETED, "Mission complete.")
            session.last_command_status = "completed"
            return
        if evaluation.terminated:
            self._session_failed(session, self._failure_reason(evaluation.safety_flags))
            return
        self._set_phase(session, SessionPhase.ENROUTE)
        session.last_command_status = "tracking mission"

    def _route_progress(self, session: MissionSession) -> object:
        """Compute route progress for the session's current mission."""
        progress = session.route_manager.progress(
            session.state.position_x_m,
            session.state.position_y_m,
            session.state.altitude_m,
            session.state.airspeed_mps,
        )
        session.last_route_progress = progress
        return progress

    def _record_trail(self, session: MissionSession) -> None:
        """Append the latest aircraft state to the short trail."""
        session.trail.append(
            TrailPointModel(
                position_x_m=session.state.position_x_m,
                position_y_m=session.state.position_y_m,
                altitude_m=session.state.altitude_m,
                sim_time_s=session.state.time_s,
            )
        )

    def _set_phase(
        self,
        session: MissionSession,
        phase: SessionPhase,
        event: str | None = None,
    ) -> None:
        """Update the session phase and log transitions once."""
        if session.phase is phase:
            return
        session.phase = phase
        session.paused = phase is SessionPhase.PAUSED
        if event is not None:
            self._append_event(event)

    def _session_failed(self, session: MissionSession, reason: str) -> None:
        """Mark the current session as failed."""
        session.last_error = reason
        session.last_command_status = "failed"
        self._set_phase(session, SessionPhase.FAILED, f"Mission failed: {reason}")

    def _append_event(self, message: str) -> None:
        """Append a timestamped event to the current session log."""
        session = self._session
        session.recent_events.appendleft(f"T+{session.state.time_s:05.1f}s {message}")

    def _snapshot_locked(self) -> SessionSnapshotModel:
        """Build the current API snapshot from the in-memory session."""
        session = self._session
        route_progress = session.last_route_progress or self._route_progress(session)
        return SessionSnapshotModel(
            session=SessionMetaModel(
                session_id=session.session_id,
                phase=session.phase.value,
                controller_mode=session.controller_mode,
                sim_time_s=session.state.time_s,
                paused=session.paused,
            ),
            aircraft=AircraftTelemetryModel(
                position_x_m=session.state.position_x_m,
                position_y_m=session.state.position_y_m,
                altitude_m=session.state.altitude_m,
                roll_rad=session.state.roll_rad,
                pitch_rad=session.state.pitch_rad,
                heading_rad=session.state.heading_rad,
                airspeed_mps=session.state.airspeed_mps,
                groundspeed_mps=session.state.groundspeed_mps,
                vertical_speed_mps=session.state.vertical_speed_mps,
                on_ground=session.state.on_ground,
            ),
            mission=MissionStateModel(
                mission_name=session.mission.name,
                waypoints=MissionModel.from_mission(session.mission).waypoints,
                active_waypoint_index=int(route_progress.waypoint_index),
                desired_track_rad=float(route_progress.desired_track_rad),
                distance_to_waypoint_m=float(route_progress.distance_to_waypoint_m),
            ),
            runtime_health=RuntimeHealthModel(
                last_command_status=session.last_command_status,
                last_error=session.last_error,
                recent_events=list(session.recent_events),
            ),
            trail=list(session.trail),
            runway=RunwayModel.from_runway(session.runway),
        )

    def _command_response_locked(self, message: str) -> CommandResponseModel:
        """Build a standard command response using the current snapshot."""
        return CommandResponseModel(ok=True, message=message, session=self._snapshot_locked())

    def _fail_command(self, message: str) -> None:
        """Raise a command error and preserve the latest reason."""
        self._session.last_error = message
        raise HTTPException(status_code=409, detail=message)

    def _failure_reason(self, flags: dict[str, bool]) -> str:
        """Translate task safety flags into a short user-facing failure reason."""
        for key, value in flags.items():
            if value:
                return key.replace("_", " ")
        return "controller envelope violation"
