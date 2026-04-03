"""Pydantic API schemas for the mission-control backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from flightlab.world.mission import Mission, Waypoint
from flightlab.world.runway import Runway


class RunwayModel(BaseModel):
    """Serializable runway definition."""

    name: str = "09"
    length_m: float = 900.0
    width_m: float = 30.0
    heading_rad: float = 0.0
    threshold_x_m: float = 0.0
    threshold_y_m: float = 0.0
    elevation_m: float = 120.0

    def to_runway(self) -> Runway:
        """Convert the payload to a flightlab runway."""
        return Runway(
            name=self.name,
            length_m=self.length_m,
            width_m=self.width_m,
            heading_rad=self.heading_rad,
            threshold_x_m=self.threshold_x_m,
            threshold_y_m=self.threshold_y_m,
            elevation_m=self.elevation_m,
        )

    @classmethod
    def from_runway(cls, runway: Runway) -> RunwayModel:
        """Build an API runway payload from a flightlab runway."""
        return cls(
            name=runway.name,
            length_m=runway.length_m,
            width_m=runway.width_m,
            heading_rad=runway.heading_rad,
            threshold_x_m=runway.threshold_x_m,
            threshold_y_m=runway.threshold_y_m,
            elevation_m=runway.elevation_m,
        )


class WaypointModel(BaseModel):
    """Serializable waypoint definition."""

    name: str
    x_m: float
    y_m: float
    altitude_m: float = Field(gt=0.0)
    target_airspeed_mps: float = Field(gt=0.0)
    acceptance_radius_m: float = Field(default=35.0, gt=0.0)

    def to_waypoint(self) -> Waypoint:
        """Convert the payload to a flightlab waypoint."""
        return Waypoint(
            name=self.name,
            x_m=self.x_m,
            y_m=self.y_m,
            altitude_m=self.altitude_m,
            target_airspeed_mps=self.target_airspeed_mps,
            acceptance_radius_m=self.acceptance_radius_m,
        )

    @classmethod
    def from_waypoint(cls, waypoint: Waypoint) -> WaypointModel:
        """Build an API waypoint payload from a flightlab waypoint."""
        return cls(
            name=waypoint.name,
            x_m=waypoint.x_m,
            y_m=waypoint.y_m,
            altitude_m=waypoint.altitude_m,
            target_airspeed_mps=waypoint.target_airspeed_mps,
            acceptance_radius_m=waypoint.acceptance_radius_m,
        )


class MissionModel(BaseModel):
    """Serializable mission definition."""

    name: str = "mission-control"
    waypoints: list[WaypointModel]

    @field_validator("waypoints")
    @classmethod
    def _validate_waypoints(cls, value: list[WaypointModel]) -> list[WaypointModel]:
        if not value:
            raise ValueError("Mission must contain at least one waypoint.")
        return value

    def to_mission(self) -> Mission:
        """Convert the payload to a flightlab mission."""
        return Mission(
            name=self.name, waypoints=tuple(item.to_waypoint() for item in self.waypoints)
        )

    @classmethod
    def from_mission(cls, mission: Mission) -> MissionModel:
        """Build an API mission payload from a flightlab mission."""
        return cls(
            name=mission.name,
            waypoints=[WaypointModel.from_waypoint(item) for item in mission.waypoints],
        )


class SessionStartRequest(BaseModel):
    """Request payload for starting a fresh mission session."""

    controller_mode: Literal["pid", "rl_phase_switched"] = "pid"
    mission: MissionModel | None = None
    runway: RunwayModel | None = None


class SessionMetaModel(BaseModel):
    """Session metadata exposed to the frontend."""

    session_id: str
    phase: str
    controller_mode: str
    sim_time_s: float
    paused: bool


class AircraftTelemetryModel(BaseModel):
    """Aircraft telemetry exposed to the frontend."""

    position_x_m: float
    position_y_m: float
    altitude_m: float
    roll_rad: float
    pitch_rad: float
    heading_rad: float
    airspeed_mps: float
    groundspeed_mps: float
    vertical_speed_mps: float
    on_ground: bool


class MissionStateModel(BaseModel):
    """Mission state exposed to the frontend."""

    mission_name: str
    waypoints: list[WaypointModel]
    active_waypoint_index: int
    desired_track_rad: float
    distance_to_waypoint_m: float


class RuntimeHealthModel(BaseModel):
    """Runtime health and event data exposed to the frontend."""

    last_command_status: str
    last_error: str | None
    recent_events: list[str]


class TrailPointModel(BaseModel):
    """Short historical trail point used for path rendering."""

    position_x_m: float
    position_y_m: float
    altitude_m: float
    sim_time_s: float


class SessionSnapshotModel(BaseModel):
    """Complete session snapshot shared by REST and websocket endpoints."""

    session: SessionMetaModel
    aircraft: AircraftTelemetryModel
    mission: MissionStateModel
    runtime_health: RuntimeHealthModel
    trail: list[TrailPointModel]
    runway: RunwayModel


class ControllerOptionModel(BaseModel):
    """Available controller modes exposed by the API."""

    mode: Literal["pid", "rl_phase_switched"]
    label: str
    description: str
    available: bool
    details: dict[str, str] = Field(default_factory=dict)


class CommandResponseModel(BaseModel):
    """Standard response for command-style endpoints."""

    ok: bool
    message: str
    session: SessionSnapshotModel
