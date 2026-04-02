"""Waypoint route management."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flightlab.core.geometry import project_point_to_segment
from flightlab.world.mission import Mission, Waypoint


@dataclass(frozen=True)
class RouteProgress:
    """Snapshot of mission tracking state."""

    current_waypoint: Waypoint
    waypoint_index: int
    distance_to_waypoint_m: float
    cross_track_error_m: float
    altitude_error_m: float
    speed_error_mps: float
    completed_waypoint: bool
    mission_complete: bool
    desired_track_rad: float


class RouteManager:
    """Track progression through an ordered waypoint mission."""

    def __init__(self, mission: Mission) -> None:
        self.mission = mission
        self.reset()

    def reset(self) -> None:
        """Reset route progress to the first waypoint."""
        self._index = 0

    @property
    def current_waypoint(self) -> Waypoint:
        """Return the active target waypoint."""
        return self.mission.waypoints[self._index]

    def progress(
        self,
        x_m: float,
        y_m: float,
        altitude_m: float,
        airspeed_mps: float,
    ) -> RouteProgress:
        """Advance the route if needed and return tracking metrics."""
        current = self.current_waypoint
        previous = self.mission.waypoints[max(self._index - 1, 0)]

        def compute_metrics(
            target: Waypoint, previous_target: Waypoint
        ) -> tuple[float, float, float, float, float]:
            dx_m = target.x_m - x_m
            dy_m = target.y_m - y_m
            projection_x_m, projection_y_m, _ = project_point_to_segment(
                point=(x_m, y_m),
                segment_start=(previous_target.x_m, previous_target.y_m),
                segment_end=(target.x_m, target.y_m),
            )
            return (
                math.hypot(dx_m, dy_m),
                math.hypot(x_m - projection_x_m, y_m - projection_y_m),
                target.altitude_m - altitude_m,
                target.target_airspeed_mps - airspeed_mps,
                math.atan2(dy_m, dx_m),
            )

        (
            distance_to_waypoint_m,
            cross_track_error_m,
            altitude_error_m,
            speed_error_mps,
            desired_track_rad,
        ) = compute_metrics(current, previous)
        completed_waypoint = distance_to_waypoint_m <= current.acceptance_radius_m and abs(
            altitude_error_m
        ) <= max(10.0, current.acceptance_radius_m / 2.0)

        if completed_waypoint and self._index < len(self.mission.waypoints) - 1:
            self._index += 1
            current = self.current_waypoint
            previous = self.mission.waypoints[self._index - 1]
            (
                distance_to_waypoint_m,
                cross_track_error_m,
                altitude_error_m,
                speed_error_mps,
                desired_track_rad,
            ) = compute_metrics(current, previous)
        mission_complete = completed_waypoint and self._index == len(self.mission.waypoints) - 1
        return RouteProgress(
            current_waypoint=current,
            waypoint_index=self._index,
            distance_to_waypoint_m=distance_to_waypoint_m,
            cross_track_error_m=cross_track_error_m,
            altitude_error_m=altitude_error_m,
            speed_error_mps=speed_error_mps,
            completed_waypoint=completed_waypoint,
            mission_complete=mission_complete,
            desired_track_rad=desired_track_rad,
        )
