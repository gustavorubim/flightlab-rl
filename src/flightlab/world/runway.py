"""Runway geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

from flightlab.core.geometry import rotate_point_2d, signed_smallest_angle


@dataclass(frozen=True)
class Runway:
    """Simple runway model in a local tangent plane."""

    name: str
    length_m: float
    width_m: float
    heading_rad: float
    threshold_x_m: float = 0.0
    threshold_y_m: float = 0.0
    elevation_m: float = 0.0

    def local_coordinates(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Return runway along-track and lateral coordinates."""
        dx = x_m - self.threshold_x_m
        dy = y_m - self.threshold_y_m
        along_m, lateral_m = rotate_point_2d(dx, dy, -self.heading_rad)
        return along_m, lateral_m

    def heading_error_rad(self, heading_rad: float) -> float:
        """Return signed heading error relative to runway heading."""
        return signed_smallest_angle(self.heading_rad, heading_rad)
