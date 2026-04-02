"""Approach and glideslope utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flightlab.world.runway import Runway


@dataclass(frozen=True)
class GlideslopeReference:
    """Reference model for a constant-angle runway approach."""

    runway: Runway
    glide_angle_deg: float = 3.0

    @property
    def glide_angle_rad(self) -> float:
        """Return the glide angle in radians."""
        return math.radians(self.glide_angle_deg)

    def target_altitude_m(self, along_track_m: float) -> float:
        """Return the target altitude above mean sea level for a runway-relative point."""
        distance_before_threshold_m = max(-along_track_m, 0.0)
        return (
            self.runway.elevation_m + math.tan(self.glide_angle_rad) * distance_before_threshold_m
        )

    def altitude_error_m(self, along_track_m: float, altitude_m: float) -> float:
        """Return altitude deviation from the glideslope."""
        return self.target_altitude_m(along_track_m) - altitude_m
