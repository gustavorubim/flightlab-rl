"""Geometry helpers used across guidance and tasks."""

from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to a closed interval."""
    return max(low, min(high, value))


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def signed_smallest_angle(reference_rad: float, value_rad: float) -> float:
    """Return the signed shortest angle from reference to value."""
    return wrap_angle_rad(value_rad - reference_rad)


def rotate_point_2d(x_m: float, y_m: float, angle_rad: float) -> tuple[float, float]:
    """Rotate a 2D point around the origin."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (x_m * cos_a - y_m * sin_a, x_m * sin_a + y_m * cos_a)


def project_point_to_segment(
    point: tuple[float, float],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> tuple[float, float, float]:
    """Project a point to a 2D line segment and return the closest point and normalized progress."""
    px, py = point
    sx, sy = segment_start
    ex, ey = segment_end
    dx = ex - sx
    dy = ey - sy
    norm_sq = dx * dx + dy * dy
    if norm_sq == 0.0:
        return sx, sy, 0.0
    t = clamp(((px - sx) * dx + (py - sy) * dy) / norm_sq, 0.0, 1.0)
    return sx + t * dx, sy + t * dy, t
