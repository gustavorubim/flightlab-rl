"""Unit conversion helpers."""

KNOT_TO_MPS = 0.514444
FOOT_TO_METER = 0.3048


def knots_to_mps(value: float) -> float:
    """Convert knots to meters per second."""
    return value * KNOT_TO_MPS


def mps_to_knots(value: float) -> float:
    """Convert meters per second to knots."""
    return value / KNOT_TO_MPS


def feet_to_meters(value: float) -> float:
    """Convert feet to meters."""
    return value * FOOT_TO_METER


def meters_to_feet(value: float) -> float:
    """Convert meters to feet."""
    return value / FOOT_TO_METER
