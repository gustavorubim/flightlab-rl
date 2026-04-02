from __future__ import annotations

import math

import numpy as np

from flightlab.core.geometry import (
    clamp,
    project_point_to_segment,
    rotate_point_2d,
    signed_smallest_angle,
    wrap_angle_rad,
)
from flightlab.core.seed import seeded_rng
from flightlab.core.time import SimulationClock
from flightlab.core.types import ControlCommand
from flightlab.core.units import feet_to_meters, knots_to_mps, meters_to_feet, mps_to_knots


def test_unit_conversions_round_trip() -> None:
    assert math.isclose(knots_to_mps(100.0), 51.4444, rel_tol=1e-6)
    assert math.isclose(mps_to_knots(knots_to_mps(42.0)), 42.0, rel_tol=1e-6)
    assert math.isclose(feet_to_meters(1000.0), 304.8, rel_tol=1e-9)
    assert math.isclose(meters_to_feet(feet_to_meters(321.0)), 321.0, rel_tol=1e-9)


def test_geometry_helpers_cover_wrap_rotate_and_projection() -> None:
    assert clamp(3.0, 0.0, 2.0) == 2.0
    assert math.isclose(wrap_angle_rad(4.0), 4.0 - 2.0 * math.pi, rel_tol=1e-6)
    assert math.isclose(signed_smallest_angle(0.1, 0.2), 0.1, rel_tol=1e-6)
    x_m, y_m = rotate_point_2d(1.0, 0.0, math.pi / 2.0)
    assert math.isclose(x_m, 0.0, abs_tol=1e-6)
    assert math.isclose(y_m, 1.0, abs_tol=1e-6)
    px_m, py_m, progress = project_point_to_segment((2.0, 2.0), (0.0, 0.0), (4.0, 0.0))
    assert (px_m, py_m, progress) == (2.0, 0.0, 0.5)
    degenerate = project_point_to_segment((1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    assert degenerate == (0.0, 0.0, 0.0)


def test_seeded_rng_and_clock_are_deterministic() -> None:
    first = seeded_rng(7).normal(size=5)
    second = seeded_rng(7).normal(size=5)
    assert np.allclose(first, second)

    clock = SimulationClock(dt_s=0.2)
    assert clock.tick() == 0.2
    assert clock.step_count == 1
    clock.reset()
    assert clock.time_s == 0.0
    assert clock.step_count == 0


def test_control_command_clipping_and_array_round_trip() -> None:
    command = ControlCommand(elevator=2.0, aileron=-2.0, rudder=0.5, throttle=-0.5).clipped()
    assert command == ControlCommand(elevator=1.0, aileron=-1.0, rudder=0.5, throttle=0.0)
    restored = ControlCommand.from_array(command.as_array())
    assert restored == command
