# ruff: noqa: E402

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flightlab.core.types import AircraftState


def _build_state(**overrides: float | bool) -> AircraftState:
    values = {
        "position_x_m": 0.0,
        "position_y_m": 0.0,
        "altitude_m": 120.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "heading_rad": 0.0,
        "u_mps": 20.0,
        "v_mps": 0.0,
        "w_mps": 0.0,
        "p_radps": 0.0,
        "q_radps": 0.0,
        "r_radps": 0.0,
        "airspeed_mps": 20.0,
        "groundspeed_mps": 20.0,
        "vertical_speed_mps": 0.0,
        "angle_of_attack_rad": 0.05,
        "sideslip_rad": 0.0,
        "throttle": 0.5,
        "elevator": 0.0,
        "aileron": 0.0,
        "rudder": 0.0,
        "mass_kg": 1200.0,
        "cg_offset_m": 0.0,
        "on_ground": False,
        "time_s": 0.0,
    }
    values.update(overrides)
    return AircraftState(**values)


@pytest.fixture
def make_state():
    def factory(**overrides: float | bool) -> AircraftState:
        return _build_state(**overrides)

    return factory
