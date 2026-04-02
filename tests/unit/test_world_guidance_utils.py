from __future__ import annotations

import math

import pytest

from flightlab.guidance.approach import GlideslopeReference
from flightlab.guidance.route import RouteManager
from flightlab.utils.config import load_yaml
from flightlab.world.mission import Mission, mission_from_dict, mission_from_path
from flightlab.world.runway import Runway


def test_runway_local_coordinates_and_heading_error() -> None:
    runway = Runway(name="09", length_m=800.0, width_m=30.0, heading_rad=math.pi / 2.0)
    along_m, lateral_m = runway.local_coordinates(10.0, 0.0)
    assert along_m == pytest.approx(0.0, abs=1e-6)
    assert lateral_m == pytest.approx(-10.0, abs=1e-6)
    assert runway.heading_error_rad(math.pi) == pytest.approx(math.pi / 2.0)


def test_mission_loading_and_route_progression(tmp_path) -> None:
    mission_payload = {
        "name": "demo",
        "waypoints": [
            {"name": "a", "x_m": 0.0, "y_m": 0.0, "altitude_m": 100.0, "target_airspeed_mps": 25.0},
            {
                "name": "b",
                "x_m": 100.0,
                "y_m": 0.0,
                "altitude_m": 110.0,
                "target_airspeed_mps": 26.0,
            },
        ],
    }
    mission = mission_from_dict(mission_payload)
    assert isinstance(mission, Mission)
    config_path = tmp_path / "mission.yaml"
    config_path.write_text(
        "name: demo\nwaypoints:\n"
        "  - name: a\n    x_m: 0\n    y_m: 0\n    altitude_m: 100\n    target_airspeed_mps: 25\n"
        "  - name: b\n    x_m: 100\n    y_m: 0\n    altitude_m: 110\n    target_airspeed_mps: 26\n",
        encoding="utf-8",
    )
    loaded = mission_from_path(config_path)
    route = RouteManager(loaded)
    progress_first = route.progress(0.0, 0.0, 100.0, 25.0)
    assert progress_first.completed_waypoint is True
    assert route.current_waypoint.name == "b"
    progress_second = route.progress(100.0, 0.0, 110.0, 26.0)
    assert progress_second.mission_complete is True
    assert progress_second.cross_track_error_m == pytest.approx(0.0)
    route.reset()
    assert route.current_waypoint.name == "a"
    assert load_yaml(config_path)["name"] == "demo"


def test_mission_rejects_empty_waypoints() -> None:
    with pytest.raises(ValueError):
        Mission(name="empty", waypoints=())


def test_glideslope_reference_computes_target_altitude() -> None:
    runway = Runway(name="27", length_m=900.0, width_m=30.0, heading_rad=0.0, elevation_m=120.0)
    glideslope = GlideslopeReference(runway=runway, glide_angle_deg=3.0)
    altitude = glideslope.target_altitude_m(-100.0)
    assert altitude > runway.elevation_m
    error = glideslope.altitude_error_m(-100.0, altitude)
    assert error == pytest.approx(0.0, abs=1e-6)
