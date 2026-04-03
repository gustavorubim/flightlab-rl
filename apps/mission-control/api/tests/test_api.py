from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml
from app.main import create_app
from fastapi.testclient import TestClient


def _write_registry(path: Path) -> None:
    payload = {
        "controller_modes": {
            "pid": {
                "label": "PID",
                "description": "Deterministic PID mission driver.",
            },
            "rl_phase_switched": {
                "label": "RL",
                "description": "Phase-switched RL checkpoints.",
                "takeoff": {
                    "label": "Takeoff RL",
                    "description": "Takeoff checkpoint.",
                    "task": "takeoff",
                    "algorithm": "ppo",
                    "model_path": "missing_takeoff",
                },
                "flight_plan": {
                    "label": "Flight Plan RL",
                    "description": "Flight plan checkpoint.",
                    "task": "flight_plan",
                    "algorithm": "ppo",
                    "model_path": "missing_flight_plan",
                },
            },
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    registry_path = tmp_path / "controllers.yaml"
    _write_registry(registry_path)
    monkeypatch.setenv("MISSION_CONTROL_CONTROLLER_REGISTRY", str(registry_path))
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_start_takeoff_and_websocket(client: TestClient) -> None:
    start_response = client.post("/api/session/start", json={"controller_mode": "pid"})
    assert start_response.status_code == 200
    takeoff_response = client.post("/api/commands/takeoff")
    assert takeoff_response.status_code == 200
    with client.websocket_connect("/ws/telemetry") as websocket:
        initial = websocket.receive_json()
        assert initial["session"]["controller_mode"] == "pid"
        assert "aircraft" in initial
        time.sleep(0.25)
        follow_up = websocket.receive_json()
        assert follow_up["session"]["phase"] in {"takeoff_roll", "climb_out", "enroute"}


def test_pause_resume_and_replan(client: TestClient) -> None:
    client.post("/api/session/start", json={"controller_mode": "pid"})
    client.post("/api/commands/takeoff")
    time.sleep(1.2)
    pause_response = client.post("/api/commands/pause")
    assert pause_response.status_code == 200
    mission_response = client.put(
        "/api/mission",
        json={
            "name": "paused-divert",
            "waypoints": [
                {
                    "name": "d1",
                    "x_m": 520.0,
                    "y_m": 40.0,
                    "altitude_m": 165.0,
                    "target_airspeed_mps": 27.0,
                    "acceptance_radius_m": 35.0,
                }
            ],
        },
    )
    assert mission_response.status_code == 200
    snapshot = client.get("/api/session")
    assert snapshot.status_code == 200
    assert snapshot.json()["mission"]["mission_name"] == "paused-divert"
    resume_response = client.post("/api/commands/resume")
    assert resume_response.status_code == 200
