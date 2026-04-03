from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from app.config import ControllerRegistry
from app.schemas import MissionModel, SessionStartRequest, WaypointModel
from app.session import MissionRuntimeService, SessionPhase
from fastapi import HTTPException


def _write_registry(
    path: Path,
    *,
    takeoff_model: str = "missing_takeoff",
    flight_plan_model: str = "missing_flight_plan",
) -> None:
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
                    "model_path": takeoff_model,
                },
                "flight_plan": {
                    "label": "Flight Plan RL",
                    "description": "Flight plan checkpoint.",
                    "task": "flight_plan",
                    "algorithm": "ppo",
                    "model_path": flight_plan_model,
                },
            },
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_pid_session_transitions_to_enroute(tmp_path: Path) -> None:
    registry_path = tmp_path / "controllers.yaml"
    _write_registry(registry_path)
    runtime = MissionRuntimeService(controller_registry_path=registry_path, tick_hz=20.0)
    await runtime.startup()
    try:
        await runtime.start_session(SessionStartRequest(controller_mode="pid"))
        await runtime.arm_takeoff()
        for _ in range(300):
            await runtime.tick_once_for_test()
            snapshot = await runtime.get_snapshot()
            if snapshot.session.phase == SessionPhase.ENROUTE.value:
                break
        assert snapshot.session.phase == SessionPhase.ENROUTE.value
        assert snapshot.aircraft.altitude_m > snapshot.runway.elevation_m
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_replacing_mission_in_enroute_updates_active_waypoint(tmp_path: Path) -> None:
    registry_path = tmp_path / "controllers.yaml"
    _write_registry(registry_path)
    runtime = MissionRuntimeService(controller_registry_path=registry_path, tick_hz=20.0)
    await runtime.startup()
    try:
        await runtime.start_session(SessionStartRequest(controller_mode="pid"))
        await runtime.arm_takeoff()
        for _ in range(300):
            await runtime.tick_once_for_test()
            if (await runtime.get_snapshot()).session.phase == SessionPhase.ENROUTE.value:
                break
        new_mission = MissionModel(
            name="divert",
            waypoints=[
                WaypointModel(
                    name="divert-1",
                    x_m=600.0,
                    y_m=50.0,
                    altitude_m=170.0,
                    target_airspeed_mps=27.0,
                )
            ],
        )
        response = await runtime.replace_mission(new_mission)
        assert response.ok is True
        snapshot = await runtime.get_snapshot()
        assert snapshot.mission.mission_name == "divert"
        assert snapshot.mission.waypoints[0].name == "divert-1"
        assert snapshot.mission.active_waypoint_index == 0
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_rl_mode_rejects_missing_checkpoints(tmp_path: Path) -> None:
    registry_path = tmp_path / "controllers.yaml"
    _write_registry(registry_path)
    runtime = MissionRuntimeService(controller_registry_path=registry_path)
    with pytest.raises(HTTPException, match="unavailable"):
        await runtime.start_session(SessionStartRequest(controller_mode="rl_phase_switched"))


@pytest.mark.asyncio
async def test_invalid_phase_commands_raise_conflict(tmp_path: Path) -> None:
    registry_path = tmp_path / "controllers.yaml"
    _write_registry(registry_path)
    runtime = MissionRuntimeService(controller_registry_path=registry_path)
    await runtime.start_session(SessionStartRequest(controller_mode="pid"))
    await runtime.pause()
    with pytest.raises(HTTPException, match="Pause is only valid"):
        await runtime.pause()
    await runtime.resume()
    with pytest.raises(HTTPException, match="Resume is only valid"):
        await runtime.resume()


def test_controller_registry_reports_missing_rl_checkpoints(tmp_path: Path) -> None:
    registry_path = tmp_path / "controllers.yaml"
    _write_registry(registry_path)
    registry = ControllerRegistry.from_path(registry_path)
    assert registry.pid.label == "PID"
    assert registry.rl_phase_switched.available is False
