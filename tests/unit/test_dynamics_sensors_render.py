from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from flightlab.core.types import ControlCommand
from flightlab.dynamics.base import DynamicsConfig
from flightlab.dynamics.jsbsim_adapter import JSBSimDynamics
from flightlab.dynamics.kinematic import KinematicDynamics
from flightlab.render.replay import EpisodeRecorder
from flightlab.render.video import render_episode_video
from flightlab.sensors.observation import ObservationBuilder


def test_observation_builder_and_aircraft_vector(make_state) -> None:
    state = make_state()
    builder = ObservationBuilder()
    observation = builder.build(
        state,
        task_delta_x_m=10.0,
        task_delta_y_m=-5.0,
        task_delta_altitude_m=2.0,
        task_target_speed_error_mps=1.5,
        altitude_agl_m=20.0,
    )
    assert observation.shape == (25,)
    assert builder.feature_names[-1] == "altitude_agl_m"


def test_kinematic_dynamics_step_is_deterministic(make_state) -> None:
    config = DynamicsConfig(dt_s=0.1, runway_elevation_m=120.0, lift_off_speed_mps=24.0)
    dynamics = KinematicDynamics(config)
    initial = make_state(
        altitude_m=120.0,
        on_ground=True,
        airspeed_mps=5.0,
        groundspeed_mps=5.0,
        throttle=0.2,
    )
    dynamics.reset(initial)
    for _ in range(30):
        state = dynamics.step(ControlCommand(elevator=0.2, aileron=0.0, rudder=0.0, throttle=1.0))
    assert state.time_s == pytest.approx(3.0)
    assert state.airspeed_mps > initial.airspeed_mps
    assert state.position_x_m > initial.position_x_m
    assert state.altitude_m >= config.runway_elevation_m


def test_kinematic_dynamics_requires_reset() -> None:
    dynamics = KinematicDynamics()
    try:
        _ = dynamics.state
    except RuntimeError as exc:
        assert "reset" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError")


def test_jsbsim_adapter_raises_when_optional_dependency_missing(monkeypatch) -> None:
    monkeypatch.setattr("flightlab.dynamics.jsbsim_adapter.jsbsim", None)
    try:
        JSBSimDynamics()
    except RuntimeError as exc:
        assert "JSBSim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError")


def test_episode_recorder_exports_json(tmp_path, make_state) -> None:
    recorder = EpisodeRecorder()
    state = make_state()
    recorder.record_reset(state, {"task_phase": "RESET"})
    recorder.record_step(state, [0.0, 0.0, 0.0, 0.5], 1.0, {"task_phase": "STEP"})
    path = recorder.export_json(tmp_path / "replay.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["kind"] == "reset"
    assert payload[1]["reward"] == 1.0


def test_render_episode_video_uses_ffmpeg(monkeypatch, tmp_path, make_state) -> None:
    recorder = EpisodeRecorder()
    state = make_state()
    recorder.record_reset(state, {"task_phase": "RESET", "task_name": "takeoff"})
    recorder.record_step(
        make_state(position_x_m=10.0, position_y_m=5.0, heading_rad=0.2),
        [0.1, 0.0, 0.0, 0.8],
        0.5,
        {"task_phase": "TAKEOFF_ROLL", "task_name": "takeoff", "reward": 0.5},
    )

    class FakeProcess:
        def __init__(self, command):
            class FakeStdin(io.BytesIO):
                def close(self) -> None:
                    self.flush()

            self.command = command
            self.stdin = FakeStdin()
            self.stderr = io.BytesIO()
            self._output_path = Path(command[-1])

        def wait(self) -> int:
            self._output_path.write_bytes(b"fake-mp4")
            return 0

    fake_processes: list[FakeProcess] = []

    def fake_popen(command, stdin, stdout, stderr):
        process = FakeProcess(command)
        fake_processes.append(process)
        return process

    monkeypatch.setattr("flightlab.render.video.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("flightlab.render.video.subprocess.Popen", fake_popen)
    output_path = render_episode_video(
        recorder.as_list(),
        tmp_path / "rollout.mp4",
        task_name="takeoff",
    )
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake-mp4"
    assert fake_processes
    assert len(fake_processes[0].stdin.getvalue()) > 0


def test_render_episode_video_requires_ffmpeg(monkeypatch, make_state) -> None:
    recorder = EpisodeRecorder()
    recorder.record_reset(make_state(), {"task_phase": "RESET"})
    monkeypatch.setattr("flightlab.render.video.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError):
        render_episode_video(recorder.as_list(), "ignored.mp4")
