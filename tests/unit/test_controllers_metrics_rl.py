from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import flightlab.rl.baselines as baselines
from flightlab.controllers.pid import PIDAutopilot, PIDController
from flightlab.metrics import summarize_episodes
from flightlab.rl.baselines import TrainingResult, load_model_class, train_baseline
from flightlab.rl.training_artifacts import (
    MonitorEpisodeRecord,
    load_monitor_records,
    moving_average,
    plot_training_curves,
    summarize_monitor_records,
)


def test_pid_controller_and_autopilot(make_state) -> None:
    controller = PIDController(kp=1.0, ki=0.5, kd=0.1)
    output = controller.update(0.2, 0.1)
    assert output > 0.0
    controller.reset()
    assert controller.integral == 0.0

    autopilot = PIDAutopilot()
    command = autopilot.command(
        make_state(sideslip_rad=0.1),
        heading_error_rad=0.2,
        altitude_error_m=10.0,
        speed_error_mps=3.0,
        dt_s=0.1,
    )
    assert 0.0 <= command.throttle <= 1.0
    assert -1.0 <= command.aileron <= 1.0


def test_summarize_episodes_handles_empty_and_non_empty_input() -> None:
    empty = summarize_episodes([])
    assert empty.success_rate == 0.0

    summary = summarize_episodes(
        [
            {
                "success": True,
                "crash": False,
                "stall": False,
                "runway_excursion": False,
                "average_cross_track_error_m": 2.0,
                "altitude_rmse_m": 1.0,
                "action_smoothness": 0.2,
                "episode_return": 5.0,
                "completion_time_s": 10.0,
            },
            {
                "success": False,
                "crash": True,
                "stall": True,
                "runway_excursion": False,
                "average_cross_track_error_m": 4.0,
                "altitude_rmse_m": 3.0,
                "action_smoothness": 0.4,
                "episode_return": 1.0,
                "completion_time_s": 12.0,
            },
        ]
    )
    assert summary.success_rate == 0.5
    assert summary.crash_rate == 0.5
    assert summary.average_return == 3.0

    partial = summarize_episodes([{}])
    assert partial.success_rate == 0.0
    assert partial.average_return == 0.0


def test_train_baseline_validates_algorithm_before_import() -> None:
    with pytest.raises(ValueError):
        train_baseline(algorithm="invalid", task="flight_plan", total_timesteps=10, seed=1)


def test_train_baseline_uses_fake_sb3(monkeypatch) -> None:
    @dataclass
    class FakeModel:
        policy: str
        env: object
        seed: int
        verbose: int

        def learn(self, total_timesteps: int) -> None:
            self.total_timesteps = total_timesteps

        def save(self, path: str) -> None:
            self.saved_path = path

    monkeypatch.setattr(
        "flightlab.rl.baselines._load_sb3",
        lambda: (FakeModel, FakeModel),
    )
    result = train_baseline(
        algorithm="ppo",
        task="takeoff",
        total_timesteps=12,
        seed=3,
        output_path="artifacts/fake_model",
    )
    assert isinstance(result, TrainingResult)
    assert result.model_class_name == "FakeModel"
    assert result.total_timesteps == 12
    assert result.output_path == "artifacts/fake_model"


def test_monitor_artifact_helpers(tmp_path) -> None:
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(
        '#{"t_start": 0.0, "env_id": "demo"}\n'
        "r,l,t,success,task_phase\n"
        "1.5,10,0.1,True,CLIMB\n"
        "-0.5,12,0.2,False,ROLL\n",
        encoding="utf-8",
    )
    records = load_monitor_records(monitor_path)
    assert records == [
        MonitorEpisodeRecord(reward=1.5, length=10.0, elapsed_time_s=0.1, success=True),
        MonitorEpisodeRecord(reward=-0.5, length=12.0, elapsed_time_s=0.2, success=False),
    ]
    assert moving_average([1.0, 2.0, 4.0], window=2) == [1.0, 1.5, 3.0]
    summary = summarize_monitor_records(records, reward_window=2)
    assert summary.episodes_logged == 2
    assert summary.final_episode_return == -0.5
    assert summary.final_window_return == pytest.approx(0.5)
    assert summary.final_window_success_rate == pytest.approx(0.5)


def test_plot_training_curves_creates_png(tmp_path) -> None:
    plot_path = tmp_path / "training.png"
    result = plot_training_curves(
        [
            MonitorEpisodeRecord(reward=-2.0, length=15.0, elapsed_time_s=0.1, success=False),
            MonitorEpisodeRecord(reward=3.0, length=11.0, elapsed_time_s=0.2, success=True),
            MonitorEpisodeRecord(reward=4.0, length=9.0, elapsed_time_s=0.3, success=True),
        ],
        plot_path,
        title="Demo Training",
    )
    assert result == str(plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_training_artifacts_handle_empty_inputs_and_plain_csv(tmp_path) -> None:
    monitor_path = tmp_path / "monitor_plain.csv"
    monitor_path.write_text(
        "r,l,t,success,task_phase\n"
        "2.5,4,0.1,False,ROLL\n",
        encoding="utf-8",
    )
    records = load_monitor_records(monitor_path)
    assert records[0].reward == 2.5
    assert moving_average([], window=4) == []
    empty_summary = summarize_monitor_records([])
    assert empty_summary.episodes_logged == 0
    with pytest.raises(ValueError):
        plot_training_curves([], tmp_path / "empty.png", title="Empty")


def test_train_baseline_writes_training_artifacts(monkeypatch, tmp_path) -> None:
    @dataclass
    class FakeModel:
        policy: str
        env: object
        seed: int
        verbose: int

        def learn(self, total_timesteps: int) -> None:
            self.total_timesteps = total_timesteps

        def save(self, path: str) -> None:
            Path(f"{path}.zip").write_text("fake", encoding="utf-8")

    class FakeMonitor:
        def __init__(self, env: object, *, filename: str, info_keywords: tuple[str, ...]) -> None:
            del info_keywords
            self.env = env
            Path(filename).write_text(
                '#{"t_start": 0.0, "env_id": "demo"}\n'
                "r,l,t,success,task_phase\n"
                "2.0,5,0.1,True,CLIMB\n"
                "1.0,4,0.2,False,ROLL\n",
                encoding="utf-8",
            )

        def close(self) -> None:
            self.env.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self.env, name)

    monkeypatch.setattr("flightlab.rl.baselines._load_sb3", lambda: (FakeModel, FakeModel))
    monkeypatch.setattr("flightlab.rl.baselines._load_monitor_class", lambda: FakeMonitor)
    monkeypatch.setattr(
        "flightlab.rl.baselines._evaluate_model",
        lambda **_: {
            "average_return": 3.0,
            "success_rate": 0.5,
            "average_completion_time_s": 8.0,
        },
    )
    log_dir = tmp_path / "logs"
    result = train_baseline(
        algorithm="ppo",
        task="takeoff",
        total_timesteps=12,
        seed=3,
        output_path=tmp_path / "model",
        log_dir=log_dir,
        plot_training=True,
        evaluation_episodes=2,
    )
    assert isinstance(result, TrainingResult)
    assert result.log_dir == str(log_dir)
    assert result.monitor_path == str(log_dir / "monitor.csv")
    assert result.plot_path == str(log_dir / "training.png")
    assert result.summary_path == str(log_dir / "summary.json")
    summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["episodes_logged"] == 2
    assert summary["evaluation"]["success_rate"] == 0.5
    assert Path(result.plot_path).exists()


def test_sb3_import_helpers_raise_focused_errors(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "stable_baselines3":
            raise ImportError("no sb3")
        if name == "stable_baselines3.common.monitor":
            raise ImportError("no monitor")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="Stable-Baselines3 is not installed"):
        baselines._load_sb3()
    with pytest.raises(RuntimeError, match="monitor support is not available"):
        baselines._load_monitor_class()


def test_load_model_class_supports_sac(monkeypatch) -> None:
    @dataclass
    class FakePPO:
        policy: str
        env: object
        seed: int
        verbose: int

    @dataclass
    class FakeSAC:
        policy: str
        env: object
        seed: int
        verbose: int

    monkeypatch.setattr("flightlab.rl.baselines._load_sb3", lambda: (FakePPO, FakeSAC))
    assert load_model_class("sac") is FakeSAC


def test_default_log_dir_and_evaluate_model(monkeypatch) -> None:
    assert baselines._default_log_dir(
        algorithm="ppo",
        task="takeoff",
        seed=7,
        output_path=None,
    ) == Path("artifacts/ppo_takeoff_seed7_training")

    class FakeModel:
        def predict(self, observation, deterministic: bool):  # type: ignore[no-untyped-def]
            del observation, deterministic
            return np.zeros(4, dtype=np.float32), None

    class FakeEnv:
        def __init__(self, summary: dict[str, float | bool], *, include_info: bool) -> None:
            self.summary = summary
            self.include_info = include_info
            self.closed = False

        def reset(self, *, seed: int | None = None):  # type: ignore[no-untyped-def]
            del seed
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):  # type: ignore[no-untyped-def]
            del action
            info = {"episode_summary": self.summary} if self.include_info else {}
            return np.zeros(1, dtype=np.float32), 0.0, True, False, info

        def episode_summary(self) -> dict[str, float | bool]:
            return self.summary

        def close(self) -> None:
            self.closed = True

    summaries = [
        {"episode_return": 10.0, "success": True, "completion_time_s": 5.0},
        {"episode_return": 6.0, "success": False, "completion_time_s": 7.0},
    ]
    environments: list[FakeEnv] = []

    def fake_make_env(task: str, *, seed: int | None = None) -> FakeEnv:
        del task, seed
        env = FakeEnv(summaries[len(environments)], include_info=len(environments) == 0)
        environments.append(env)
        return env

    monkeypatch.setattr("flightlab.rl.baselines.make_env", fake_make_env)
    evaluation = baselines._evaluate_model(model=FakeModel(), task="takeoff", seed=1, episodes=2)
    assert evaluation == {
        "average_return": 8.0,
        "success_rate": 0.5,
        "average_completion_time_s": 6.0,
    }
    assert all(env.closed for env in environments)


def test_train_baseline_auto_log_dir_without_records(monkeypatch) -> None:
    @dataclass
    class FakeModel:
        policy: str
        env: object
        seed: int
        verbose: int

        def learn(self, total_timesteps: int) -> None:
            self.total_timesteps = total_timesteps

        def save(self, path: str) -> None:
            raise AssertionError("save should not be called without output_path")

    class EmptyMonitor:
        def __init__(self, env: object, *, filename: str, info_keywords: tuple[str, ...]) -> None:
            del info_keywords
            self.env = env
            Path(filename).write_text(
                '#{"t_start": 0.0, "env_id": "demo"}\n'
                "r,l,t,success,task_phase\n",
                encoding="utf-8",
            )

        def close(self) -> None:
            self.env.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self.env, name)

    monkeypatch.setattr("flightlab.rl.baselines._load_sb3", lambda: (FakeModel, FakeModel))
    monkeypatch.setattr("flightlab.rl.baselines._load_monitor_class", lambda: EmptyMonitor)
    result = train_baseline(
        algorithm="ppo",
        task="takeoff",
        total_timesteps=5,
        seed=7,
        plot_training=True,
    )
    assert result.output_path is None
    assert result.log_dir == "artifacts/ppo_takeoff_seed7_training"
    assert result.monitor_path == "artifacts/ppo_takeoff_seed7_training/monitor.csv"
    assert result.plot_path is None
    summary = json.loads(
        Path("artifacts/ppo_takeoff_seed7_training/summary.json").read_text(encoding="utf-8")
    )
    assert summary["episodes_logged"] == 0
    Path("artifacts/ppo_takeoff_seed7_training/monitor.csv").unlink()
    Path("artifacts/ppo_takeoff_seed7_training/summary.json").unlink()
    Path("artifacts/ppo_takeoff_seed7_training").rmdir()


def test_load_model_class_validates_algorithm(monkeypatch) -> None:
    @dataclass
    class FakeModel:
        policy: str
        env: object
        seed: int
        verbose: int

    monkeypatch.setattr(
        "flightlab.rl.baselines._load_sb3",
        lambda: (FakeModel, FakeModel),
    )
    assert load_model_class("ppo") is FakeModel
    with pytest.raises(ValueError):
        load_model_class("bad")
