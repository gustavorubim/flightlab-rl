from __future__ import annotations

from dataclasses import dataclass

import pytest

from flightlab.controllers.pid import PIDAutopilot, PIDController
from flightlab.metrics import summarize_episodes
from flightlab.rl.baselines import TrainingResult, train_baseline


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

    monkeypatch.setattr(
        "flightlab.rl.baselines._load_sb3",
        lambda: (FakeModel, FakeModel),
    )
    result = train_baseline(algorithm="ppo", task="takeoff", total_timesteps=12, seed=3)
    assert isinstance(result, TrainingResult)
    assert result.model_class_name == "FakeModel"
    assert result.total_timesteps == 12
