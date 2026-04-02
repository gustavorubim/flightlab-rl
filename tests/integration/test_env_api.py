from __future__ import annotations

import numpy as np

from flightlab.envs import make_env, register_envs
from flightlab.envs.flight_plan import FlightPlanEnv, FlightPlanEnvConfig
from flightlab.envs.landing import LandingEnv, LandingEnvConfig
from flightlab.envs.takeoff import TakeoffEnv, TakeoffEnvConfig


def test_register_envs_is_idempotent() -> None:
    register_envs()
    register_envs()


def test_builtin_envs_reset_and_step() -> None:
    for task in ("flight_plan", "takeoff", "landing"):
        env = make_env(task, seed=5)
        observation, info = env.reset(seed=5)
        assert observation.shape == env.observation_space.shape
        assert info["task_phase"] == "RESET"
        next_observation, reward, terminated, truncated, step_info = env.step(
            np.asarray([0.0, 0.0, 0.0, 0.7], dtype=np.float32)
        )
        assert next_observation.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_breakdown" in step_info
        assert "safety_flags" in step_info
        assert env.render()


def test_env_episode_summary_and_replay_export(tmp_path) -> None:
    takeoff = TakeoffEnv(
        seed=1,
        config=TakeoffEnvConfig(task=TakeoffEnvConfig().task.__class__(max_steps=2)),
    )
    takeoff.reset(seed=1)
    summary = {}
    for _ in range(2):
        _observation, _reward, _terminated, _truncated, info = takeoff.step(
            np.asarray([0.0, 0.0, 0.0, 0.7], dtype=np.float32)
        )
        summary = info.get("episode_summary", summary)
    assert "episode_return" in summary
    replay_path = takeoff.export_replay(str(tmp_path / "takeoff.json"))
    assert replay_path.endswith("takeoff.json")

    flight_plan = FlightPlanEnv(
        seed=2,
        config=FlightPlanEnvConfig(task=FlightPlanEnvConfig().task.__class__(max_steps=1)),
    )
    flight_plan.reset(seed=2)
    _ = flight_plan.step(np.asarray([0.0, 0.0, 0.0, 0.6], dtype=np.float32))

    landing = LandingEnv(
        seed=3,
        config=LandingEnvConfig(task=LandingEnvConfig().task.__class__(max_steps=1)),
    )
    landing.reset(seed=3)
    _ = landing.step(np.asarray([0.0, 0.0, 0.0, 0.5], dtype=np.float32))


def test_landing_env_reports_touchdown_metrics() -> None:
    landing = LandingEnv(seed=4)
    landing.reset(seed=4)
    landing._previous_on_ground = False
    landing._previous_vertical_speed_mps = -3.0
    landing._previous_on_ground = False
    touchdown_state = landing.state.__class__(**{**landing.state.__dict__, "on_ground": True})
    evaluation = landing._evaluate(touchdown_state)
    assert evaluation.metrics["touchdown_sink_rate_mps"] == 3.0
