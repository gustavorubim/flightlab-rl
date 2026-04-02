from __future__ import annotations

import numpy as np

from flightlab.envs import make_env


def rollout(task: str, seed: int) -> tuple[list[np.ndarray], list[float]]:
    env = make_env(task, seed=seed)
    observations = []
    rewards = []
    action = np.asarray([0.05, -0.02, 0.01, 0.7], dtype=np.float32)
    observation, _info = env.reset(seed=seed)
    observations.append(observation)
    for _ in range(8):
        observation, reward, terminated, truncated, _info = env.step(action)
        observations.append(observation)
        rewards.append(reward)
        if terminated or truncated:
            break
    return observations, rewards


def test_same_seed_produces_identical_rollouts() -> None:
    first_obs, first_rewards = rollout("flight_plan", 11)
    second_obs, second_rewards = rollout("flight_plan", 11)
    assert len(first_obs) == len(second_obs)
    assert len(first_rewards) == len(second_rewards)
    for left, right in zip(first_obs, second_obs, strict=True):
        assert np.allclose(left, right)
    assert np.allclose(first_rewards, second_rewards)


def test_different_seeds_change_initial_conditions() -> None:
    first_obs, _ = rollout("takeoff", 1)
    second_obs, _ = rollout("takeoff", 2)
    assert not np.allclose(first_obs[0], second_obs[0])
