"""Stable-Baselines3 training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flightlab.envs import make_env

algorithm_choices = ("ppo", "sac")


@dataclass(frozen=True)
class TrainingResult:
    """Output from a training run."""

    algorithm: str
    task: str
    total_timesteps: int
    seed: int
    model_class_name: str
    output_path: str | None = None


def _load_sb3() -> tuple[Any, Any]:
    """Load Stable-Baselines3 algorithms or raise a focused error."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:  # pragma: no cover - tested via monkeypatch.
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install the `rl` extra to train models."
        ) from exc
    return PPO, SAC


def load_model_class(algorithm: str) -> Any:
    """Return the Stable-Baselines3 model class for an algorithm name."""
    normalized_algorithm = algorithm.lower()
    if normalized_algorithm not in algorithm_choices:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Expected one of {algorithm_choices}."
        )
    ppo_cls, sac_cls = _load_sb3()
    return ppo_cls if normalized_algorithm == "ppo" else sac_cls


def train_baseline(
    *,
    algorithm: str,
    task: str,
    total_timesteps: int,
    seed: int,
    verbose: int = 0,
    output_path: str | Path | None = None,
) -> TrainingResult:
    """Train a PPO or SAC baseline on one of the built-in environments."""
    normalized_algorithm = algorithm.lower()
    model_cls = load_model_class(normalized_algorithm)
    env = make_env(task, seed=seed)
    model = model_cls("MlpPolicy", env, seed=seed, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    resolved_output_path: str | None = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target))
        resolved_output_path = str(target)
    return TrainingResult(
        algorithm=normalized_algorithm,
        task=task,
        total_timesteps=total_timesteps,
        seed=seed,
        model_class_name=type(model).__name__,
        output_path=resolved_output_path,
    )
