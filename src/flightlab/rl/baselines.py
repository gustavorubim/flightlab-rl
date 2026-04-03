"""Stable-Baselines3 training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flightlab.envs import make_env
from flightlab.rl.training_artifacts import (
    load_monitor_records,
    plot_training_curves,
    summarize_monitor_records,
    write_training_summary,
)

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
    log_dir: str | None = None
    monitor_path: str | None = None
    summary_path: str | None = None
    plot_path: str | None = None


def _load_sb3() -> tuple[Any, Any]:
    """Load Stable-Baselines3 algorithms or raise a focused error."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:  # pragma: no cover - tested via monkeypatch.
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install the `rl` extra to train models."
        ) from exc
    return PPO, SAC


def _load_monitor_class() -> Any:
    """Load Stable-Baselines3's monitor wrapper."""
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover - covered through _load_sb3.
        raise RuntimeError(
            "Stable-Baselines3 monitor support is not available. Install the `rl` extra."
        ) from exc
    return Monitor


def load_model_class(algorithm: str) -> Any:
    """Return the Stable-Baselines3 model class for an algorithm name."""
    normalized_algorithm = algorithm.lower()
    if normalized_algorithm not in algorithm_choices:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Expected one of {algorithm_choices}."
        )
    ppo_cls, sac_cls = _load_sb3()
    return ppo_cls if normalized_algorithm == "ppo" else sac_cls


def _default_log_dir(
    *,
    algorithm: str,
    task: str,
    seed: int,
    output_path: str | Path | None,
) -> Path:
    """Build a default training-artifact directory."""
    if output_path is not None:
        target = Path(output_path)
        return target.parent / f"{target.name}_training"
    return Path("artifacts") / f"{algorithm}_{task}_seed{seed}_training"


def _evaluate_model(
    *,
    model: Any,
    task: str,
    seed: int,
    episodes: int,
) -> dict[str, float]:
    """Run deterministic post-training evaluation for a model."""
    returns: list[float] = []
    successes: list[float] = []
    completion_times: list[float] = []
    for index in range(episodes):
        episode_seed = seed + 100 + index
        env = make_env(task, seed=episode_seed)
        observation, _info = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _state = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, last_info = env.step(action)
        summary = last_info.get("episode_summary")
        if not isinstance(summary, dict):
            summary = env.episode_summary()
        returns.append(float(summary.get("episode_return", 0.0)))
        successes.append(float(bool(summary.get("success", False))))
        completion_times.append(float(summary.get("completion_time_s", 0.0)))
        env.close()
    return {
        "average_return": sum(returns) / max(len(returns), 1),
        "success_rate": sum(successes) / max(len(successes), 1),
        "average_completion_time_s": sum(completion_times) / max(len(completion_times), 1),
    }


def train_baseline(
    *,
    algorithm: str,
    task: str,
    total_timesteps: int,
    seed: int,
    verbose: int = 0,
    output_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    plot_training: bool = False,
    evaluation_episodes: int = 0,
) -> TrainingResult:
    """Train a PPO or SAC baseline on one of the built-in environments."""
    normalized_algorithm = algorithm.lower()
    model_cls = load_model_class(normalized_algorithm)
    resolved_log_dir: Path | None = None
    monitor_path: str | None = None
    summary_path: str | None = None
    plot_path: str | None = None
    env = make_env(task, seed=seed)
    if log_dir is not None or plot_training or evaluation_episodes > 0:
        resolved_log_dir = (
            Path(log_dir)
            if log_dir is not None
            else _default_log_dir(
                algorithm=normalized_algorithm,
                task=task,
                seed=seed,
                output_path=output_path,
            )
        )
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        monitor_path = str(resolved_log_dir / "monitor.csv")
        monitor_cls = _load_monitor_class()
        env = monitor_cls(
            env,
            filename=monitor_path,
            info_keywords=("success", "task_phase"),
        )
    model = model_cls("MlpPolicy", env, seed=seed, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    resolved_output_path: str | None = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target))
        resolved_output_path = str(target)
    if resolved_log_dir is not None and monitor_path is not None:
        records = load_monitor_records(monitor_path)
        evaluation = (
            _evaluate_model(model=model, task=task, seed=seed, episodes=evaluation_episodes)
            if evaluation_episodes > 0
            else {}
        )
        if plot_training and records:
            plot_path = plot_training_curves(
                records,
                resolved_log_dir / "training.png",
                title=(
                    f"{task.replace('_', ' ').title()} "
                    f"{normalized_algorithm.upper()} Training (seed {seed})"
                ),
            )
        summary_path = write_training_summary(
            resolved_log_dir / "summary.json",
            task=task,
            algorithm=normalized_algorithm,
            seed=seed,
            timesteps=total_timesteps,
            monitor_summary=summarize_monitor_records(records),
            evaluation=evaluation,
            model_path=resolved_output_path,
            monitor_path=monitor_path,
            plot_path=plot_path,
        )
    env.close()
    return TrainingResult(
        algorithm=normalized_algorithm,
        task=task,
        total_timesteps=total_timesteps,
        seed=seed,
        model_class_name=type(model).__name__,
        output_path=resolved_output_path,
        log_dir=str(resolved_log_dir) if resolved_log_dir is not None else None,
        monitor_path=monitor_path,
        summary_path=summary_path,
        plot_path=plot_path,
    )
