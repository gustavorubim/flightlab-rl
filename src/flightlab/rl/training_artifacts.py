"""Training-monitor parsing and plotting helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MonitorEpisodeRecord:
    """Single episode row parsed from a Stable-Baselines3 monitor CSV."""

    reward: float
    length: float
    elapsed_time_s: float
    success: bool = False


@dataclass(frozen=True)
class TrainingCurveSummary:
    """Compact summary computed from monitor records."""

    episodes_logged: int
    final_episode_return: float
    best_episode_return: float
    mean_episode_return: float
    final_window_return: float
    final_window_success_rate: float


def load_monitor_records(path: str | Path) -> list[MonitorEpisodeRecord]:
    """Load episode rows from a Stable-Baselines3 monitor file."""
    resolved_path = Path(path)
    records: list[MonitorEpisodeRecord] = []
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        if not first_line.startswith("#"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                MonitorEpisodeRecord(
                    reward=float(row["r"]),
                    length=float(row["l"]),
                    elapsed_time_s=float(row["t"]),
                    success=row.get("success", "False") == "True",
                )
            )
    return records


def moving_average(values: Sequence[float], *, window: int) -> list[float]:
    """Return a causal moving average with the same length as the input."""
    if not values:
        return []
    resolved_window = max(1, min(int(window), len(values)))
    array = np.asarray(values, dtype=float)
    cumulative = np.cumsum(array)
    averages: list[float] = []
    for index in range(len(array)):
        start = max(0, index - resolved_window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start > 0 else 0.0)
        averages.append(float(total / float(index - start + 1)))
    return averages


def summarize_monitor_records(
    records: Sequence[MonitorEpisodeRecord], *, reward_window: int = 25
) -> TrainingCurveSummary:
    """Build a compact summary for a monitor-record sequence."""
    if not records:
        return TrainingCurveSummary(
            episodes_logged=0,
            final_episode_return=0.0,
            best_episode_return=0.0,
            mean_episode_return=0.0,
            final_window_return=0.0,
            final_window_success_rate=0.0,
        )
    rewards = [record.reward for record in records]
    successes = [float(record.success) for record in records]
    reward_average = moving_average(rewards, window=reward_window)
    success_average = moving_average(successes, window=reward_window)
    return TrainingCurveSummary(
        episodes_logged=len(records),
        final_episode_return=float(rewards[-1]),
        best_episode_return=float(max(rewards)),
        mean_episode_return=float(np.mean(rewards)),
        final_window_return=float(reward_average[-1]),
        final_window_success_rate=float(success_average[-1]),
    )


def summary_to_dict(summary: TrainingCurveSummary) -> dict[str, float | int]:
    """Convert a summary dataclass to a JSON-friendly dictionary."""
    return dict(asdict(summary))


def write_training_summary(
    path: str | Path,
    *,
    task: str,
    algorithm: str,
    seed: int,
    timesteps: int,
    monitor_summary: TrainingCurveSummary,
    evaluation: dict[str, float] | None = None,
    model_path: str | None = None,
    monitor_path: str | None = None,
    plot_path: str | None = None,
) -> str:
    """Write a JSON summary for a training run."""
    payload: dict[str, object] = {
        "task": task,
        "algorithm": algorithm,
        "seed": seed,
        "timesteps": timesteps,
        **summary_to_dict(monitor_summary),
        "evaluation": evaluation or {},
        "model_path": model_path,
        "monitor_path": monitor_path,
        "plot_path": plot_path,
    }
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(resolved_path)


def plot_training_curves(
    records: Sequence[MonitorEpisodeRecord],
    path: str | Path,
    *,
    title: str,
    reward_window: int = 25,
) -> str:
    """Render reward and episode-length curves for a training run."""
    if not records:
        raise ValueError("Cannot plot training curves without monitor records.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards = [record.reward for record in records]
    lengths = [record.length for record in records]
    successes = [float(record.success) for record in records]
    reward_average = moving_average(rewards, window=reward_window)
    success_average = moving_average(successes, window=reward_window)
    cumulative_steps = np.cumsum(np.asarray(lengths, dtype=float))
    episode_index = np.arange(1, len(records) + 1)

    figure, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    figure.suptitle(title, fontsize=14)

    axes[0].plot(
        episode_index,
        rewards,
        color="#457b9d",
        alpha=0.35,
        linewidth=1.0,
        label="episode return",
    )
    axes[0].plot(
        episode_index,
        reward_average,
        color="#d62828",
        linewidth=2.0,
        label=f"{reward_window}-episode moving average",
    )
    axes[0].set_ylabel("Return")
    axes[0].set_title("Reward per episode")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        cumulative_steps,
        rewards,
        color="#2a9d8f",
        alpha=0.3,
        linewidth=1.0,
        label="episode return",
    )
    axes[1].plot(
        cumulative_steps,
        reward_average,
        color="#264653",
        linewidth=2.0,
        label=f"{reward_window}-episode moving average",
    )
    axes[1].set_ylabel("Return")
    axes[1].set_title("Reward over environment timesteps")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(
        episode_index,
        lengths,
        color="#6a4c93",
        alpha=0.35,
        linewidth=1.0,
        label="episode length",
    )
    axes[2].plot(
        episode_index,
        success_average,
        color="#f77f00",
        linewidth=2.0,
        label=f"{reward_window}-episode success rate",
    )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Length / Success")
    axes[2].set_title("Episode length and short-horizon success rate")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    figure.tight_layout()
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(resolved_path)
