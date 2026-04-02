"""Episode and benchmark metric aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate metrics across multiple episodes."""

    success_rate: float
    crash_rate: float
    stall_rate: float
    runway_excursion_rate: float
    average_cross_track_error_m: float
    altitude_rmse_m: float
    average_action_smoothness: float
    average_return: float
    average_completion_time_s: float


def _mean(values: Iterable[float]) -> float:
    sequence = list(values)
    return fmean(sequence) if sequence else 0.0


def summarize_episodes(episodes: list[dict[str, object]]) -> BenchmarkSummary:
    """Summarize benchmark metrics over a list of episode dictionaries."""
    if not episodes:
        return BenchmarkSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return BenchmarkSummary(
        success_rate=_mean(float(bool(episode["success"])) for episode in episodes),
        crash_rate=_mean(float(bool(episode["crash"])) for episode in episodes),
        stall_rate=_mean(float(bool(episode["stall"])) for episode in episodes),
        runway_excursion_rate=_mean(
            float(bool(episode["runway_excursion"])) for episode in episodes
        ),
        average_cross_track_error_m=_mean(
            float(episode["average_cross_track_error_m"]) for episode in episodes
        ),
        altitude_rmse_m=_mean(float(episode["altitude_rmse_m"]) for episode in episodes),
        average_action_smoothness=_mean(
            float(episode["action_smoothness"]) for episode in episodes
        ),
        average_return=_mean(float(episode["episode_return"]) for episode in episodes),
        average_completion_time_s=_mean(
            float(episode["completion_time_s"]) for episode in episodes
        ),
    )
