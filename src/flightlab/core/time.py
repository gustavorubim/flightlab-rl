"""Simulation timekeeping utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationClock:
    """A simple deterministic simulation clock."""

    dt_s: float
    time_s: float = 0.0
    step_count: int = 0

    def reset(self) -> None:
        """Reset time to zero."""
        self.time_s = 0.0
        self.step_count = 0

    def tick(self) -> float:
        """Advance the clock by one simulation step."""
        self.step_count += 1
        self.time_s += self.dt_s
        return self.time_s
