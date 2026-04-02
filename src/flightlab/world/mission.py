"""Mission and waypoint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Waypoint:
    """A waypoint used by route guidance."""

    name: str
    x_m: float
    y_m: float
    altitude_m: float
    target_airspeed_mps: float
    acceptance_radius_m: float = 35.0


@dataclass(frozen=True)
class Mission:
    """An ordered set of waypoints."""

    name: str
    waypoints: tuple[Waypoint, ...]

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("Mission must contain at least one waypoint.")


def mission_from_dict(payload: dict[str, Any]) -> Mission:
    """Build a mission from a YAML-compatible dictionary."""
    waypoints = tuple(
        Waypoint(
            name=str(item["name"]),
            x_m=float(item["x_m"]),
            y_m=float(item["y_m"]),
            altitude_m=float(item["altitude_m"]),
            target_airspeed_mps=float(item["target_airspeed_mps"]),
            acceptance_radius_m=float(item.get("acceptance_radius_m", 35.0)),
        )
        for item in payload["waypoints"]
    )
    return Mission(name=str(payload["name"]), waypoints=waypoints)


def mission_from_path(path: str | Path) -> Mission:
    """Load a mission definition from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return mission_from_dict(yaml.safe_load(handle))
