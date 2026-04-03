"""Mission-control configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONTROLLER_CONFIG = REPO_ROOT / "apps/mission-control/config/controllers.yaml"


def _candidate_model_paths(path: Path) -> tuple[Path, ...]:
    """Return candidate stable-baselines checkpoint paths."""
    if path.suffix == ".zip":
        return (path, path.with_suffix(""))
    return (path, path.with_suffix(".zip"))


def _resolve_model_path(raw_value: str, *, base_dir: Path) -> Path:
    """Resolve a model path relative to the configuration file."""
    path = Path(raw_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass(frozen=True)
class PIDModeConfig:
    """Configuration for the PID live controller mode."""

    label: str
    description: str


@dataclass(frozen=True)
class RLCheckpointConfig:
    """Configuration for a task-specific RL checkpoint."""

    label: str
    description: str
    task: str
    algorithm: str
    model_path: Path

    @property
    def available(self) -> bool:
        """Return whether the configured checkpoint exists on disk."""
        return any(candidate.exists() for candidate in _candidate_model_paths(self.model_path))

    @property
    def load_path(self) -> Path:
        """Return the concrete path to load with Stable-Baselines3."""
        for candidate in _candidate_model_paths(self.model_path):
            if candidate.exists():
                return candidate
        return self.model_path


@dataclass(frozen=True)
class RLPhaseSwitchedConfig:
    """Configuration for the phase-switched RL controller mode."""

    label: str
    description: str
    takeoff: RLCheckpointConfig
    flight_plan: RLCheckpointConfig

    @property
    def available(self) -> bool:
        """Return whether both required checkpoints exist."""
        return self.takeoff.available and self.flight_plan.available


@dataclass(frozen=True)
class ControllerRegistry:
    """Mission-control controller registry."""

    pid: PIDModeConfig
    rl_phase_switched: RLPhaseSwitchedConfig

    @classmethod
    def from_path(cls, path: str | Path = DEFAULT_CONTROLLER_CONFIG) -> ControllerRegistry:
        """Load registry configuration from disk."""
        resolved_path = Path(path)
        payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        modes = payload.get("controller_modes", payload.get("controllers", {}))
        base_dir = resolved_path.parent
        rl_payload = modes["rl_phase_switched"]
        takeoff_payload = rl_payload.get("takeoff")
        if takeoff_payload is None:
            takeoff_payload = {
                "label": "Takeoff RL",
                "description": "Task-specific takeoff checkpoint.",
                "task": "takeoff",
                "algorithm": rl_payload["takeoff_algorithm"],
                "model_path": rl_payload["takeoff_model_path"],
            }
        flight_plan_payload = rl_payload.get("flight_plan")
        if flight_plan_payload is None:
            flight_plan_payload = {
                "label": "Flight Plan RL",
                "description": "Task-specific flight-plan checkpoint.",
                "task": "flight_plan",
                "algorithm": rl_payload["flight_plan_algorithm"],
                "model_path": rl_payload["flight_plan_model_path"],
            }
        return cls(
            pid=PIDModeConfig(
                label=str(modes["pid"].get("label", modes["pid"].get("display_name", "PID"))),
                description=str(modes["pid"]["description"]),
            ),
            rl_phase_switched=RLPhaseSwitchedConfig(
                label=str(rl_payload.get("label", rl_payload.get("display_name", "RL"))),
                description=str(rl_payload["description"]),
                takeoff=_checkpoint_from_dict(
                    takeoff_payload,
                    base_dir=base_dir,
                ),
                flight_plan=_checkpoint_from_dict(
                    flight_plan_payload,
                    base_dir=base_dir,
                ),
            ),
        )


def _checkpoint_from_dict(payload: dict[str, Any], *, base_dir: Path) -> RLCheckpointConfig:
    """Build an RL checkpoint configuration from YAML data."""
    return RLCheckpointConfig(
        label=str(payload["label"]),
        description=str(payload["description"]),
        task=str(payload["task"]),
        algorithm=str(payload["algorithm"]),
        model_path=_resolve_model_path(str(payload["model_path"]), base_dir=base_dir),
    )


def controller_registry_path() -> Path:
    """Return the configured controller-registry path."""
    raw_value = os.environ.get("MISSION_CONTROL_CONTROLLER_REGISTRY")
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return DEFAULT_CONTROLLER_CONFIG
