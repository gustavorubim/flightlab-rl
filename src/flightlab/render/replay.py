"""Replay logging and export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightlab.core.types import AircraftState


class EpisodeRecorder:
    """Collect reset and step records for deterministic replay."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear all recorded state."""
        self._records = []

    def record_reset(self, state: AircraftState, info: dict[str, Any]) -> None:
        """Record the initial environment state."""
        self._records.append({"kind": "reset", "state": _state_to_dict(state), "info": info})

    def record_step(
        self,
        state: AircraftState,
        action: list[float],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """Record one environment transition."""
        self._records.append(
            {
                "kind": "step",
                "state": _state_to_dict(state),
                "action": action,
                "reward": reward,
                "info": info,
            }
        )

    def as_list(self) -> list[dict[str, Any]]:
        """Return a serializable replay payload."""
        return list(self._records)

    def export_json(self, path: str | Path) -> Path:
        """Export the replay to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.as_list(), handle, indent=2, sort_keys=True)
        return target


def _state_to_dict(state: AircraftState) -> dict[str, float | bool]:
    """Convert an aircraft state to a JSON-friendly mapping."""
    return {
        "position_x_m": state.position_x_m,
        "position_y_m": state.position_y_m,
        "altitude_m": state.altitude_m,
        "roll_rad": state.roll_rad,
        "pitch_rad": state.pitch_rad,
        "heading_rad": state.heading_rad,
        "u_mps": state.u_mps,
        "v_mps": state.v_mps,
        "w_mps": state.w_mps,
        "p_radps": state.p_radps,
        "q_radps": state.q_radps,
        "r_radps": state.r_radps,
        "airspeed_mps": state.airspeed_mps,
        "groundspeed_mps": state.groundspeed_mps,
        "vertical_speed_mps": state.vertical_speed_mps,
        "angle_of_attack_rad": state.angle_of_attack_rad,
        "sideslip_rad": state.sideslip_rad,
        "throttle": state.throttle,
        "elevator": state.elevator,
        "aileron": state.aileron,
        "rudder": state.rudder,
        "on_ground": state.on_ground,
        "time_s": state.time_s,
    }
