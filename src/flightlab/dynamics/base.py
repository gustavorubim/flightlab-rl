"""Dynamics backend interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flightlab.core.types import AircraftState, ControlCommand


@dataclass
class DynamicsConfig:
    """Configuration for the lightweight headless dynamics backend."""

    dt_s: float = 0.1
    runway_elevation_m: float = 0.0
    wind_east_mps: float = 0.0
    wind_north_mps: float = 0.0
    actuator_tau_s: float = 0.25
    lift_off_speed_mps: float = 24.0
    stall_speed_mps: float = 18.0
    nominal_mass_kg: float = 1200.0
    mass_scale: float = 1.0
    cg_offset_m: float = 0.0


class DynamicsModel(Protocol):
    """Protocol implemented by every flight dynamics backend."""

    @property
    def state(self) -> AircraftState:
        """Return the latest aircraft state."""

    def reset(self, initial_state: AircraftState) -> AircraftState:
        """Reset the backend to a specific state."""

    def step(self, command: ControlCommand) -> AircraftState:
        """Advance the backend by one time step."""
