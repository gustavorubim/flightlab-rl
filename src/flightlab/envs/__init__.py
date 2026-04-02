"""Environment factory and task registrations."""

from __future__ import annotations

from gymnasium.envs.registration import register

from flightlab.envs.flight_plan import FlightPlanEnv
from flightlab.envs.landing import LandingEnv
from flightlab.envs.takeoff import TakeoffEnv

_REGISTERED = False


def register_envs() -> None:
    """Register built-in environments with Gymnasium once."""
    global _REGISTERED
    if _REGISTERED:
        return
    register(id="FlightLabFlightPlan-v0", entry_point="flightlab.envs.flight_plan:FlightPlanEnv")
    register(id="FlightLabTakeoff-v0", entry_point="flightlab.envs.takeoff:TakeoffEnv")
    register(id="FlightLabLanding-v0", entry_point="flightlab.envs.landing:LandingEnv")
    _REGISTERED = True


def make_env(task: str, *, seed: int | None = None) -> FlightPlanEnv | TakeoffEnv | LandingEnv:
    """Create a built-in environment by task name."""
    register_envs()
    normalized_task = task.lower()
    if normalized_task in {"flight_plan", "flight-plan", "route"}:
        return FlightPlanEnv(seed=seed)
    if normalized_task == "takeoff":
        return TakeoffEnv(seed=seed)
    if normalized_task == "landing":
        return LandingEnv(seed=seed)
    raise ValueError(f"Unsupported task '{task}'.")


__all__ = ["FlightPlanEnv", "LandingEnv", "TakeoffEnv", "make_env", "register_envs"]
