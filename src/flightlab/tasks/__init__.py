"""Task evaluators and phase logic."""

from flightlab.tasks.flight_plan import FlightPlanPhase, FlightPlanTaskConfig, evaluate_flight_plan
from flightlab.tasks.landing import LandingPhase, LandingTaskConfig, evaluate_landing
from flightlab.tasks.takeoff import TakeoffPhase, TakeoffTaskConfig, evaluate_takeoff

__all__ = [
    "FlightPlanPhase",
    "FlightPlanTaskConfig",
    "LandingPhase",
    "LandingTaskConfig",
    "TakeoffPhase",
    "TakeoffTaskConfig",
    "evaluate_flight_plan",
    "evaluate_landing",
    "evaluate_takeoff",
]
