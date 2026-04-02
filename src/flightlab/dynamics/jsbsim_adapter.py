"""Optional JSBSim integration."""

from __future__ import annotations

from dataclasses import dataclass

from flightlab.core.types import AircraftState, ControlCommand
from flightlab.core.units import feet_to_meters, meters_to_feet
from flightlab.dynamics.base import DynamicsConfig

try:  # pragma: no cover - exercised only when JSBSim is installed.
    import jsbsim  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - covered by unit tests through the guard path.
    jsbsim = None


@dataclass
class JSBSimConfig(DynamicsConfig):
    """Configuration for the JSBSim-backed dynamics adapter."""

    model_name: str = "c172p"
    root_dir: str | None = None


class JSBSimDynamics:  # pragma: no cover - optional integration surface.
    """Thin wrapper around `jsbsim.FGFDMExec`."""

    def __init__(self, config: JSBSimConfig | None = None) -> None:
        if jsbsim is None:
            raise RuntimeError(
                "JSBSim is not installed. Install the `sim` extra to use this backend."
            )
        self.config = config or JSBSimConfig()
        self._fdm = jsbsim.FGFDMExec(self.config.root_dir)
        self._fdm.set_dt(self.config.dt_s)
        if not self._fdm.load_model(self.config.model_name):
            raise RuntimeError(f"Unable to load JSBSim model '{self.config.model_name}'.")
        self._state = AircraftState(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    @property
    def state(self) -> AircraftState:
        return self._state

    def reset(self, initial_state: AircraftState) -> AircraftState:
        self._fdm["ic/h-sl-ft"] = meters_to_feet(initial_state.altitude_m)
        self._fdm["ic/u-fps"] = meters_to_feet(initial_state.u_mps)
        self._fdm["ic/v-fps"] = meters_to_feet(initial_state.v_mps)
        self._fdm["ic/w-fps"] = meters_to_feet(initial_state.w_mps)
        self._fdm["ic/phi-rad"] = initial_state.roll_rad
        self._fdm["ic/theta-rad"] = initial_state.pitch_rad
        self._fdm["ic/psi-rad"] = initial_state.heading_rad
        self._fdm.run_ic()
        self._state = initial_state
        return self._state

    def step(self, command: ControlCommand) -> AircraftState:
        clipped = command.clipped()
        self._fdm["fcs/elevator-cmd-norm"] = clipped.elevator
        self._fdm["fcs/aileron-cmd-norm"] = clipped.aileron
        self._fdm["fcs/rudder-cmd-norm"] = clipped.rudder
        self._fdm["fcs/throttle-cmd-norm"] = clipped.throttle
        self._fdm.run()
        self._state = AircraftState(
            position_x_m=self._state.position_x_m,
            position_y_m=self._state.position_y_m,
            altitude_m=feet_to_meters(self._fdm["position/h-sl-ft"]),
            roll_rad=self._fdm["attitude/phi-rad"],
            pitch_rad=self._fdm["attitude/theta-rad"],
            heading_rad=self._fdm["attitude/psi-rad"],
            u_mps=feet_to_meters(self._fdm["velocities/u-fps"]),
            v_mps=feet_to_meters(self._fdm["velocities/v-fps"]),
            w_mps=feet_to_meters(self._fdm["velocities/w-fps"]),
            p_radps=self._fdm["velocities/p-rad_sec"],
            q_radps=self._fdm["velocities/q-rad_sec"],
            r_radps=self._fdm["velocities/r-rad_sec"],
            airspeed_mps=feet_to_meters(self._fdm["velocities/vtrue-fps"]),
            groundspeed_mps=feet_to_meters(self._fdm["velocities/vg-fps"]),
            vertical_speed_mps=feet_to_meters(self._fdm["velocities/h-dot-fps"]),
            angle_of_attack_rad=self._fdm["aero/alpha-rad"],
            sideslip_rad=self._fdm["aero/beta-rad"],
            throttle=clipped.throttle,
            elevator=clipped.elevator,
            aileron=clipped.aileron,
            rudder=clipped.rudder,
            on_ground=bool(self._fdm["gear/wow"]),
            time_s=self._state.time_s + self.config.dt_s,
        )
        return self._state
