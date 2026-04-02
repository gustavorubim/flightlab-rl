"""Observation vector generation."""

from __future__ import annotations

import numpy as np

from flightlab.core.types import AircraftState


class ObservationBuilder:
    """Create low-dimensional state observations for Gymnasium environments."""

    base_feature_names = (
        "u_mps",
        "v_mps",
        "w_mps",
        "p_radps",
        "q_radps",
        "r_radps",
        "roll_rad",
        "pitch_rad",
        "sin_heading",
        "cos_heading",
        "altitude_msl_m",
        "vertical_speed_mps",
        "airspeed_mps",
        "groundspeed_mps",
        "angle_of_attack_rad",
        "sideslip_rad",
        "throttle",
        "elevator",
        "aileron",
        "rudder",
    )
    task_feature_names = (
        "task_delta_x_m",
        "task_delta_y_m",
        "task_delta_altitude_m",
        "task_target_speed_error_mps",
        "altitude_agl_m",
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the complete observation feature list."""
        return self.base_feature_names + self.task_feature_names

    def build(
        self,
        state: AircraftState,
        *,
        task_delta_x_m: float = 0.0,
        task_delta_y_m: float = 0.0,
        task_delta_altitude_m: float = 0.0,
        task_target_speed_error_mps: float = 0.0,
        altitude_agl_m: float = 0.0,
    ) -> np.ndarray:
        """Build a float32 observation vector."""
        extras = np.asarray(
            [
                task_delta_x_m,
                task_delta_y_m,
                task_delta_altitude_m,
                task_target_speed_error_mps,
                altitude_agl_m,
            ],
            dtype=np.float32,
        )
        return np.concatenate((state.as_observation_vector(), extras), dtype=np.float32)
