"""Evaluate a deterministic PID baseline on a built-in task."""

from __future__ import annotations

import argparse
import math

from _bootstrap import bootstrap_src_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("flight_plan", "takeoff", "landing"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    """Run evaluation and print the final episode summary."""
    bootstrap_src_path()
    from flightlab.controllers import PIDAutopilot
    from flightlab.core.geometry import signed_smallest_angle
    from flightlab.envs import make_env

    args = parse_args()
    env = make_env(args.task, seed=args.seed)
    controller = PIDAutopilot()
    observation, info = env.reset(seed=args.seed)
    del observation, info
    last_info = {}
    for _ in range(args.steps):
        state = env.state
        if args.task == "flight_plan":
            target = (
                env._latest_progress.current_waypoint
                if env._latest_progress
                else env.config.mission.waypoints[0]
            )
            heading_error = signed_smallest_angle(
                state.heading_rad,
                math.atan2(target.y_m - state.position_y_m, target.x_m - state.position_x_m),
            )
            altitude_error = target.altitude_m - state.altitude_m
            speed_error = target.target_airspeed_mps - state.airspeed_mps
        elif args.task == "takeoff":
            heading_error = env.config.runway.heading_error_rad(state.heading_rad)
            altitude_error = env.config.task.success_altitude_agl_m - (
                state.altitude_m - env.config.runway.elevation_m
            )
            speed_error = env.config.task.rotation_speed_mps - state.airspeed_mps
        else:
            along_m, _ = env.config.runway.local_coordinates(state.position_x_m, state.position_y_m)
            heading_error = env.config.runway.heading_error_rad(state.heading_rad)
            altitude_error = env._glideslope.altitude_error_m(along_m, state.altitude_m)
            speed_error = 20.0 - state.airspeed_mps
        action = controller.command(
            state,
            heading_error_rad=heading_error,
            altitude_error_m=altitude_error,
            speed_error_mps=speed_error,
            dt_s=env._dynamics.config.dt_s,
        ).as_array()
        _observation, _reward, terminated, truncated, last_info = env.step(action)
        if terminated or truncated:
            break
    summary = last_info.get("episode_summary")
    if not isinstance(summary, dict):
        summary = env.episode_summary()
    print(summary)


if __name__ == "__main__":
    main()
