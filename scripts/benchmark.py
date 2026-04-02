"""Run a benchmark over multiple episodes."""

from __future__ import annotations

import argparse

import numpy as np
from _bootstrap import bootstrap_src_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("flight_plan", "takeoff", "landing"), required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    """Run a simple zero-policy benchmark and print the aggregate summary."""
    bootstrap_src_path()
    from flightlab.envs import make_env
    from flightlab.metrics import summarize_episodes

    args = parse_args()
    episodes: list[dict[str, object]] = []
    for episode_index in range(args.episodes):
        env = make_env(args.task, seed=args.seed + episode_index)
        env.reset(seed=args.seed + episode_index)
        action = np.asarray([0.0, 0.0, 0.0, 0.6], dtype=np.float32)
        final_info: dict[str, object] = {}
        for _ in range(args.steps):
            _observation, _reward, terminated, truncated, final_info = env.step(action)
            if terminated or truncated:
                break
        summary = final_info.get("episode_summary")
        if not isinstance(summary, dict):
            summary = env.episode_summary()
        episodes.append(summary)
    print(summarize_episodes(episodes))


if __name__ == "__main__":
    main()
