"""Run an environment episode and print compact state traces."""

from __future__ import annotations

import argparse

import numpy as np
from _bootstrap import bootstrap_src_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("flight_plan", "takeoff", "landing"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """Play an episode with a zero-action policy."""
    bootstrap_src_path()
    from flightlab.envs import make_env

    args = parse_args()
    env = make_env(args.task, seed=args.seed)
    env.reset(seed=args.seed)
    action = np.asarray([0.0, 0.0, 0.0, 0.6], dtype=np.float32)
    for _ in range(args.steps):
        _observation, _reward, terminated, truncated, _info = env.step(action)
        print(env.render())
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
