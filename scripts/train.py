"""Train a Stable-Baselines3 baseline."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--task", choices=("flight_plan", "takeoff", "landing"), required=True)
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    bootstrap_src_path()
    from flightlab.rl import train_baseline

    args = parse_args()
    result = train_baseline(
        algorithm=args.algorithm,
        task=args.task,
        total_timesteps=args.timesteps,
        seed=args.seed,
        verbose=1,
    )
    print(result)


if __name__ == "__main__":
    main()
