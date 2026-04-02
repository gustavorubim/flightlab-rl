"""Load a trained Stable-Baselines3 policy, run an episode, and render artifacts."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--task", choices=("flight_plan", "takeoff", "landing"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--replay-output")
    parser.add_argument("--video-output")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one rollout from a trained policy."""
    bootstrap_src_path()
    from flightlab.envs import make_env
    from flightlab.rl import load_model_class

    args = parse_args()
    env = make_env(args.task, seed=args.seed)
    model_class = load_model_class(args.algorithm)
    model = model_class.load(args.model)
    observation, _info = env.reset(seed=args.seed)
    last_info: dict[str, object] = {}
    for _ in range(args.steps):
        action, _state = model.predict(observation, deterministic=not args.stochastic)
        observation, _reward, terminated, truncated, last_info = env.step(action)
        if terminated or truncated:
            break
    summary = last_info.get("episode_summary")
    if not isinstance(summary, dict):
        summary = env.episode_summary()
    print(summary)
    if args.replay_output:
        print(env.export_replay(args.replay_output))
    if args.video_output:
        print(env.export_video(args.video_output, fps=args.fps))


if __name__ == "__main__":
    main()
