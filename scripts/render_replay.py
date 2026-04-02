"""Render a replay JSON file to an MP4 video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap_src_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task")
    parser.add_argument("--title")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    """Render a replay JSON file into an MP4 video."""
    bootstrap_src_path()
    from flightlab.render import render_episode_video

    args = parse_args()
    with Path(args.replay).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    print(
        render_episode_video(
            records,
            args.output,
            task_name=args.task,
            title=args.title,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    )


if __name__ == "__main__":
    main()
