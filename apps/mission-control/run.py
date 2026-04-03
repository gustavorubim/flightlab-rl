#!/usr/bin/env python3
"""Single-command entrypoint for the mission-control app."""

from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _main() -> int:
    from app.dev_launcher import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
