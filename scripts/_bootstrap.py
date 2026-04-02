"""Bootstrap local imports for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    """Insert the repository src directory into `sys.path`."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
