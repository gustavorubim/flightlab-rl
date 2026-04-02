"""Repo-root shim for the `src/flightlab` package during local execution."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "flightlab"
if SRC_PACKAGE.is_dir():
    __path__.append(str(SRC_PACKAGE))

__all__ = ["make_env"]


def __getattr__(name: str):
    if name == "make_env":
        from flightlab.envs import make_env

        return make_env
    raise AttributeError(name)
