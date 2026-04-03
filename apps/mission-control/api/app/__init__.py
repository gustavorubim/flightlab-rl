"""Mission-control API package."""

from __future__ import annotations

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    """Lazily expose the FastAPI app objects without eager import side effects."""
    if name in __all__:
        from app.main import app, create_app

        exports = {"app": app, "create_app": create_app}
        return exports[name]
    raise AttributeError(f"module 'app' has no attribute {name!r}")
