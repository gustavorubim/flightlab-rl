"""Single-command local launcher for the mission-control app."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSpec:
    """A local service command managed by the launcher."""

    name: str
    argv: tuple[str, ...]
    cwd: Path


def mission_control_root() -> Path:
    """Return the mission-control app root."""
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[4]


def repo_python_path(base_dir: Path | None = None) -> Path:
    """Return the repo virtualenv Python path."""
    root = base_dir or repo_root()
    return root / ".venv/bin/python"


def build_backend_service(base_dir: Path | None = None) -> ServiceSpec:
    """Build the backend command spec."""
    root = base_dir or repo_root()
    return ServiceSpec(
        name="api",
        argv=(
            str(repo_python_path(root)),
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ),
        cwd=root / "apps/mission-control/api",
    )


def build_frontend_service(base_dir: Path | None = None) -> ServiceSpec:
    """Build the frontend command spec."""
    root = base_dir or repo_root()
    return ServiceSpec(
        name="web",
        argv=("npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"),
        cwd=root / "apps/mission-control/web",
    )


def missing_prerequisites(
    *,
    base_dir: Path | None = None,
    check_python_imports: bool = True,
    require_npm: bool = True,
) -> list[str]:
    """Return any missing local prerequisites."""
    root = base_dir or repo_root()
    errors: list[str] = []
    python_path = repo_python_path(root)

    if not python_path.exists():
        errors.append(
            "Repo virtualenv is missing. Run: uv venv && source .venv/bin/activate "
            "&& uv pip install -e '.[dev,rl]' "
            "&& uv pip install -r apps/mission-control/api/requirements.txt"
        )

    if require_npm and shutil.which("npm") is None:
        errors.append("`npm` was not found on PATH. Install Node.js 20+ and retry.")

    if check_python_imports and not errors:
        result = subprocess.run(
            [str(python_path), "-c", "import fastapi, uvicorn, flightlab"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            errors.append(
                "Mission-control Python dependencies are missing. Run: "
                "uv pip install -e '.[dev,rl]' && "
                "uv pip install -r apps/mission-control/api/requirements.txt"
            )

    return errors


def install_frontend_deps_if_needed(base_dir: Path | None = None) -> None:
    """Install frontend dependencies when node_modules is absent."""
    root = base_dir or repo_root()
    web_root = root / "apps/mission-control/web"
    if (web_root / "node_modules").exists():
        return
    subprocess.run(["npm", "install"], cwd=web_root, check=True)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate a service process group."""
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def launch_local_services(base_dir: Path | None = None, *, skip_npm_install: bool = False) -> int:
    """Launch both local services and wait until shutdown."""
    root = base_dir or repo_root()
    errors = missing_prerequisites(base_dir=root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if not skip_npm_install:
        install_frontend_deps_if_needed(root)

    services = [build_backend_service(root), build_frontend_service(root)]
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    print("Starting mission-control services...")
    print("UI:   http://localhost:5173")
    print("API:  http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("Press Ctrl-C to stop both services.")

    try:
        for service in services:
            process = subprocess.Popen(
                list(service.argv),
                cwd=service.cwd,
                start_new_session=True,
            )
            processes.append((service.name, process))

        while True:
            for name, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        print(
                            f"{name} exited with status {return_code}. Shutting down.",
                            file=sys.stderr,
                        )
                    return return_code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping mission-control services...")
        return 0
    finally:
        for _name, process in reversed(processes):
            _terminate_process(process)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the single-command launcher."""
    parser = argparse.ArgumentParser(description="Run mission-control locally.")
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Do not run npm install automatically when node_modules is missing.",
    )
    args = parser.parse_args(argv)
    return launch_local_services(skip_npm_install=args.skip_npm_install)


if __name__ == "__main__":
    raise SystemExit(main())
