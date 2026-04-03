from __future__ import annotations

from pathlib import Path

from app.dev_launcher import (
    build_backend_service,
    build_frontend_service,
    missing_prerequisites,
    repo_python_path,
)


def test_build_backend_service_uses_repo_venv(tmp_path: Path) -> None:
    service = build_backend_service(tmp_path)
    assert service.name == "api"
    assert service.cwd == tmp_path / "apps/mission-control/api"
    assert service.argv[:4] == (
        str(repo_python_path(tmp_path)),
        "-m",
        "uvicorn",
        "app.main:app",
    )


def test_build_frontend_service_uses_web_dir(tmp_path: Path) -> None:
    service = build_frontend_service(tmp_path)
    assert service.name == "web"
    assert service.cwd == tmp_path / "apps/mission-control/web"
    assert service.argv == ("npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173")


def test_missing_prerequisites_reports_missing_venv_and_npm(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("app.dev_launcher.shutil.which", lambda _name: None)
    errors = missing_prerequisites(base_dir=tmp_path, check_python_imports=False)
    assert any("Repo virtualenv is missing" in error for error in errors)
    assert any("`npm` was not found" in error for error in errors)
