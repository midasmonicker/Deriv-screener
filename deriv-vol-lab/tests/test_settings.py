"""Tests for environment-backed settings."""

import runpy
from pathlib import Path

import pytest

from deriv_vol_lab.settings import Settings


def test_settings_load_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings map the documented environment variables to typed fields."""
    monkeypatch.setenv("DERIV_APP_ID", "12345")
    monkeypatch.setenv("DB_PATH", "tmp/test.db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.deriv_app_id == 12345
    assert settings.db_path == Path("tmp/test.db")
    assert settings.log_level == "DEBUG"


def test_package_entry_point_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The package entry point initializes and logs without business logic."""
    runpy.run_module("deriv_vol_lab", run_name="__main__")

    assert "deriv_vol_lab.ready" in capsys.readouterr().out
