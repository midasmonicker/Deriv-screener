"""Tests for environment-backed settings."""

import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError

from deriv_vol_lab.settings import Settings


def test_settings_load_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings map the documented environment variables to typed fields."""
    monkeypatch.setenv("DERIV_APP_ID", "12345")
    monkeypatch.setenv("DERIV_ENDPOINT", "wss://example.test/websockets/v3")
    monkeypatch.setenv("DB_PATH", "tmp/test.db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.deriv_app_id == 12345
    assert settings.deriv_endpoint == "wss://example.test/websockets/v3"
    assert settings.deriv_handshake_max_attempts == 5
    assert settings.db_path == Path("tmp/test.db")
    assert settings.log_level == "DEBUG"


def test_settings_default_deriv_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default endpoint is the current Deriv public market-data socket."""
    monkeypatch.delenv("DERIV_ENDPOINT", raising=False)

    assert Settings().deriv_endpoint == "wss://api.derivws.com/trading/v1/options/ws/public"


def test_settings_app_id_is_optional_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DERIV_APP_ID", raising=False)

    assert Settings().deriv_app_id is None


def test_settings_handshake_attempts_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="deriv_handshake_max_attempts"):
        Settings(deriv_handshake_max_attempts=0)


@pytest.mark.parametrize(
    "endpoint",
    [
        "wss://://binaryws.com",
        "https://ws.derivws.com",
        "",
        "wss://ws.derivws.com/has whitespace",
    ],
)
def test_settings_reject_invalid_deriv_endpoint(endpoint: str) -> None:
    """Invalid endpoint overrides fail validation with the field name."""
    with pytest.raises(ValidationError, match="deriv_endpoint"):
        Settings(deriv_endpoint=endpoint)


def test_package_entry_point_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The package entry point initializes and logs without business logic."""
    monkeypatch.setattr("sys.argv", ["deriv-vol-lab"])
    runpy.run_module("deriv_vol_lab", run_name="__main__")

    assert "deriv_vol_lab.ready" in capsys.readouterr().out


@pytest.mark.parametrize("blank", ["", "   "])
def test_settings_blank_app_id_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """GitHub Actions exports an undefined variable as an empty string."""
    monkeypatch.setenv("DERIV_APP_ID", blank)

    assert Settings().deriv_app_id is None


def test_settings_numeric_app_id_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DERIV_APP_ID", "12345")

    assert Settings().deriv_app_id == 12345


def test_settings_non_numeric_app_id_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DERIV_APP_ID", "abc")

    with pytest.raises(ValidationError):
        Settings()
