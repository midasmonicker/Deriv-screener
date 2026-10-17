"""Environment-backed application settings."""

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix="",
        extra="ignore",
    )

    deriv_app_id: int | None = None
    deriv_endpoint: str = "wss://api.derivws.com/trading/v1/options/ws/public"
    deriv_handshake_max_attempts: Annotated[int, Field(gt=0)] = 5
    db_path: Path = Path("data/deriv-vol-lab.duckdb")
    snapshot_path: Path = Path("snapshots/latest.json")
    log_level: LogLevel = "INFO"

    @field_validator("deriv_app_id", mode="before")
    @classmethod
    def blank_app_id_is_unset(cls, value: object) -> object:
        """Treat a blank DERIV_APP_ID as unset.

        CI systems such as GitHub Actions export an undefined variable as an empty
        string, and the app ID is optional.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("deriv_endpoint")
    @classmethod
    def validate_deriv_endpoint(cls, value: str) -> str:
        """Require a well-formed secure Deriv WebSocket endpoint."""
        if not value or any(character.isspace() for character in value):
            raise ValueError("deriv_endpoint must not be empty or contain whitespace")
        if value.count("://") != 1:
            raise ValueError("deriv_endpoint must contain exactly one '://'")
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError("deriv_endpoint must be a valid wss URL with a hostname") from exc
        if parsed.scheme != "wss" or not parsed.hostname:
            raise ValueError("deriv_endpoint must use wss and include a hostname")
        return value
