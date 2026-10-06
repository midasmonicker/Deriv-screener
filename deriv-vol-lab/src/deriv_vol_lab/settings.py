"""Environment-backed application settings."""

from pathlib import Path
from typing import Literal

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
    db_path: Path = Path("data/deriv-vol-lab.duckdb")
    snapshot_path: Path = Path("snapshots/latest.json")
    log_level: LogLevel = "INFO"
