"""Read-only FastAPI endpoints over the latest scheduled analytics snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.resources import files

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from deriv_vol_lab.api.snapshot import (
    AnalyticsSnapshot,
    ScreenSnapshotRow,
    SnapshotUnavailableError,
    SymbolQualityStatus,
    default_snapshot_path,
    load_snapshot,
)
from deriv_vol_lab.data.quality import DataQualityReport
from deriv_vol_lab.stats.analysis import VolatilityMonitorReport

app = FastAPI(
    title="Deriv Vol Lab",
    description="Read-only descriptive analytics over scheduled snapshots.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    snapshot_available: bool
    generated_at_epoch: int | None
    snapshot_age_seconds: int | None


class ScreenResponse(BaseModel):
    generated_at_epoch: int
    granularity: int
    rows: list[ScreenSnapshotRow]


class DataQualityResponse(BaseModel):
    generated_at_epoch: int
    by_symbol: list[SymbolQualityStatus]
    report: DataQualityReport


class VolatilityMonitorResponse(BaseModel):
    generated_at_epoch: int
    report: VolatilityMonitorReport


def _snapshot() -> AnalyticsSnapshot:
    try:
        return load_snapshot(default_snapshot_path())
    except SnapshotUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    """Serve the static dashboard from the installed package."""
    return files("deriv_vol_lab.api").joinpath("dashboard.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health and whether a scheduled snapshot can be served."""
    try:
        snapshot = load_snapshot(default_snapshot_path())
    except SnapshotUnavailableError:
        return HealthResponse(
            status="degraded",
            snapshot_available=False,
            generated_at_epoch=None,
            snapshot_age_seconds=None,
        )
    now = int(datetime.now(UTC).timestamp())
    return HealthResponse(
        status="ok",
        snapshot_available=True,
        generated_at_epoch=snapshot.generated_at_epoch,
        snapshot_age_seconds=max(0, now - snapshot.generated_at_epoch),
    )


@app.get("/screen", response_model=ScreenResponse)
def screen() -> ScreenResponse:
    """Return the latest descriptive screen and snapshot timestamp."""
    snapshot = _snapshot()
    return ScreenResponse(
        generated_at_epoch=snapshot.generated_at_epoch,
        granularity=snapshot.granularity,
        rows=snapshot.screen,
    )


@app.get("/data-quality", response_model=DataQualityResponse)
def data_quality() -> DataQualityResponse:
    """Return quality diagnostics and per-symbol status from the same snapshot."""
    snapshot = _snapshot()
    return DataQualityResponse(
        generated_at_epoch=snapshot.generated_at_epoch,
        by_symbol=snapshot.quality_by_symbol,
        report=snapshot.quality,
    )


@app.get("/stats/vol-monitor", response_model=VolatilityMonitorResponse)
def volatility_monitor() -> VolatilityMonitorResponse:
    """Return the latest 99% chi-square volatility-monitor window per symbol."""
    snapshot = _snapshot()
    return VolatilityMonitorResponse(
        generated_at_epoch=snapshot.generated_at_epoch,
        report=snapshot.volatility_monitor,
    )
