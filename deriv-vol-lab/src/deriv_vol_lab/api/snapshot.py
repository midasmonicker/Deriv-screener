"""Build, persist, and load immutable analytics snapshots."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from deriv_vol_lab.data.models import Candle
from deriv_vol_lab.data.quality import DataQualityReport, build_quality_report
from deriv_vol_lab.data.symbols import SYMBOLS
from deriv_vol_lab.features import screen_candles
from deriv_vol_lab.stats.analysis import VolatilityMonitorReport, rolling_volatility_monitor

GRANULARITY_SECONDS = 60
SCREEN_LOOKBACK = 2_000
VOL_MONITOR_WINDOW = 500
SCREEN_PERMUTATIONS = 199
FlagReliability = dict[str, float | int | None]


class ScreenSnapshotRow(BaseModel):
    """JSON-safe descriptive market snapshot for one synthetic index."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    epoch: int
    price: float | None
    rsi: float | None
    bollinger_z: float | None
    ema20: float | None
    ema50: float | None
    ema20_50_trend: str
    realised_vol: float | None
    stated_vol: float
    vol_ratio: float | None
    rolling_autocorrelation: float | None
    hurst: float | None
    variance_ratio: float | None
    flags: list[str]
    reliability: dict[str, FlagReliability]


class SymbolQualityStatus(BaseModel):
    """Recent per-symbol data health for the dashboard."""

    symbol: str
    status: Literal["healthy", "warning", "missing"]
    coverage_pct: float | None
    duplicate_epochs: int
    non_monotonic_timestamps: int
    invalid_prices: int
    abnormal_jumps: int
    last_epoch: int | None


class AnalyticsSnapshot(BaseModel):
    """Read-only dashboard payload written by the scheduled ingestion job."""

    model_config = ConfigDict(frozen=True)

    generated_at_epoch: int
    granularity: int
    candle_count: int
    screen: list[ScreenSnapshotRow]
    quality: DataQualityReport
    quality_by_symbol: list[SymbolQualityStatus]
    volatility_monitor: VolatilityMonitorReport


def build_snapshot(
    candles_by_symbol: dict[str, list[Candle]],
    *,
    generated_at_epoch: int,
    seed: int = 0,
) -> AnalyticsSnapshot:
    """Compute all read-only API views from one consistent set of candles."""
    if generated_at_epoch < 0:
        raise ValueError("generated_at_epoch must be non-negative")
    records = [candle for candles in candles_by_symbol.values() for candle in candles]
    all_candles = [candle for candle in records if candle.granularity == GRANULARITY_SECONDS]
    if len(all_candles) != len(records):
        raise ValueError("Snapshot candles must use 60-second granularity")
    if any(candle.symbol not in SYMBOLS for candle in all_candles):
        raise ValueError("Snapshot contains a symbol outside the local registry")
    canonical_candles = {
        symbol: sorted(candles, key=lambda candle: candle.epoch)[-SCREEN_LOOKBACK:]
        for symbol, candles in candles_by_symbol.items()
        if candles
    }
    screen_frame = screen_candles(
        canonical_candles,
        latest_n=SCREEN_LOOKBACK,
        permutations=SCREEN_PERMUTATIONS,
        seed=seed,
    )
    screen_rows = [_screen_row(row) for row in screen_frame.to_dict(orient="records")]

    if records:
        start_epoch = min(candle.epoch for candle in all_candles)
        end_epoch = max(candle.epoch for candle in all_candles)
        quality = build_quality_report(
            records,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
        )
    else:
        quality = build_quality_report([])
    quality_by_symbol = _quality_statuses(all_candles, quality)

    monitor_closes = {
        symbol: pd.Series(
            [float(candle.close) for candle in candles[-(VOL_MONITOR_WINDOW + 1) :]],
            index=pd.Index([candle.epoch for candle in candles[-(VOL_MONITOR_WINDOW + 1) :]]),
            dtype=float,
        )
        for symbol, candles in canonical_candles.items()
        if len(candles) >= VOL_MONITOR_WINDOW + 1
    }
    monitor = rolling_volatility_monitor(
        monitor_closes,
        window=VOL_MONITOR_WINDOW,
        confidence_level=0.99,
        sampling_intervals={symbol: GRANULARITY_SECONDS for symbol in monitor_closes},
    )
    return AnalyticsSnapshot(
        generated_at_epoch=generated_at_epoch,
        granularity=GRANULARITY_SECONDS,
        candle_count=len(all_candles),
        screen=screen_rows,
        quality=quality,
        quality_by_symbol=quality_by_symbol,
        volatility_monitor=monitor,
    )


def write_snapshot(snapshot: AnalyticsSnapshot, path: Path) -> None:
    """Atomically persist a JSON snapshot to the configured shared snapshot path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(snapshot.model_dump_json(indent=2))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_snapshot(path: Path) -> AnalyticsSnapshot:
    """Load and validate the current persisted dashboard snapshot."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SnapshotUnavailableError(f"Analytics snapshot does not exist: {path}") from exc
    try:
        return AnalyticsSnapshot.model_validate_json(content)
    except ValueError as exc:
        raise SnapshotUnavailableError(f"Analytics snapshot is invalid: {path}") from exc


class SnapshotUnavailableError(RuntimeError):
    """Raised when no valid scheduled analytics snapshot is available."""


def _screen_row(row: Mapping[object, object]) -> ScreenSnapshotRow:
    def optional_number(value: object) -> float | None:
        if isinstance(value, (int, float)) and isfinite(float(value)):
            return float(value)
        return None

    reliability: dict[str, FlagReliability] = {}
    for flag in (
        "overbought",
        "stretched",
        "vol_mismatch",
        "trending_regime",
        "mean_reverting_regime",
    ):
        reliability[flag] = {
            "edge_calibration": optional_number(row.get(f"{flag}_edge_calibration")),
            "p_value_unadjusted_exploratory": optional_number(
                row.get(f"{flag}_p_value_unadjusted_exploratory")
            ),
            "p_value_adjusted": optional_number(row.get(f"{flag}_p_value_adjusted")),
            "hit_rate": optional_number(row.get(f"{flag}_hit_rate")),
            "observations": optional_number(row.get(f"{flag}_observations")),
        }
    flags_value = row.get("flags", ())
    flags = [str(flag) for flag in flags_value] if isinstance(flags_value, (tuple, list)) else []
    symbol = row.get("symbol")
    epoch = row.get("epoch")
    if not isinstance(symbol, str):
        raise ValueError("Screener row is missing its symbol")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError("Screener row is missing an integer epoch")
    return ScreenSnapshotRow(
        symbol=symbol,
        epoch=epoch,
        price=optional_number(row.get("price")),
        rsi=optional_number(row.get("rsi")),
        bollinger_z=optional_number(row.get("bollinger_z")),
        ema20=optional_number(row.get("ema20")),
        ema50=optional_number(row.get("ema50")),
        ema20_50_trend=str(row.get("ema20_50_trend", "insufficient_data")),
        realised_vol=optional_number(row.get("realised_vol")),
        stated_vol=optional_number(row.get("stated_vol")) or 0.0,
        vol_ratio=optional_number(row.get("vol_ratio")),
        rolling_autocorrelation=optional_number(row.get("rolling_autocorrelation")),
        hurst=optional_number(row.get("hurst")),
        variance_ratio=optional_number(row.get("variance_ratio")),
        flags=flags,
        reliability=reliability,
    )


def _quality_statuses(
    candles: list[Candle],
    report: DataQualityReport,
) -> list[SymbolQualityStatus]:
    statuses: list[SymbolQualityStatus] = []
    for symbol in SYMBOLS:
        records = [candle for candle in candles if candle.symbol == symbol]
        if not records:
            statuses.append(
                SymbolQualityStatus(
                    symbol=symbol,
                    status="missing",
                    coverage_pct=None,
                    duplicate_epochs=0,
                    non_monotonic_timestamps=0,
                    invalid_prices=0,
                    abnormal_jumps=0,
                    last_epoch=None,
                )
            )
            continue
        coverage = [
            item.coverage_pct
            for item in report.daily_coverage
            if item.symbol == symbol and item.record_kind == "candle"
        ]
        duplicates = sum(item.symbol == symbol for item in report.duplicate_epochs)
        non_monotonic = sum(item.symbol == symbol for item in report.non_monotonic_timestamps)
        invalid = sum(item.symbol == symbol for item in report.invalid_prices)
        jumps = sum(item.symbol == symbol for item in report.abnormal_jumps)
        minimum_coverage = min(coverage) if coverage else 0.0
        healthy = (
            minimum_coverage >= 99.0
            and duplicates == 0
            and non_monotonic == 0
            and invalid == 0
            and jumps == 0
        )
        statuses.append(
            SymbolQualityStatus(
                symbol=symbol,
                status="healthy" if healthy else "warning",
                coverage_pct=minimum_coverage,
                duplicate_epochs=duplicates,
                non_monotonic_timestamps=non_monotonic,
                invalid_prices=invalid,
                abnormal_jumps=jumps,
                last_epoch=max(item.epoch for item in records),
            )
        )
    return statuses


def default_snapshot_path() -> Path:
    """Resolve the deployed or local snapshot JSON path."""
    from deriv_vol_lab.settings import Settings

    return Settings().snapshot_path
