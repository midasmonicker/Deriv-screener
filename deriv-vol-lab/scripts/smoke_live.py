"""Network smoke check against Deriv's public, read-only market-data API."""

from __future__ import annotations

import asyncio
import time

import pandas as pd

from deriv_vol_lab.data import SYMBOLS, DerivWebSocketClient, DuckDBStorage
from deriv_vol_lab.data.quality import DataQualityReport
from deriv_vol_lab.settings import Settings
from deriv_vol_lab.stats.analysis import (
    VolatilityWindowResult,
    rolling_volatility_monitor,
)

GRANULARITY_SECONDS = 60
BACKFILL_DAYS = 3
MINIMUM_COVERAGE_PCT = 99.0
MONITOR_WINDOW = 500
CONFIDENCE_LEVEL = 0.99


async def run_smoke() -> int:
    """Backfill all registered symbols, report quality/volatility, and return status."""
    settings = Settings()
    if settings.deriv_app_id is None:
        raise ValueError("DERIV_APP_ID must be set to use the public Deriv API")

    end_epoch = int(time.time()) // GRANULARITY_SECONDS * GRANULARITY_SECONDS - GRANULARITY_SECONDS
    start_epoch = end_epoch - BACKFILL_DAYS * 86_400 + GRANULARITY_SECONDS
    client = DerivWebSocketClient(settings.deriv_app_id)
    try:
        await client.connect(validate_symbols=True)
        with DuckDBStorage(settings.db_path) as storage:
            for symbol in SYMBOLS:
                candles = await client.backfill_candles(
                    symbol,
                    BACKFILL_DAYS,
                    granularity=GRANULARITY_SECONDS,
                    end_epoch=end_epoch,
                )
                storage.write_candles(candles)

            records = storage.read_candles(
                GRANULARITY_SECONDS,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            )
            quality = storage.quality_report(
                records=records,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            )
            missing_by_symbol = {
                symbol: len(
                    storage.find_gaps(
                        symbol,
                        start_epoch,
                        end_epoch,
                        granularity=GRANULARITY_SECONDS,
                    )
                )
                for symbol in SYMBOLS
            }
    finally:
        await client.close()

    closes: dict[str, pd.Series[float]] = {}
    for symbol in SYMBOLS:
        symbol_candles = [candle for candle in records if candle.symbol == symbol]
        if symbol_candles:
            closes[symbol] = pd.Series(
                [float(candle.close) for candle in symbol_candles],
                index=pd.Index([candle.epoch for candle in symbol_candles], name="epoch"),
                dtype=float,
            )
    monitor = rolling_volatility_monitor(
        closes,
        window=MONITOR_WINDOW,
        confidence_level=CONFIDENCE_LEVEL,
        sampling_intervals={symbol: GRANULARITY_SECONDS for symbol in SYMBOLS},
    )
    latest_monitor = {result.symbol: result for result in monitor.windows}
    coverage_by_symbol = _coverage_by_symbol(quality)

    failed = False
    print(
        f"Deriv public market-data smoke check: {BACKFILL_DAYS} days, "
        f"{GRANULARITY_SECONDS}s candles, {CONFIDENCE_LEVEL:.0%} CI"
    )
    for symbol in SYMBOLS:
        coverage = coverage_by_symbol.get(symbol, (0, 0))
        observed, expected = coverage
        coverage_pct = 100.0 * observed / expected if expected else 0.0
        result = latest_monitor.get(symbol)
        _print_symbol(
            symbol,
            missing_by_symbol[symbol],
            coverage_pct,
            result,
        )
        if coverage_pct < MINIMUM_COVERAGE_PCT:
            print(f"  ALERT: coverage {coverage_pct:.3f}% is below {MINIMUM_COVERAGE_PCT:.1f}%")
            failed = True
        if result is None or result.status != "ok" or result.alert is True:
            failed = True
    return 1 if failed else 0


def main() -> None:
    """Run the live smoke check as a command-line script."""
    raise SystemExit(asyncio.run(run_smoke()))


def _coverage_by_symbol(report: DataQualityReport) -> dict[str, tuple[int, int]]:
    totals: dict[str, tuple[int, int]] = {}
    for item in report.daily_coverage:
        if item.record_kind != "candle" or item.granularity != GRANULARITY_SECONDS:
            continue
        observed, expected = totals.get(item.symbol, (0, 0))
        totals[item.symbol] = (
            observed + item.observed_records,
            expected + item.expected_records,
        )
    return totals


def _print_symbol(
    symbol: str,
    gap_count: int,
    coverage_pct: float,
    result: VolatilityWindowResult | None,
) -> None:
    prefix = f"{symbol}: gaps={gap_count}, coverage={coverage_pct:.3f}%"
    if result is None or result.status != "ok":
        observations = 0 if result is None else result.observations
        print(f"{prefix}, vol=insufficient data ({observations}/{MONITOR_WINDOW} returns)")
        print("  ALERT: no valid 99% volatility window")
        return

    stated = result.stated_volatility
    realized = result.realized_volatility
    lower = result.lower_confidence_bound
    upper = result.upper_confidence_bound
    if stated <= 0 or realized is None or lower is None or upper is None:
        print(f"{prefix}, vol=invalid monitor result")
        print("  ALERT: volatility monitor returned invalid bounds")
        return

    print(
        f"{prefix}, vol ratio={realized / stated:.4f}, "
        f"99% CI=[{lower / stated:.4f}, {upper / stated:.4f}]"
    )
    if result.alert:
        print("  ALERT: stated volatility is outside the 99% confidence interval")


if __name__ == "__main__":
    main()
