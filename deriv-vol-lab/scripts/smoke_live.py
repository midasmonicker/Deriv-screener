"""Network smoke check against Deriv's public, read-only market-data API."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from math import comb, isfinite, sqrt
from typing import TypeVar

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from deriv_vol_lab.data import (
    SYMBOLS,
    Candle,
    DerivWebSocketClient,
    DuckDBStorage,
    HandshakeConnectionError,
)
from deriv_vol_lab.data.quality import DataQualityReport
from deriv_vol_lab.features.screener import screen_candles
from deriv_vol_lab.settings import Settings
from deriv_vol_lab.stats.analysis import (
    VolatilityMonitorReport,
    VolatilityWindowResult,
    benjamini_hochberg,
    rolling_volatility_monitor,
)

GRANULARITY_SECONDS = 60
BACKFILL_DAYS = 7
MINIMUM_COVERAGE_PCT = 99.0
MONITOR_WINDOW = 500
CONFIDENCE_LEVEL = 0.99
BREACH_RATE_NULL = 0.01
FDR_THRESHOLD = 0.01
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
_T = TypeVar("_T")


class _SmokeDeadlineExceeded(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


async def _await_with_deadline(
    operation: Callable[[], Awaitable[_T]],
    deadline: float | None,
    stage: str,
) -> _T:
    if deadline is None:
        return await operation()
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise _SmokeDeadlineExceeded(stage)
    try:
        return await asyncio.wait_for(operation(), timeout=remaining)
    except TimeoutError as exc:
        raise _SmokeDeadlineExceeded(stage) from exc


def _check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise _SmokeDeadlineExceeded(stage)


async def run_smoke(days: int | None = None, *, max_seconds: float | None = None) -> int:
    """Backfill all registered symbols, report quality/volatility, and return status."""
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be a positive number")
    deadline = time.perf_counter() + max_seconds if max_seconds is not None else None
    settings = Settings()
    backfill_days = BACKFILL_DAYS if days is None else days
    if backfill_days <= 0:
        raise ValueError("days must be a positive integer")

    end_epoch = int(time.time()) // GRANULARITY_SECONDS * GRANULARITY_SECONDS - GRANULARITY_SECONDS
    start_epoch = end_epoch - backfill_days * 86_400 + GRANULARITY_SECONDS

    client = DerivWebSocketClient(
        settings.deriv_app_id,
        endpoint=settings.deriv_endpoint,
        max_handshake_attempts=settings.deriv_handshake_max_attempts,
    )

    try:
        await _await_with_deadline(
            lambda: client.connect(validate_symbols=True), deadline, "connect"
        )

        available_symbols = client.available_symbols
        candles_by_symbol: dict[str, list[Candle]] = {}
        with DuckDBStorage(settings.db_path) as storage:
            for symbol in SYMBOLS:
                if symbol not in available_symbols:
                    print(f"{symbol}: skipped (not present in active_symbols)")
                    continue
                _check_deadline(deadline, "fetch")
                stage_started = time.perf_counter()
                candles = await _await_with_deadline(
                    lambda symbol=symbol: client.backfill_candles(
                        symbol,
                        backfill_days,
                        granularity=GRANULARITY_SECONDS,
                        end_epoch=end_epoch,
                    ),
                    deadline,
                    "fetch",
                )
                fetch_seconds = time.perf_counter() - stage_started

                _check_deadline(deadline, "write")
                stage_started = time.perf_counter()
                storage.write_candles(candles)
                _check_deadline(deadline, "write")
                write_seconds = time.perf_counter() - stage_started
                candles_by_symbol[symbol] = candles
                print(
                    f"{symbol}: fetched {len(candles)} candles in {fetch_seconds:.3f} s, "
                    f"wrote in {write_seconds:.3f} s"
                )

            records_by_symbol: dict[str, list[Candle]] = {}
            quality_reports: list[DataQualityReport] = []
            missing_by_symbol: dict[str, int] = {}
            for symbol in candles_by_symbol:
                _check_deadline(deadline, "read-back")
                stage_started = time.perf_counter()
                records_by_symbol[symbol] = storage.read_candles(
                    GRANULARITY_SECONDS,
                    symbol,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                )
                missing_by_symbol[symbol] = len(
                    storage.find_gaps(
                        symbol,
                        start_epoch,
                        end_epoch,
                        granularity=GRANULARITY_SECONDS,
                    )
                )
                _check_deadline(deadline, "read-back")
                print(f"{symbol}: read-back in {time.perf_counter() - stage_started:.3f} s")

                _check_deadline(deadline, "quality report")
                stage_started = time.perf_counter()
                quality_reports.append(
                    storage.quality_report(
                        records=records_by_symbol[symbol],
                        start_epoch=start_epoch,
                        end_epoch=end_epoch,
                    )
                )
                _check_deadline(deadline, "quality report")
                print(f"{symbol}: quality report in {time.perf_counter() - stage_started:.3f} s")
            quality = _merge_quality_reports(quality_reports)
    except _SmokeDeadlineExceeded as exc:
        print(f"Smoke check exceeded --max-seconds during {exc.stage}")
        return 2
    except HandshakeConnectionError as exc:
        print(str(exc))
        return 2
    finally:
        await client.close()

    closes: dict[str, pd.Series[float]] = {}
    monitor_windows: list[VolatilityWindowResult] = []
    try:
        for symbol, symbol_candles in candles_by_symbol.items():
            symbol_candles = records_by_symbol[symbol]
            if symbol_candles:
                closes[symbol] = pd.Series(
                    [float(candle.close) for candle in symbol_candles],
                    index=pd.Index([candle.epoch for candle in symbol_candles], name="epoch"),
                    dtype=float,
                )
                _check_deadline(deadline, "volatility monitor")
                stage_started = time.perf_counter()
                symbol_monitor = rolling_volatility_monitor(
                    {symbol: closes[symbol]},
                    window=MONITOR_WINDOW,
                    confidence_level=CONFIDENCE_LEVEL,
                    sampling_intervals={symbol: GRANULARITY_SECONDS},
                )
                monitor_windows.extend(symbol_monitor.windows)
                _check_deadline(deadline, "volatility monitor")
                print(
                    f"{symbol}: volatility monitor in {time.perf_counter() - stage_started:.3f} s"
                )

                _check_deadline(deadline, "screener")
                stage_started = time.perf_counter()
                screen_candles(
                    {symbol: symbol_candles},
                    latest_n=500,
                    permutations=1,
                    seed=0,
                )
                _check_deadline(deadline, "screener")
                print(f"{symbol}: screener in {time.perf_counter() - stage_started:.3f} s")
    except _SmokeDeadlineExceeded as exc:
        print(f"Smoke check exceeded --max-seconds during {exc.stage}")
        return 2
    monitor = VolatilityMonitorReport(
        confidence_level=CONFIDENCE_LEVEL,
        windows=monitor_windows,
    )
    latest_monitor = {result.symbol: result for result in monitor.windows}
    coverage_by_symbol = _coverage_by_symbol(quality)
    valid_windows = {
        symbol: [
            result
            for result in monitor.windows
            if result.symbol == symbol and result.status == "ok"
        ]
        for symbol in SYMBOLS
    }
    overlapping_breaches = {
        symbol: sum(result.alert is True for result in results)
        for symbol, results in valid_windows.items()
    }
    overlapping_counts = {symbol: len(results) for symbol, results in valid_windows.items()}
    non_overlapping_windows = {
        symbol: results[::MONITOR_WINDOW] for symbol, results in valid_windows.items()
    }
    non_overlapping_breaches = {
        symbol: sum(result.alert is True for result in results)
        for symbol, results in non_overlapping_windows.items()
    }
    non_overlapping_counts = {
        symbol: len(results) for symbol, results in non_overlapping_windows.items()
    }
    breach_rate_p_values = {
        symbol: (
            float(
                binomtest(
                    non_overlapping_breaches[symbol],
                    non_overlapping_counts[symbol],
                    BREACH_RATE_NULL,
                    alternative="greater",
                ).pvalue
            )
            if non_overlapping_counts[symbol]
            else 1.0
        )
        for symbol in SYMBOLS
    }
    adjusted_breach_rate_p_values = dict(
        zip(
            SYMBOLS,
            benjamini_hochberg([breach_rate_p_values[symbol] for symbol in SYMBOLS]),
            strict=True,
        )
    )
    latest_p_values = {
        symbol: _variance_test_p_value(results[-1]) if results else 1.0
        for symbol, results in valid_windows.items()
    }
    adjusted_latest_p_values = dict(
        zip(
            SYMBOLS,
            benjamini_hochberg([latest_p_values[symbol] for symbol in SYMBOLS]),
            strict=True,
        )
    )

    failed = False
    print(
        f"Deriv public market-data smoke check: {backfill_days} days, "
        f"{GRANULARITY_SECONDS}s candles, {CONFIDENCE_LEVEL:.0%} CI"
    )
    for symbol in SYMBOLS:
        if symbol not in client.available_symbols:
            continue
        coverage = coverage_by_symbol.get(symbol, (0, 0))
        observed, expected = coverage
        coverage_pct = 100.0 * observed / expected if expected else 0.0
        result = latest_monitor.get(symbol)
        _print_symbol(
            symbol,
            missing_by_symbol[symbol],
            coverage_pct,
            result,
            overlapping_breaches[symbol],
            overlapping_counts[symbol],
            non_overlapping_breaches[symbol],
            non_overlapping_counts[symbol],
            adjusted_breach_rate_p_values[symbol],
            adjusted_latest_p_values[symbol],
        )
        if non_overlapping_counts[symbol] < 10:
            print(
                f"  WARNING: only {non_overlapping_counts[symbol]} "
                "non-overlapping volatility windows; fewer than 10"
            )
        if coverage_pct < MINIMUM_COVERAGE_PCT:
            print(f"  ALERT: coverage {coverage_pct:.3f}% is below {MINIMUM_COVERAGE_PCT:.1f}%")
            failed = True
        if result is None or result.status != "ok":
            failed = True
        elif adjusted_latest_p_values[symbol] < FDR_THRESHOLD:
            print("  ALERT: latest volatility window is outside the BH-adjusted 99% CI")
            failed = True
        if adjusted_breach_rate_p_values[symbol] < FDR_THRESHOLD:
            print(
                "  ALERT: non-overlapping breach rate is significantly above "
                f"the {BREACH_RATE_NULL:.0%} null rate "
                f"(BH-adjusted p={adjusted_breach_rate_p_values[symbol]:.4g})"
            )
            failed = True
    realized_vol_ratios = {
        symbol: result.realized_volatility / result.stated_volatility
        for symbol, result in latest_monitor.items()
        if result.status == "ok"
        and result.realized_volatility is not None
        and result.stated_volatility > 0
    }
    _print_cross_symbol_diagnostics(closes, realized_vol_ratios)
    return 1 if failed else 0


def main() -> None:
    """Run the live smoke check as a command-line script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=BACKFILL_DAYS,
        help=f"number of days to backfill (default: {BACKFILL_DAYS})",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="abort when the total runtime exceeds this many seconds",
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be a positive number")
    raise SystemExit(asyncio.run(run_smoke(days=args.days, max_seconds=args.max_seconds)))


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


def _merge_quality_reports(reports: list[DataQualityReport]) -> DataQualityReport:
    return DataQualityReport(
        duplicate_epochs=[item for report in reports for item in report.duplicate_epochs],
        non_monotonic_timestamps=[
            item for report in reports for item in report.non_monotonic_timestamps
        ],
        invalid_prices=[item for report in reports for item in report.invalid_prices],
        abnormal_jumps=[item for report in reports for item in report.abnormal_jumps],
        daily_coverage=[item for report in reports for item in report.daily_coverage],
    )


def _print_symbol(
    symbol: str,
    gap_count: int,
    coverage_pct: float,
    result: VolatilityWindowResult | None,
    overlapping_breach_count: int,
    overlapping_window_count: int,
    non_overlapping_breach_count: int,
    non_overlapping_window_count: int,
    breach_rate_adjusted_p_value: float,
    latest_adjusted_p_value: float,
) -> None:
    prefix = f"{symbol}: gaps={gap_count}, coverage={coverage_pct:.3f}%"
    overlapping_fraction = (
        overlapping_breach_count / overlapping_window_count if overlapping_window_count else 0.0
    )
    non_overlapping_fraction = (
        non_overlapping_breach_count / non_overlapping_window_count
        if non_overlapping_window_count
        else 0.0
    )
    breach_summary = (
        f"overlapping, informational={overlapping_breach_count}/"
        f"{overlapping_window_count} ({overlapping_fraction:.2%}); "
        f"non-overlapping breaches={non_overlapping_breach_count}/"
        f"{non_overlapping_window_count} ({non_overlapping_fraction:.2%}), "
        f"BH-adjusted p={breach_rate_adjusted_p_value:.4g}; "
        f"latest BH-adjusted p={latest_adjusted_p_value:.4g}"
    )
    if result is None or result.status != "ok":
        observations = 0 if result is None else result.observations
        print(
            f"{prefix}, vol=insufficient data "
            f"({observations}/{MONITOR_WINDOW} returns), {breach_summary}"
        )
        print("  ALERT: no valid 99% volatility window")
        return

    stated = result.stated_volatility
    realized = result.realized_volatility
    lower = result.lower_confidence_bound
    upper = result.upper_confidence_bound
    if stated <= 0 or realized is None or lower is None or upper is None:
        print(f"{prefix}, vol=invalid monitor result, {breach_summary}")
        print("  ALERT: volatility monitor returned invalid bounds")
        return

    print(
        f"{prefix}, vol ratio={realized / stated:.4f}, "
        f"99% CI=[{lower / stated:.4f}, {upper / stated:.4f}], {breach_summary}"
    )


def _variance_test_p_value(result: VolatilityWindowResult) -> float:
    """Two-sided chi-square p-value for the latest window's stated-volatility test."""
    if (
        result.status != "ok"
        or result.realized_volatility is None
        or result.stated_volatility <= 0
        or result.observations < 2
    ):
        return 1.0
    statistic = (result.observations - 1) * (
        result.realized_volatility / result.stated_volatility
    ) ** 2
    degrees_of_freedom = result.observations - 1
    p_value = 2 * min(
        float(chi2.cdf(statistic, degrees_of_freedom)),
        float(chi2.sf(statistic, degrees_of_freedom)),
    )
    return min(1.0, p_value)


def _cross_symbol_diagnostics(
    closes: dict[str, pd.Series[float]],
) -> tuple[pd.DataFrame, float, float, bool, int]:
    """Summarize aligned one-minute log-return correlations across symbols."""
    return_series: dict[str, pd.Series[float]] = {}
    for symbol, series in closes.items():
        ordered_series = series.sort_index()
        if len(ordered_series) < 2:
            continue
        valid_prices = ordered_series.where(np.isfinite(ordered_series) & (ordered_series > 0))
        log_returns = np.log(valid_prices).diff()
        one_minute_intervals = ordered_series.index.to_series().diff() == GRANULARITY_SECONDS
        return_series[symbol] = log_returns.where(one_minute_intervals)
    aligned_returns = (
        pd.concat(return_series, axis=1, join="inner").dropna() if return_series else pd.DataFrame()
    )
    matrix = aligned_returns.corr()
    off_diagonal = [
        abs(float(matrix.iloc[row, column]))
        for row in range(len(matrix.columns))
        for column in range(row + 1, len(matrix.columns))
        if pd.notna(matrix.iloc[row, column])
    ]
    maximum = max(off_diagonal, default=0.0)
    observation_count = len(aligned_returns)
    threshold = 4 / sqrt(observation_count) if observation_count else float("inf")
    return matrix, maximum, threshold, maximum > threshold, observation_count


def _count_identical_rounded_ratios(ratios: dict[str, float]) -> int:
    """Count symbol pairs whose realized-vol ratios match at three decimals."""
    counts = Counter(round(ratio, 3) for ratio in ratios.values() if isfinite(ratio))
    return sum(comb(count, 2) for count in counts.values() if count > 1)


def _print_cross_symbol_diagnostics(
    closes: dict[str, pd.Series[float]],
    realized_vol_ratios: dict[str, float],
) -> None:
    matrix, maximum, threshold, exceeds_threshold, observations = _cross_symbol_diagnostics(closes)
    print(f"Cross-symbol 1-minute log-return correlation matrix (aligned n={observations}):")
    if matrix.empty:
        print("  no aligned returns")
    else:
        print(matrix.to_string(float_format=lambda value: f"{value:.3f}"))
    threshold_text = f"{threshold:.4f}" if isfinite(threshold) else "unavailable"
    print(
        f"Cross-symbol max |corr|={maximum:.4f}; threshold={threshold_text}; "
        f"exceeds threshold={exceeds_threshold} (informational)"
    )
    print(
        "Pairs with identical realized-vol ratios to 3 decimals="
        f"{_count_identical_rounded_ratios(realized_vol_ratios)} (informational)"
    )


if __name__ == "__main__":
    main()
