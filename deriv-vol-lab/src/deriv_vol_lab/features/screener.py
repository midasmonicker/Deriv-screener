"""Descriptive cross-symbol screening with historical flag reliability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log, nan
from typing import Final

import pandas as pd

from deriv_vol_lab.data.models import Candle
from deriv_vol_lab.data.symbols import SYMBOLS, SymbolDefinition
from deriv_vol_lab.features.indicators import (
    bollinger_zscore,
    ema,
    hurst_rs,
    rolling_autocorrelation,
    rolling_variance_ratio,
    wilder_rsi,
)
from deriv_vol_lab.stats.analysis import benjamini_hochberg, evaluate_signal

SECONDS_PER_YEAR: Final[int] = 365 * 24 * 60 * 60

FLAG_NAMES: Final[tuple[str, ...]] = (
    "overbought",
    "stretched",
    "vol_mismatch",
    "trending_regime",
    "mean_reverting_regime",
)
MIN_RELIABILITY_OBSERVATIONS: Final[int] = 5
RELIABILITY_HORIZON: Final[int] = 1

SCREEN_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "epoch",
    "price",
    "rsi",
    "bollinger_z",
    "ema20",
    "ema50",
    "ema20_50_trend",
    "realised_vol",
    "stated_vol",
    "vol_ratio",
    "rolling_autocorrelation",
    "hurst",
    "variance_ratio",
    "flags",
    *(
        column
        for flag in FLAG_NAMES
        for column in (
            f"{flag}_edge_calibration",
            f"{flag}_p_value_unadjusted_exploratory",
            f"{flag}_p_value_adjusted",
            f"{flag}_hit_rate",
            f"{flag}_observations",
        )
    ),
)


def screen_candles(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    *,
    latest_n: int = 500,
    permutations: int = 499,
    calibration_fraction: float = 0.6,
    evaluation_fraction: float = 0.1,
    seed: int,
) -> pd.DataFrame:
    """Screen latest candles while isolating calibration and holdout windows.

    The chronological input is partitioned into calibration, later holdout
    evaluation, and current-display windows. Reliability edge estimates use
    only the latest contiguous segment of calibration; the evaluation window
    is excluded from displayed features and reliability.
    Unadjusted calibration p-values are exploratory; adjusted values use
    Benjamini-Hochberg across all available flag/symbol tests in this result.
    Fewer than five historical flag events are unavailable, not evidence of no
    edge. Input gaps split rolling calculations and forward outcomes into
    separate contiguous segments.
    Descriptive thresholds are RSI >= 70, absolute Bollinger z >= 2, realized
    volatility ratio outside [0.5, 1.5], trending Hurst > 0.55 with variance
    ratio > 1, and mean-reverting Hurst < 0.45 with variance ratio < 1.
    """
    if latest_n <= 0:
        raise ValueError("latest_n must be greater than zero")
    if permutations <= 0:
        raise ValueError("permutations must be greater than zero")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    if not 0.0 < evaluation_fraction < 1.0 or calibration_fraction + evaluation_fraction >= 1.0:
        raise ValueError("calibration_fraction + evaluation_fraction must be below one")

    rows: list[dict[str, object]] = []
    for symbol, candles in sorted(candles_by_symbol.items()):
        definition = _symbol_definition(symbol)
        if not candles:
            continue
        if any(candle.symbol != symbol for candle in candles):
            raise ValueError(f"Candles for {symbol} contain a different symbol")
        if any(candle.granularity <= 0 for candle in candles):
            raise ValueError(f"Candles for {symbol} have invalid granularity")
        granularities = {candle.granularity for candle in candles}
        if len(granularities) != 1:
            raise ValueError(f"Candles for {symbol} must share a granularity")

        ordered = sorted(candles, key=lambda candle: candle.epoch)[-latest_n:]
        epochs = [candle.epoch for candle in ordered]
        if len(set(epochs)) != len(epochs):
            raise ValueError(f"Candles for {symbol} contain duplicate epochs")
        calibration_end = int(len(ordered) * calibration_fraction)
        evaluation_end = int(len(ordered) * (calibration_fraction + evaluation_fraction))
        calibration_segments = [
            _frame(segment) for segment in _split_segments(ordered[:calibration_end])
        ]
        display_segments = _split_segments(ordered[evaluation_end:])
        if not display_segments:
            continue
        display_segment = display_segments[-1]
        result = _screen_symbol(
            symbol,
            definition,
            _frame(display_segment),
            calibration_segments[-1:],
            permutations=permutations,
            seed=seed,
        )
        rows.append(result)

    result_frame = pd.DataFrame(rows, columns=SCREEN_COLUMNS)
    p_value_columns = [f"{flag}_p_value_unadjusted_exploratory" for flag in FLAG_NAMES]
    p_values: list[float] = []
    locations: list[tuple[int, str]] = []
    for row_index, row in enumerate(rows):
        for column in p_value_columns:
            value = row[column]
            if isinstance(value, float) and isfinite(value):
                p_values.append(value)
                adjusted_column = column.removesuffix("_unadjusted_exploratory") + "_adjusted"
                locations.append((row_index, adjusted_column))
    adjusted = benjamini_hochberg(p_values)
    for (row_index, column), value in zip(locations, adjusted, strict=True):
        result_frame.at[row_index, column] = value
    return result_frame


def _frame(candles: Sequence[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [float(candle.open) for candle in candles],
            "high": [float(candle.high) for candle in candles],
            "low": [float(candle.low) for candle in candles],
            "close": [float(candle.close) for candle in candles],
        },
        index=pd.Index([candle.epoch for candle in candles], name="epoch"),
    )


def _split_segments(candles: Sequence[Candle]) -> list[list[Candle]]:
    segments: list[list[Candle]] = []
    current: list[Candle] = []
    for candle in candles:
        if current and candle.epoch - current[-1].epoch != candle.granularity:
            segments.append(current)
            current = []
        current.append(candle)
    if current:
        segments.append(current)
    return segments


def _screen_symbol(
    symbol: str,
    definition: SymbolDefinition,
    frame: pd.DataFrame,
    calibration_segments: Sequence[pd.DataFrame],
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    close = frame["close"]
    rsi_values = wilder_rsi(close, period=14)
    z_values = bollinger_zscore(close, window=20)
    ema20_values = ema(close, period=20)
    ema50_values = ema(close, period=50)
    realized_values = _realized_volatility_actual_time(close, frame.index, window=20)
    log_returns = pd.Series(close.map(log).diff(), index=close.index, dtype=float)
    autocorrelation_values = rolling_autocorrelation(log_returns, window=50, lag=1)
    hurst_values = hurst_rs(log_returns, window=64)
    variance_ratio_values = rolling_variance_ratio(log_returns, window=64, horizon=2)

    stated_volatility = definition.stated_vol_pct / 100.0
    realized = _latest_float(realized_values)
    ratio = realized / stated_volatility if pd.notna(realized) else nan
    current_rsi = _latest_float(rsi_values)
    current_z = _latest_float(z_values)
    current_ema20 = _latest_float(ema20_values)
    current_ema50 = _latest_float(ema50_values)
    current_hurst = _latest_float(hurst_values)
    current_variance_ratio = _latest_float(variance_ratio_values)
    current_autocorrelation = _latest_float(autocorrelation_values)

    flag_series = _make_flag_series(
        rsi_values,
        z_values,
        realized_values,
        stated_volatility,
        hurst_values,
        variance_ratio_values,
    )
    active_flags = [name for name, values in flag_series.items() if _latest_bool(values)]

    row: dict[str, object] = {
        "symbol": symbol,
        "epoch": int(frame.index[-1]),
        "price": float(close.iloc[-1]),
        "rsi": current_rsi,
        "bollinger_z": current_z,
        "ema20": current_ema20,
        "ema50": current_ema50,
        "ema20_50_trend": _trend_label(current_ema20, current_ema50),
        "realised_vol": realized,
        "stated_vol": stated_volatility,
        "vol_ratio": ratio,
        "rolling_autocorrelation": current_autocorrelation,
        "hurst": current_hurst,
        "variance_ratio": current_variance_ratio,
        "flags": tuple(active_flags),
    }
    calibration_features = [
        _feature_flag_series(calibration_frame, definition)
        for calibration_frame in calibration_segments
    ]
    for flag_name in FLAG_NAMES:
        calibration_flags: list[pd.Series[float]] = []
        calibration_close: list[pd.Series[float]] = []
        for calibration_frame, features in zip(
            calibration_segments,
            calibration_features,
            strict=True,
        ):
            flag_values = features[flag_name].astype(float).copy()
            if not flag_values.empty:
                flag_values.iloc[-1] = 0.0
            calibration_flags.append(flag_values)
            calibration_close.append(calibration_frame["close"].astype(float))
        _add_reliability(
            row,
            flag_name,
            pd.concat(calibration_flags) if calibration_flags else pd.Series(dtype=float),
            pd.concat(calibration_close) if calibration_close else pd.Series(dtype=float),
            permutations=permutations,
            seed=_flag_seed(seed, symbol, flag_name),
        )
    return row


def _make_flag_series(
    rsi_values: pd.Series[float],
    z_values: pd.Series[float],
    realized_values: pd.Series[float],
    stated_volatility: float,
    hurst_values: pd.Series[float],
    variance_ratio_values: pd.Series[float],
) -> dict[str, pd.Series[bool]]:
    return {
        "overbought": (rsi_values >= 70.0).where(rsi_values.notna()),
        "stretched": (z_values.abs() >= 2.0).where(z_values.notna()),
        "vol_mismatch": (
            (realized_values / stated_volatility > 1.5)
            | (realized_values / stated_volatility < 0.5)
        ).where(realized_values.notna()),
        "trending_regime": ((hurst_values > 0.55) & (variance_ratio_values > 1.0)).where(
            hurst_values.notna() & variance_ratio_values.notna()
        ),
        "mean_reverting_regime": ((hurst_values < 0.45) & (variance_ratio_values < 1.0)).where(
            hurst_values.notna() & variance_ratio_values.notna()
        ),
    }


def _feature_flag_series(
    frame: pd.DataFrame,
    definition: SymbolDefinition,
) -> dict[str, pd.Series[bool]]:
    close = frame["close"]
    realized_values = _realized_volatility_actual_time(close, frame.index, window=20)
    log_returns = pd.Series(close.map(log).diff(), index=close.index, dtype=float)
    return _make_flag_series(
        wilder_rsi(close, period=14),
        bollinger_zscore(close, window=20),
        realized_values,
        definition.stated_vol_pct / 100.0,
        hurst_rs(log_returns, window=64),
        rolling_variance_ratio(log_returns, window=64, horizon=2),
    )


def _realized_volatility_actual_time(
    close: pd.Series[float],
    epochs: pd.Index,
    *,
    window: int,
) -> pd.Series[float]:
    log_returns = pd.Series(close.map(log).diff().to_numpy(), index=epochs)
    elapsed_seconds = pd.Series(epochs.to_numpy(dtype=float), index=epochs).diff()
    annualized_variance = (
        log_returns.pow(2).rolling(window=window, min_periods=window).sum()
        * SECONDS_PER_YEAR
        / elapsed_seconds.rolling(window=window, min_periods=window).sum()
    )
    result = annualized_variance.pow(0.5)
    result.name = "realized_volatility"
    return result


def _add_reliability(
    row: dict[str, object],
    flag_name: str,
    signal: pd.Series[float],
    close: pd.Series[float],
    *,
    permutations: int,
    seed: int,
) -> None:
    evaluable_events = int(signal.iloc[:-RELIABILITY_HORIZON].fillna(0).sum())
    columns = (
        f"{flag_name}_edge_calibration",
        f"{flag_name}_p_value_unadjusted_exploratory",
        f"{flag_name}_p_value_adjusted",
        f"{flag_name}_hit_rate",
        f"{flag_name}_observations",
    )
    if evaluable_events < MIN_RELIABILITY_OBSERVATIONS or len(signal) < 2:
        row[columns[0]] = nan
        row[columns[1]] = nan
        row[columns[2]] = nan
        row[columns[3]] = nan
        row[columns[4]] = evaluable_events
        return
    evaluation = evaluate_signal(
        signal,
        close,
        horizons=(RELIABILITY_HORIZON,),
        permutations=permutations,
        seed=seed,
    )
    result = evaluation.horizons[0]
    row[columns[0]] = result.mean_signed_forward_return
    row[columns[1]] = result.permutation_p_value
    row[columns[2]] = nan
    row[columns[3]] = result.hit_rate
    row[columns[4]] = result.observations


def _flag_seed(seed: int, symbol: str, flag: str) -> int:
    return seed + sum(
        (position + 1) * ord(character) for position, character in enumerate(symbol + flag)
    )


def _latest_float(values: pd.Series[float]) -> float:
    value = values.iloc[-1]
    return float(value) if pd.notna(value) else nan


def _latest_bool(values: pd.Series[bool]) -> bool:
    value = values.iloc[-1]
    return bool(value) if pd.notna(value) else False


def _trend_label(ema20_value: float, ema50_value: float) -> str:
    if not isfinite(ema20_value) or not isfinite(ema50_value):
        return "insufficient_data"
    if ema20_value > ema50_value:
        return "ema20_above_ema50"
    if ema20_value < ema50_value:
        return "ema20_below_ema50"
    return "ema20_equal_ema50"


def _symbol_definition(symbol: str) -> SymbolDefinition:
    try:
        return SYMBOLS[symbol]
    except KeyError as exc:
        raise ValueError(f"Unsupported volatility symbol: {symbol}") from exc
