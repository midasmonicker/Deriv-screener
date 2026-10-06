"""Causal, index-preserving technical and statistical features."""

from __future__ import annotations

from math import log, sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def wilder_rsi(close: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Return Wilder RSI, with the first value after ``period`` changes."""
    _validate_period(period, "period")
    delta = close.astype(float).diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = _wilder_smooth(gains, period, seed_index=period)
    average_loss = _wilder_smooth(losses, period, seed_index=period)
    relative_strength = average_gain / average_loss
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0.0)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50.0)
    return _aligned(rsi, close.index, "rsi")


def ema(values: pd.Series[float], period: int = 14) -> pd.Series[float]:
    """Return the causal exponential moving average with an ``period`` warm-up."""
    _validate_period(period, "period")
    result = values.astype(float).ewm(span=period, adjust=False, min_periods=period).mean()
    return _aligned(result, values.index, "ema")


def atr(
    high: pd.Series[float],
    low: pd.Series[float],
    close: pd.Series[float],
    period: int = 14,
) -> pd.Series[float]:
    """Return Wilder-smoothed average true range."""
    _validate_period(period, "period")
    _require_matching_indexes(high, low, close)
    high_values = high.astype(float)
    low_values = low.astype(float)
    close_values = close.astype(float)
    previous_close = close_values.shift(1)
    true_range = pd.concat(
        [
            high_values - low_values,
            (high_values - previous_close).abs(),
            (low_values - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result = _wilder_smooth(true_range, period, seed_index=period - 1)
    return _aligned(result, close.index, "atr")


def bollinger_zscore(values: pd.Series[float], window: int = 20) -> pd.Series[float]:
    """Return rolling z-score using the population standard deviation."""
    _validate_period(window, "window")
    numeric = values.astype(float)
    mean = numeric.rolling(window=window, min_periods=window).mean()
    standard_deviation = numeric.rolling(window=window, min_periods=window).std(ddof=0)
    result = ((numeric - mean) / standard_deviation).where(standard_deviation > 0)
    return _aligned(result, values.index, "bollinger_zscore")


def _periods_per_year(granularity_seconds: int) -> float:
    """Return annual sampling periods for a UTC 365-day year."""
    if granularity_seconds <= 0:
        raise ValueError("granularity_seconds must be greater than zero")
    return SECONDS_PER_YEAR / granularity_seconds


def annualize_volatility(
    volatility: pd.Series[float],
    granularity_seconds: int,
) -> pd.Series[float]:
    """Annualize per-period volatility using 365*24*3600/granularity periods."""
    result = volatility.astype(float) * sqrt(_periods_per_year(granularity_seconds))
    return _aligned(result, volatility.index, "annualized_volatility")


def realized_volatility(
    close: pd.Series[float],
    window: int = 20,
    granularity_seconds: int = 1,
) -> pd.Series[float]:
    """Return annualized close-to-close log-return volatility."""
    _validate_period(window, "window")
    log_close = pd.Series(np.log(close.astype(float).to_numpy()), index=close.index)
    log_returns = log_close.diff()
    period_volatility = log_returns.rolling(window=window, min_periods=window).std(ddof=1)
    result = annualize_volatility(period_volatility, granularity_seconds)
    return _aligned(result, close.index, "realized_volatility")


def parkinson_volatility(
    high: pd.Series[float],
    low: pd.Series[float],
    window: int = 20,
    granularity_seconds: int = 1,
) -> pd.Series[float]:
    """Return annualized Parkinson range-based volatility."""
    _validate_period(window, "window")
    _require_matching_indexes(high, low)
    high_values = high.astype(float)
    low_values = low.astype(float)
    if ((high_values <= 0) | (low_values <= 0)).any():
        raise ValueError("Parkinson volatility requires positive high and low prices")
    log_range = pd.Series(
        np.log(high_values.to_numpy() / low_values.to_numpy()),
        index=high.index,
    )
    squared_log_range = log_range.pow(2)
    period_variance = squared_log_range.rolling(window=window, min_periods=window).mean()
    period_volatility = np.sqrt(period_variance / (4 * log(2)))
    result = annualize_volatility(period_volatility, granularity_seconds)
    return _aligned(result, high.index, "parkinson_volatility")


def rolling_autocorrelation(
    values: pd.Series[float],
    window: int = 50,
    lag: int = 1,
) -> pd.Series[float]:
    """Return rolling Pearson autocorrelation at one lag from 1 through 10."""
    _validate_period(window, "window")
    if not 1 <= lag <= 10:
        raise ValueError("lag must be between 1 and 10")
    if window <= lag:
        raise ValueError("window must be greater than lag")

    def calculate(sample: FloatArray) -> float:
        left = sample[:-lag]
        right = sample[lag:]
        if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    result = (
        values.astype(float)
        .rolling(window=window, min_periods=window)
        .apply(
            calculate,
            raw=True,
        )
    )
    return _aligned(result, values.index, "rolling_autocorrelation")


def hurst_rs(values: pd.Series[float], window: int = 100) -> pd.Series[float]:
    """Return the rolling Hurst exponent estimated by rescaled-range analysis."""
    _validate_period(window, "window", minimum=16)
    result = (
        values.astype(float)
        .rolling(window=window, min_periods=window)
        .apply(
            _rs_hurst,
            raw=True,
        )
    )
    return _aligned(result, values.index, "hurst_rs")


def hurst_variance_ratio(
    values: pd.Series[float],
    window: int = 100,
    horizon: int = 2,
) -> pd.Series[float]:
    """Return rolling Hurst estimates from variance scaling across ``horizon``."""
    _validate_period(window, "window", minimum=3)
    if not 2 <= horizon < window:
        raise ValueError("horizon must be at least 2 and less than window")

    def calculate(sample: FloatArray) -> float:
        variance_one = float(np.var(sample, ddof=1))
        if variance_one <= 0:
            return float("nan")
        aggregated = np.convolve(sample, np.ones(horizon), mode="valid")
        variance_horizon = float(np.var(aggregated, ddof=1))
        if variance_horizon <= 0:
            return float("nan")
        return log(variance_horizon / variance_one) / (2 * log(horizon))

    result = (
        values.astype(float)
        .rolling(window=window, min_periods=window)
        .apply(
            calculate,
            raw=True,
        )
    )
    return _aligned(result, values.index, "hurst_variance_ratio")


def rolling_variance_ratio(
    returns: pd.Series[float],
    window: int = 100,
    horizon: int = 2,
) -> pd.Series[float]:
    """Return rolling Lo-MacKinlay overlapping variance ratios."""
    _validate_period(window, "window", minimum=3)
    if not 2 <= horizon < window:
        raise ValueError("horizon must be at least 2 and less than window")

    def calculate(sample: FloatArray) -> float:
        sample_size = sample.size
        demeaned = sample - sample.mean()
        variance_one = float(np.dot(demeaned, demeaned) / (sample_size - 1))
        if variance_one <= 0:
            return float("nan")
        aggregated = np.convolve(demeaned, np.ones(horizon), mode="valid")
        effective_observations = (
            horizon * (sample_size - horizon + 1) * (1.0 - horizon / sample_size)
        )
        variance_horizon = float(np.dot(aggregated, aggregated) / effective_observations)
        return variance_horizon / variance_one

    result = (
        returns.astype(float)
        .rolling(window=window, min_periods=window)
        .apply(
            calculate,
            raw=True,
        )
    )
    return _aligned(result, returns.index, "variance_ratio")


def return_skew(returns: pd.Series[float], window: int = 20) -> pd.Series[float]:
    """Return rolling sample skewness of input returns."""
    _validate_period(window, "window", minimum=3)
    result = returns.astype(float).rolling(window=window, min_periods=window).skew()
    return _aligned(result, returns.index, "return_skew")


def return_kurtosis(returns: pd.Series[float], window: int = 20) -> pd.Series[float]:
    """Return rolling unbiased Fisher excess kurtosis of input returns."""
    _validate_period(window, "window", minimum=4)
    result = returns.astype(float).rolling(window=window, min_periods=window).kurt()
    return _aligned(result, returns.index, "return_kurtosis")


def _rs_hurst(sample: FloatArray) -> float:
    sample = sample[np.isfinite(sample)]
    sizes = np.asarray(
        [size for size in (8, 16, 32, 64, 128, 256) if size <= sample.size // 2],
        dtype=np.float64,
    )
    if sizes.size < 2:
        return float("nan")

    rescaled_ranges: list[float] = []
    for size_value in sizes:
        size = int(size_value)
        chunk_count = sample.size // size
        chunks = sample[: chunk_count * size].reshape(chunk_count, size)
        centered = chunks - chunks.mean(axis=1, keepdims=True)
        cumulative = centered.cumsum(axis=1)
        ranges = cumulative.max(axis=1) - cumulative.min(axis=1)
        standard_deviations = chunks.std(axis=1, ddof=1)
        valid = standard_deviations > 0
        if not valid.any():
            rescaled_ranges.append(float("nan"))
        else:
            rescaled_ranges.append(float(np.mean(ranges[valid] / standard_deviations[valid])))
    rs = np.asarray(rescaled_ranges)
    valid_rs = np.isfinite(rs) & (rs > 0)
    if valid_rs.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(sizes[valid_rs]), np.log(rs[valid_rs]), 1)[0])


def _validate_period(value: int, name: str, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _wilder_smooth(values: pd.Series[float], period: int, seed_index: int) -> pd.Series[float]:
    seeded = values.astype(float).copy()
    if len(seeded) <= seed_index:
        return pd.Series(np.nan, index=values.index, dtype=float)
    initial_value = values.rolling(window=period, min_periods=period).mean().iloc[seed_index]
    if pd.isna(initial_value):
        return pd.Series(np.nan, index=values.index, dtype=float)
    seeded.iloc[:seed_index] = np.nan
    seeded.iloc[seed_index] = float(initial_value)
    return seeded.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()


def _require_matching_indexes(*series: pd.Series[float]) -> None:
    if not series:
        return
    reference = series[0].index
    if any(not reference.equals(item.index) for item in series[1:]):
        raise ValueError("Input series must have identical indexes")


def _aligned(result: pd.Series[float], index: pd.Index, name: str) -> pd.Series[float]:
    if not result.index.equals(index):
        raise RuntimeError(f"{name} did not preserve its input index")
    result.name = name
    return result
