"""Statistical diagnostics and null-hypothesis tools for synthetic indices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import sqrt

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import binomtest, chi2, norm

from deriv_vol_lab.data.symbols import SYMBOLS, SymbolDefinition

SECONDS_PER_YEAR = 365 * 24 * 60 * 60


class VolatilityWindowResult(BaseModel):
    """A rolling realized-volatility estimate and its confidence interval."""

    symbol: str
    window_end: int | str
    observations: int
    stated_volatility: float
    realized_volatility: float
    confidence_level: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    alert: bool


class VolatilityMonitorReport(BaseModel):
    """Rolling per-symbol volatility checks."""

    confidence_level: float
    windows: list[VolatilityWindowResult]


class TestResult(BaseModel):
    """One named statistical test result with family-wise BH adjustment."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    test: str
    statistic: float
    p_value: float
    adjusted_p_value: float = 1.0
    reject_fdr: bool = False


class ForwardHorizonResult(BaseModel):
    """Forward-return statistics; n_effective excludes ties and overlapping labels."""

    horizon: int
    observations: int
    tie_count: int = 0
    n_effective: int = 0
    mean_forward_return: float
    median_forward_return: float
    standard_deviation: float
    hit_rate: float
    binomial_p_value: float | None
    binomial_is_inferential: bool = True
    permutation_p_value: float
    stationary_bootstrap_p_value: float | None = None
    mean_signed_forward_return: float


class SignalEvaluation(BaseModel):
    """Signal performance at multiple forward horizons."""

    horizons: list[ForwardHorizonResult]
    permutation_count: int
    seed: int


def rolling_volatility_monitor(
    closes: Mapping[str, pd.Series[float]],
    *,
    window: int = 1000,
    confidence_level: float = 0.99,
    sampling_intervals: Mapping[str, int] | None = None,
) -> VolatilityMonitorReport:
    """Compare annualized realized volatility with the stated value per rolling window.

    The confidence interval is the exact chi-square interval for a normally
    distributed sample variance. Repeated windows and symbols create a
    multiple-testing problem: 99% intervals do not provide 99% family-wise
    coverage. Treat alerts as diagnostics or adjust thresholds across the
    symbol/window family before interpreting them as discoveries. Pass
    ``sampling_intervals`` when close prices are sampled more slowly than the
    symbol's native tick cadence, such as candles; values are seconds per
    observed price interval and default to each symbol's registered tick rate.
    """
    _validate_positive_int(window, "window", minimum=2)
    _validate_probability(confidence_level, "confidence_level")
    windows: list[VolatilityWindowResult] = []

    for symbol, close in closes.items():
        definition = _symbol_definition(symbol)
        numeric = _validated_positive_prices(close, "close")
        log_prices = pd.Series(np.log(numeric.to_numpy()), index=numeric.index)
        returns = log_prices.diff().dropna()
        interval = (
            definition.tick_interval_sec
            if sampling_intervals is None
            else sampling_intervals.get(symbol, definition.tick_interval_sec)
        )
        _validate_positive_int(interval, f"sampling_intervals[{symbol}]", minimum=1)
        annualization = sqrt(SECONDS_PER_YEAR / interval)
        degrees_of_freedom = window - 1
        alpha = 1.0 - confidence_level

        for end_position in range(window, len(returns) + 1):
            sample = returns.iloc[end_position - window : end_position].to_numpy(dtype=float)
            sample_variance = float(np.var(sample, ddof=1))
            realized = sqrt(sample_variance) * annualization
            lower = (
                sqrt(
                    degrees_of_freedom
                    * sample_variance
                    / float(chi2.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
                )
                * annualization
            )
            upper = (
                sqrt(
                    degrees_of_freedom
                    * sample_variance
                    / float(chi2.ppf(alpha / 2.0, degrees_of_freedom))
                )
                * annualization
            )
            end_index = returns.index[end_position - 1]
            windows.append(
                VolatilityWindowResult(
                    symbol=symbol,
                    window_end=_index_value(end_index),
                    observations=window,
                    stated_volatility=definition.stated_vol_pct / 100.0,
                    realized_volatility=realized,
                    confidence_level=confidence_level,
                    lower_confidence_bound=lower,
                    upper_confidence_bound=upper,
                    alert=not lower <= definition.stated_vol_pct / 100.0 <= upper,
                )
            )
    return VolatilityMonitorReport(confidence_level=confidence_level, windows=windows)


def ljung_box_test(returns: pd.Series[float], lags: int = 10) -> tuple[float, float]:
    """Return Ljung-Box Q and chi-square p-value for return autocorrelation."""
    _validate_test_sample(returns, lags, "lags")
    values = _finite_values(returns)
    if values.size <= lags:
        raise ValueError("returns must contain more observations than lags")
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return 0.0, 1.0
    autocorrelations = np.asarray(
        [np.dot(centered[lag:], centered[:-lag]) / denominator for lag in range(1, lags + 1)]
    )
    sample_size = values.size
    statistic = (
        sample_size
        * (sample_size + 2)
        * float(np.sum(autocorrelations**2 / (sample_size - np.arange(1, lags + 1))))
    )
    return statistic, float(chi2.sf(statistic, lags))


def runs_test(returns: pd.Series[float]) -> tuple[float, float]:
    """Return continuity-corrected Wald-Wolfowitz runs z-statistic and p-value."""
    values = _finite_values(returns)
    if values.size < 2:
        raise ValueError("runs test requires at least two returns")
    signs = values[values != 0] > 0
    n_positive = int(signs.sum())
    n_negative = int(signs.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return 0.0, 1.0
    run_count = int(1 + np.count_nonzero(signs[1:] != signs[:-1]))
    total = n_positive + n_negative
    expected = 1.0 + 2.0 * n_positive * n_negative / total
    variance = (
        2.0
        * n_positive
        * n_negative
        * (2.0 * n_positive * n_negative - total)
        / (total**2 * (total - 1))
    )
    if variance <= 0:
        return 0.0, 1.0
    difference = abs(run_count - expected)
    corrected = max(difference - 0.5, 0.0)
    statistic = np.sign(run_count - expected) * corrected / sqrt(variance)
    return float(statistic), float(2.0 * norm.sf(abs(statistic)))


def variance_ratio_test(
    returns: pd.Series[float],
    horizon: int = 2,
) -> tuple[float, float]:
    """Return Lo-MacKinlay homoscedastic variance-ratio z-statistic and p-value."""
    _validate_positive_int(horizon, "horizon", minimum=2)
    values = _finite_values(returns)
    sample_size = values.size
    if sample_size <= horizon:
        raise ValueError("returns must contain more observations than horizon")
    mean = float(values.mean())
    demeaned = values - mean
    variance_one = float(np.dot(demeaned, demeaned) / (sample_size - 1))
    if variance_one == 0:
        return 0.0, 1.0
    aggregated = np.convolve(demeaned, np.ones(horizon), mode="valid")
    effective_observations = horizon * (sample_size - horizon + 1) * (1.0 - horizon / sample_size)
    variance_horizon = float(np.dot(aggregated, aggregated) / effective_observations)
    ratio = variance_horizon / variance_one
    asymptotic_variance = 2.0 * (2 * horizon - 1) * (horizon - 1) / (3.0 * horizon * sample_size)
    statistic = (ratio - 1.0) / sqrt(asymptotic_variance)
    return float(statistic), float(2.0 * norm.sf(abs(statistic)))


def arch_lm_test(
    returns: pd.Series[float],
    lags: int = 5,
) -> tuple[float, float]:
    """Return Engle ARCH-LM statistic and asymptotic chi-square p-value."""
    _validate_positive_int(lags, "lags", minimum=1)
    values = _finite_values(returns)
    if values.size <= lags + 1:
        raise ValueError("returns must contain more than lags + 1 observations")
    squared = values**2
    target = squared[lags:]
    design = np.column_stack(
        [
            np.ones(target.size),
            *(squared[lags - lag : values.size - lag] for lag in range(1, lags + 1)),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    total_sum_squares = float(np.sum((target - target.mean()) ** 2))
    if total_sum_squares == 0:
        return 0.0, 1.0
    r_squared = 1.0 - float(np.dot(residuals, residuals)) / total_sum_squares
    statistic = max(0.0, target.size * r_squared)
    return statistic, float(chi2.sf(statistic, lags))


def randomness_test_battery(
    returns_by_symbol: Mapping[str, pd.Series[float]],
    *,
    lags: int = 10,
    variance_ratio_horizon: int = 2,
    arch_lags: int = 5,
    fdr_alpha: float = 0.05,
) -> list[TestResult]:
    """Run four return-randomness tests and apply BH correction across all results.

    The battery pools every symbol/test p-value into one Benjamini-Hochberg
    family. This addresses, but does not eliminate, the multiple-testing problem:
    dependence among symbols/tests and repeated analysis choices can still
    increase false discoveries. The test assumptions (especially Gaussian
    asymptotics and Lo-MacKinlay's homoscedastic variance) also matter.
    """
    _validate_probability(fdr_alpha, "fdr_alpha")
    results: list[TestResult] = []
    tests: tuple[tuple[str, Callable[[pd.Series[float]], tuple[float, float]]], ...] = (
        ("ljung_box", lambda series: ljung_box_test(series, lags)),
        ("runs", runs_test),
        ("variance_ratio", lambda series: variance_ratio_test(series, variance_ratio_horizon)),
        ("arch_lm", lambda series: arch_lm_test(series, arch_lags)),
    )
    for symbol, returns in returns_by_symbol.items():
        _symbol_definition(symbol)
        for test_name, test_function in tests:
            statistic, p_value = test_function(returns)
            results.append(
                TestResult(symbol=symbol, test=test_name, statistic=statistic, p_value=p_value)
            )

    adjusted = benjamini_hochberg([result.p_value for result in results])
    return [
        result.model_copy(
            update={
                "adjusted_p_value": adjusted_p,
                "reject_fdr": adjusted_p <= fdr_alpha,
            }
        )
        for result, adjusted_p in zip(results, adjusted, strict=True)
    ]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values in input order."""
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(np.asarray(p_values))
    sorted_values = np.asarray(p_values, dtype=float)[order]
    adjusted_sorted = np.minimum.accumulate(
        (sorted_values * count / np.arange(1, count + 1))[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def simulate_driftless_gbm(
    symbol: str,
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    start_price: float = 100.0,
) -> np.ndarray:
    """Generate seeded, driftless GBM paths matching registry volatility and tick rate.

    The arithmetic drift is zero: log-price increments use ``-sigma²*dt/2``.
    Every random simulation must use an explicit seed; this function requires
    one and uses NumPy's local generator without mutating global RNG state.
    """
    definition = _symbol_definition(symbol)
    _validate_positive_int(n_paths, "n_paths", minimum=1)
    _validate_positive_int(n_steps, "n_steps", minimum=1)
    if start_price <= 0:
        raise ValueError("start_price must be greater than zero")
    annual_volatility = definition.stated_vol_pct / 100.0
    dt = definition.tick_interval_sec / SECONDS_PER_YEAR
    generator = np.random.default_rng(seed)
    shocks = generator.standard_normal((n_paths, n_steps))
    log_increments = -0.5 * annual_volatility**2 * dt + annual_volatility * sqrt(dt) * shocks
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = start_price
    paths[:, 1:] = start_price * np.exp(np.cumsum(log_increments, axis=1))
    return paths


def evaluate_signal(
    signal: pd.Series[float],
    close: pd.Series[float],
    *,
    horizons: Sequence[int] = (1, 5, 10),
    permutations: int = 999,
    stationary_bootstrap: bool = False,
    bootstrap_count: int = 999,
    expected_block_length: float | None = None,
    seed: int,
) -> SignalEvaluation:
    """Evaluate signal returns with tie-aware and serial-dependence-aware inference.

    Circular shifts test the alignment of the signal to outcomes while
    preserving signal order. For horizons greater than one, inference uses
    decisions at positions ``0, h, 2h, ...`` to avoid overlapping outcomes;
    the reported binomial p-value is then descriptive only and is marked
    non-inferential. Ties are excluded from binomial trials and counted
    separately. The optional stationary bootstrap resamples centered,
    flat-inclusive signed returns using Politis-Romano stationary blocks.

    Multiple horizons or signals create a multiple-testing problem; adjust
    p-values across the full pre-specified family before claiming an edge.
    """
    if not signal.index.equals(close.index):
        raise ValueError("signal and close must have identical indexes")
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be non-empty positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    _validate_positive_int(permutations, "permutations", minimum=1)
    if stationary_bootstrap:
        _validate_positive_int(bootstrap_count, "bootstrap_count", minimum=1)
        if expected_block_length is not None and (
            not np.isfinite(expected_block_length) or expected_block_length < 1
        ):
            raise ValueError("expected_block_length must be finite and at least one")
    prices = _validated_positive_prices(close, "close")
    directions = np.sign(signal.astype(float).to_numpy())
    generator = np.random.default_rng(seed)
    results: list[ForwardHorizonResult] = []

    for horizon in horizons:
        future = prices.shift(-horizon)
        forward_returns = future / prices - 1.0
        returns = forward_returns.to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(directions) & np.isfinite(returns)
        active_mask = valid_mask & (directions != 0)
        selected_returns = returns[active_mask]
        selected_directions = directions[active_mask]
        observations = selected_returns.size
        if observations == 0:
            raise ValueError(f"No non-zero signal observations for horizon {horizon}")
        signed_returns = selected_directions * selected_returns

        inference_positions = np.arange(0, len(directions), horizon)
        inference_valid = valid_mask[inference_positions]
        inference_returns = returns[inference_positions][inference_valid]
        inference_directions = directions[inference_positions][inference_valid]
        inference_active = inference_directions != 0
        inference_signed = (
            inference_directions[inference_active] * inference_returns[inference_active]
        )
        inference_ties = inference_signed == 0
        tie_count = int(np.count_nonzero(signed_returns == 0))
        non_tied = inference_signed[~inference_ties]
        n_effective = int(non_tied.size)
        if inference_signed.size == 0:
            raise ValueError(f"No non-zero signal observations for horizon {horizon}")
        hit_count = int(np.count_nonzero(non_tied > 0))
        binomial_p = (
            float(binomtest(hit_count, non_tied.size, p=0.5, alternative="two-sided").pvalue)
            if non_tied.size
            else 1.0
        )
        observed_statistic = float(inference_signed.mean())
        valid_returns = returns[inference_positions][inference_valid]
        valid_directions = directions[inference_positions][inference_valid]
        if valid_directions.size < 2:
            raise ValueError("At least two non-overlapping valid decisions are required")
        shifted_statistics = np.empty(permutations, dtype=float)
        for permutation in range(permutations):
            offset = int(generator.integers(1, valid_directions.size))
            shifted_directions = np.roll(valid_directions, offset)
            active = shifted_directions != 0
            shifted_statistics[permutation] = (
                float(np.mean(shifted_directions[active] * valid_returns[active]))
                if active.any()
                else 0.0
            )
        permutation_p = float(
            (1 + np.count_nonzero(np.abs(shifted_statistics) >= abs(observed_statistic)))
            / (permutations + 1)
        )
        bootstrap_p: float | None = None
        if stationary_bootstrap:
            flat_signed = np.zeros(valid_returns.size, dtype=np.float64)
            flat_signed[valid_directions != 0] = (
                valid_directions[valid_directions != 0] * valid_returns[valid_directions != 0]
            )
            block_length = (
                float(expected_block_length)
                if expected_block_length is not None
                else max(1.0, float(horizon))
            )
            bootstrap_p = _stationary_bootstrap_p_value(
                flat_signed,
                bootstrap_count=bootstrap_count,
                expected_block_length=block_length,
                generator=generator,
            )
        results.append(
            ForwardHorizonResult(
                horizon=horizon,
                observations=observations,
                tie_count=tie_count,
                n_effective=n_effective,
                mean_forward_return=float(selected_returns.mean()),
                median_forward_return=float(np.median(selected_returns)),
                standard_deviation=float(selected_returns.std(ddof=1)) if observations > 1 else 0.0,
                hit_rate=hit_count / non_tied.size if non_tied.size else 0.0,
                binomial_p_value=binomial_p,
                binomial_is_inferential=horizon == 1,
                permutation_p_value=permutation_p,
                stationary_bootstrap_p_value=bootstrap_p,
                mean_signed_forward_return=observed_statistic,
            )
        )
    return SignalEvaluation(horizons=results, permutation_count=permutations, seed=seed)


def _stationary_bootstrap_p_value(
    signed_returns: np.ndarray,
    *,
    bootstrap_count: int,
    expected_block_length: float,
    generator: np.random.Generator,
) -> float:
    """Test zero mean using a centered circular Politis-Romano bootstrap."""
    sample_size = signed_returns.size
    if sample_size < 2:
        raise ValueError("Stationary bootstrap requires at least two decisions")
    observed = float(signed_returns.mean())
    centered = signed_returns - observed
    restart_probability = 1.0 / expected_block_length
    bootstrap_means = np.empty(bootstrap_count, dtype=np.float64)
    for bootstrap_index in range(bootstrap_count):
        indices = np.empty(sample_size, dtype=np.int64)
        indices[0] = int(generator.integers(sample_size))
        for position in range(1, sample_size):
            if generator.random() < restart_probability:
                indices[position] = int(generator.integers(sample_size))
            else:
                indices[position] = (indices[position - 1] + 1) % sample_size
        bootstrap_means[bootstrap_index] = float(centered[indices].mean())
    return float(
        (1 + np.count_nonzero(np.abs(bootstrap_means) >= abs(observed))) / (bootstrap_count + 1)
    )


def _validate_test_sample(series: pd.Series[float], lag: int, name: str) -> None:
    _validate_positive_int(lag, name, minimum=1)
    if _finite_values(series).size < 3:
        raise ValueError("at least three finite returns are required")


def _finite_values(series: pd.Series[float]) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("series must contain finite observations")
    return values


def _validated_positive_prices(series: pd.Series[float], name: str) -> pd.Series[float]:
    values = series.astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError(f"{name} prices must be finite and greater than zero")
    return values


def _symbol_definition(symbol: str) -> SymbolDefinition:
    try:
        return SYMBOLS[symbol]
    except KeyError as exc:
        raise ValueError(f"Unsupported volatility symbol: {symbol}") from exc


def _validate_positive_int(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _index_value(value: object) -> int | str:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)
