"""Offline, descriptive tests of screener flags against stored 60-second candles."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from math import exp, isfinite, sqrt
from os import cpu_count
from pathlib import Path
from typing import Literal, NamedTuple

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError
from scipy.stats import binom, binomtest, norm

from deriv_vol_lab.backtest.engine import ProductSpec, TradeResult, forward_trade_returns
from deriv_vol_lab.data import SYMBOLS, Candle, DuckDBStorage
from deriv_vol_lab.features.screener import (
    FLAG_DEFINITIONS,
    FLAG_NAMES,
    causal_flag_series,
)
from deriv_vol_lab.settings import Settings
from deriv_vol_lab.stats.analysis import benjamini_hochberg

GRANULARITY_SECONDS = 60
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
DEFAULT_HORIZONS = (1, 5, 15, 60)
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 12345
DEFAULT_PERMUTATIONS = 2_000
DEFAULT_BOOTSTRAP_REPLICATIONS = 250
DEFAULT_NULL_REPLICATIONS = 20
MINIMUM_CANDLES = 2_000
DIRECTION = Literal["long", "short"]
SPLIT = Literal["discovery", "confirmation"]
DIRECTIONS: tuple[DIRECTION, ...] = ("long", "short")


class EdgeTestRow(BaseModel):
    """One symbol- or pool-level hypothesis for a split and horizon."""

    model_config = ConfigDict(frozen=True)

    scope: Literal["symbol", "pooled"]
    symbol: str
    flag: str
    horizon: int
    direction: DIRECTION
    split: SPLIT
    n_signals: int
    n_effective: int
    n_non_tied: int
    gross_mean_bps: float | None
    net_mean_bps: float | None
    hit_rate: float | None
    binomial_p: float
    circular_shift_p: float
    stouffer_z: float | None = None
    bootstrap_low_bps: float | None = None
    bootstrap_high_bps: float | None = None
    minimum_detectable_hit_deviation_pct: float | None = None
    adjusted_p: float = 1.0
    binomial_adjusted_p: float = 1.0
    bonferroni_adjusted_p: float = 1.0
    confirmed: bool = False


class EdgeAnalysis(BaseModel):
    """Full, multiple-testing-adjusted analysis for one dataset."""

    rows: list[EdgeTestRow]
    test_count: int
    bonferroni_count: int
    alpha: float


class NullControlResult(BaseModel):
    """False-positive rates from repeated driftless GBM runs."""

    replications: int
    tests_per_replication: int
    raw_significant_fraction: float
    bh_surviving_fraction: float
    confirmed_edges: int


class _CandidateOutcomes(NamedTuple):
    decision_epochs: np.ndarray
    gross_returns_bps: np.ndarray
    net_returns_bps: np.ndarray


@dataclass(frozen=True, slots=True)
class _ComputedTest:
    row: EdgeTestRow
    gross_returns_bps: np.ndarray
    net_returns_bps: np.ndarray
    bootstrap_groups: list[np.ndarray]
    shift_groups: list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class _NullWorkerConfig:
    epochs_by_symbol: dict[str, tuple[int, ...]]
    horizons: tuple[int, ...]
    alpha: float
    seed: int
    permutations: int
    bootstrap_reps: int
    product: ProductSpec | None


class _NullReplicationSummary(NamedTuple):
    replication: int
    elapsed_seconds: float
    test_count: int
    raw_significant: int
    bh_surviving: int
    confirmed_edges: int


_NULL_WORKER_CONFIG: _NullWorkerConfig | None = None


def _progress(message: str) -> None:
    print(f"[flag-edge] {message}", file=sys.stderr, flush=True)


def _default_jobs() -> int:
    return max(1, (cpu_count() or 1) - 1)


def _estimate_null_eta_seconds(
    *,
    replications: int,
    completed_count: int,
    worker_count: int,
    average_replication_seconds: float,
    active_replication_seconds: float,
) -> float:
    remaining_work = (
        average_replication_seconds * (replications - completed_count) - active_replication_seconds
    )
    return max(0.0, remaining_work / worker_count)


def _default_product() -> ProductSpec:
    return ProductSpec(
        contract_type="multiplier",
        stake=1.0,
        multiplier=1.0,
        stop_out_fraction=1.0,
    )


def hit_rate(outcomes: Sequence[float]) -> float | None:
    """Return the positive fraction after excluding exact ties and non-finite values."""
    non_tied = [float(value) for value in outcomes if isfinite(float(value)) and value != 0.0]
    if not non_tied:
        return None
    return sum(value > 0.0 for value in non_tied) / len(non_tied)


def binomial_p_value(outcomes: Sequence[float]) -> float:
    """One-sided exact binomial p-value for a net-return hit rate above 50%."""
    non_tied = [float(value) for value in outcomes if isfinite(float(value)) and value != 0.0]
    if not non_tied:
        return 1.0
    wins = sum(value > 0.0 for value in non_tied)
    return float(binomtest(wins, len(non_tied), 0.5, alternative="greater").pvalue)


def _all_circular_shift_sums(signal: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    signal_values = np.asarray(signal, dtype=float)
    outcome_values = np.asarray(outcomes, dtype=float)
    if signal_values.size != outcome_values.size:
        raise ValueError("signal and outcome lengths must match")
    if signal_values.size == 0:
        return np.zeros(0, dtype=float)
    correlation = np.fft.ifft(
        np.conjugate(np.fft.fft(signal_values)) * np.fft.fft(outcome_values)
    ).real
    return correlation


def circular_shift_p_value(
    signal: Sequence[bool] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    *,
    shifts: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Seeded one-sided circular-shift p-value, with a plus-one correction."""
    signal_values = np.asarray(signal, dtype=bool)
    outcome_values = np.asarray(outcomes, dtype=float)
    if signal_values.ndim != 1 or outcome_values.ndim != 1:
        raise ValueError("signal and outcomes must be one-dimensional")
    if signal_values.size != outcome_values.size:
        raise ValueError("signal and outcome lengths must match")
    if shifts <= 0:
        raise ValueError("shifts must be greater than zero")
    if not np.isfinite(outcome_values).all():
        raise ValueError("outcomes must be finite")
    signal_count = int(signal_values.sum())
    if signal_count == 0 or signal_values.size < 2:
        return 1.0

    shift_sums = _all_circular_shift_sums(signal_values, outcome_values)
    observed = float(shift_sums[0] / signal_count)
    available_offsets = np.arange(1, signal_values.size)
    generator = np.random.default_rng(seed)
    if shifts <= available_offsets.size:
        offsets = generator.choice(available_offsets, size=shifts, replace=False)
    else:
        offsets = generator.choice(available_offsets, size=shifts, replace=True)
    permuted = shift_sums[offsets] / signal_count
    return float((1 + np.count_nonzero(permuted >= observed)) / (shifts + 1))


def _segmented_circular_shift_p_value(
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    shifts: int,
    seed: int,
) -> float:
    nonempty = [
        (np.asarray(signal, dtype=bool), np.asarray(outcomes, dtype=float))
        for signal, outcomes in groups
        if len(signal) > 0
    ]
    signal_count = sum(int(signal.sum()) for signal, _ in nonempty)
    if signal_count == 0 or shifts <= 0:
        return 1.0
    observed_sum = sum(
        float(np.dot(signal.astype(float), outcomes)) for signal, outcomes in nonempty
    )
    observed_mean = observed_sum / signal_count
    generator = np.random.default_rng(seed)
    permuted_sums = np.zeros(shifts, dtype=float)
    for signal, outcomes in nonempty:
        shift_sums = _all_circular_shift_sums(signal, outcomes)
        if len(signal) < 2:
            permuted_sums += shift_sums[0]
        else:
            offsets = generator.integers(1, len(signal), size=shifts)
            permuted_sums += shift_sums[offsets]
    return float(
        (1 + np.count_nonzero(permuted_sums / signal_count >= observed_mean)) / (shifts + 1)
    )


def _stationary_sample(values: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    sample_size = len(values)
    block_length = max(1.0, sqrt(sample_size))
    restart_probability = 1.0 / block_length
    sample = np.empty(sample_size, dtype=float)
    filled = 0
    while filled < sample_size:
        start = int(generator.integers(0, sample_size))
        block_size = int(generator.geometric(restart_probability))
        copied = min(block_size, sample_size - filled)
        positions = (start + np.arange(copied)) % sample_size
        sample[filled : filled + copied] = values[positions]
        filled += copied
    return sample


def _stationary_bootstrap_ci_groups(
    groups: Sequence[np.ndarray],
    *,
    seed: int,
    replications: int,
    confidence_level: float = 0.95,
) -> tuple[float, float] | None:
    values_by_group = [np.asarray(group, dtype=float) for group in groups if len(group) > 0]
    if not values_by_group:
        return None
    if replications <= 0:
        raise ValueError("replications must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    total_count = sum(len(group) for group in values_by_group)
    generator = np.random.default_rng(seed)
    means = np.empty(replications, dtype=float)
    for replication in range(replications):
        total = 0.0
        for group in values_by_group:
            total += float(_stationary_sample(group, generator).sum())
        means[replication] = total / total_count
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def stationary_bootstrap_ci(
    outcomes: Sequence[float],
    *,
    seed: int,
    replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = 0.95,
) -> tuple[float, float] | None:
    """Stationary-bootstrap percentile confidence interval for the mean."""
    values = np.asarray(outcomes, dtype=float)
    values = values[np.isfinite(values)]
    return _stationary_bootstrap_ci_groups(
        [values],
        seed=seed,
        replications=replications,
        confidence_level=confidence_level,
    )


def minimum_detectable_hit_deviation(
    n_effective: int,
    *,
    alpha: float,
    power: float = 0.8,
) -> float | None:
    """Smallest one-sided hit-rate increase, in percentage points, at target power."""
    if n_effective < 0:
        raise ValueError("n_effective cannot be negative")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be between zero and one")
    if n_effective == 0:
        return None
    lower_count = 0
    upper_count = n_effective + 1
    while lower_count < upper_count:
        candidate = (lower_count + upper_count) // 2
        if float(binom.sf(candidate - 1, n_effective, 0.5)) <= alpha:
            upper_count = candidate
        else:
            lower_count = candidate + 1
    critical_wins = lower_count
    if critical_wins > n_effective:
        return None
    lower_rate = 0.5
    upper_rate = 1.0
    for _ in range(60):
        candidate_rate = (lower_rate + upper_rate) / 2.0
        candidate_power = float(binom.sf(critical_wins - 1, n_effective, candidate_rate))
        if candidate_power >= power:
            upper_rate = candidate_rate
        else:
            lower_rate = candidate_rate
    return (upper_rate - 0.5) * 100.0


def _weighted_stouffer(
    p_values: Sequence[float], weights: Sequence[float]
) -> tuple[float | None, float]:
    if len(p_values) != len(weights):
        raise ValueError("p-value and weight counts must match")
    usable = [(p, weight) for p, weight in zip(p_values, weights, strict=True) if weight > 0]
    if not usable:
        return None, 1.0
    clipped = np.clip([p for p, _ in usable], 1e-15, 1.0 - 1e-15)
    weight_values = np.asarray([weight for _, weight in usable], dtype=float)
    z_values = norm.isf(clipped)
    combined_z = float(
        np.dot(weight_values, z_values) / sqrt(float(np.dot(weight_values, weight_values)))
    )
    return combined_z, float(norm.sf(combined_z))


def assert_truncation_invariance(
    candles: Sequence[Candle],
    flag_builder: Callable[[Sequence[Candle]], Mapping[str, pd.Series[bool]]],
) -> None:
    """Raise when a flag value at t changes when input is truncated at t."""
    if not candles:
        return
    full_flags = flag_builder(candles)
    for position, candle in enumerate(candles):
        prefix_flags = flag_builder(candles[: position + 1])
        for name, series in full_flags.items():
            if name not in prefix_flags or position >= len(prefix_flags[name]):
                raise ValueError(f"Flag {name} is not causal at epoch {candle.epoch}")
            full_value = series.iloc[position]
            prefix_value = prefix_flags[name].iloc[-1]
            full_bool = bool(full_value) if pd.notna(full_value) else False
            prefix_bool = bool(prefix_value) if pd.notna(prefix_value) else False
            if full_bool != prefix_bool:
                raise ValueError(f"Flag {name} is not causal at epoch {candle.epoch}")


def _contiguous_segments(candles: Sequence[Candle]) -> list[list[Candle]]:
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


def build_forward_trades(
    candles: Sequence[Candle],
    flags: pd.Series[bool],
    *,
    horizon: int,
    direction: DIRECTION,
    non_overlapping: bool = True,
    product: ProductSpec | None = None,
) -> list[TradeResult]:
    """Build shared-backtester trades for a flag, honoring horizon stride and gaps."""
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero")
    epochs = pd.Index([candle.epoch for candle in candles])
    if not flags.index.equals(epochs):
        raise ValueError("flag index must equal candle epochs in chronological order")
    stride = horizon if non_overlapping else 1
    direction_value = 1.0 if direction == "long" else -1.0
    values = np.zeros(len(candles), dtype=float)
    for position, active in enumerate(flags.fillna(False).to_numpy(dtype=bool)):
        if active and position % stride == 0:
            values[position] = direction_value
    signal = pd.Series(values, index=epochs, dtype=float)
    return forward_trade_returns(candles, signal, product or _default_product(), horizon=horizon)


def make_gbm_candles(
    symbol: str,
    *,
    epochs: Sequence[int],
    stated_volatility: float,
    granularity: int,
    seed: int,
) -> list[Candle]:
    """Generate a driftless-GBM OHLC path on the supplied UTC candle epochs."""
    if stated_volatility <= 0.0 or granularity <= 0:
        raise ValueError("stated_volatility and granularity must be positive")
    generator = np.random.default_rng(seed)
    result: list[Candle] = []
    price = 100.0
    previous_epoch: int | None = None
    for epoch in epochs:
        elapsed = (
            granularity if previous_epoch is None else max(granularity, epoch - previous_epoch)
        )
        sigma = stated_volatility * sqrt(elapsed / SECONDS_PER_YEAR)
        opened = price
        log_change = float(generator.normal(-0.5 * sigma**2, sigma))
        closed = opened * exp(log_change)
        result.append(
            Candle(
                symbol=symbol,
                epoch=epoch,
                open=Decimal(str(opened)),
                high=Decimal(str(max(opened, closed))),
                low=Decimal(str(min(opened, closed))),
                close=Decimal(str(closed)),
                granularity=granularity,
            )
        )
        price = closed
        previous_epoch = epoch
    return result


def _candidate_outcomes(
    candles: Sequence[Candle],
    *,
    horizon: int,
    direction: DIRECTION,
    product: ProductSpec,
) -> list[_CandidateOutcomes]:
    result: list[_CandidateOutcomes] = []
    direction_value = 1.0 if direction == "long" else -1.0
    for segment in _contiguous_segments(candles):
        candidate_positions = list(range(0, max(0, len(segment) - horizon), horizon))
        if not candidate_positions:
            continue
        epochs = pd.Index([candle.epoch for candle in segment])
        signal_values = np.zeros(len(segment), dtype=float)
        signal_values[candidate_positions] = direction_value
        trades = forward_trade_returns(
            segment,
            pd.Series(signal_values, index=epochs, dtype=float),
            product,
            horizon=horizon,
        )
        if len(trades) != len(candidate_positions):
            raise RuntimeError("Candidate outcomes did not align with contiguous candles")
        result.append(
            _CandidateOutcomes(
                decision_epochs=np.asarray(
                    [segment[position].epoch for position in candidate_positions]
                ),
                gross_returns_bps=np.asarray(
                    [trade.gross_pnl / product.stake * 10_000.0 for trade in trades], dtype=float
                ),
                net_returns_bps=np.asarray(
                    [trade.net_pnl / product.stake * 10_000.0 for trade in trades], dtype=float
                ),
            )
        )
    return result


def _seed_for(seed: int, *labels: str | int) -> int:
    text = "|".join(str(label) for label in labels)
    return seed + sum((position + 1) * ord(character) for position, character in enumerate(text))


def _compute_symbol_test(
    *,
    symbol: str,
    flag_name: str,
    flag_values: pd.Series[bool],
    candles: Sequence[Candle],
    candidate_groups: Sequence[_CandidateOutcomes],
    n_signals: int,
    horizon: int,
    direction: DIRECTION,
    split: SPLIT,
    alpha: float,
    seed: int,
    permutations: int,
    bootstrap_reps: int,
) -> _ComputedTest:
    active_gross: list[np.ndarray] = []
    active_net: list[np.ndarray] = []
    shift_groups: list[tuple[np.ndarray, np.ndarray]] = []
    for group in candidate_groups:
        masks = (
            flag_values.reindex(pd.Index(group.decision_epochs)).fillna(False).to_numpy(dtype=bool)
        )
        shift_groups.append((masks, group.net_returns_bps))
        if masks.any():
            active_gross.append(group.gross_returns_bps[masks])
            active_net.append(group.net_returns_bps[masks])
    gross_values = np.concatenate(active_gross) if active_gross else np.zeros(0, dtype=float)
    net_values = np.concatenate(active_net) if active_net else np.zeros(0, dtype=float)
    n_effective = len(net_values)
    non_tied_count = sum(value != 0.0 for value in net_values)
    confidence = _stationary_bootstrap_ci_groups(
        active_net,
        seed=_seed_for(seed, symbol, flag_name, horizon, direction, split, "ci"),
        replications=bootstrap_reps,
    )
    lower, upper = (None, None) if confidence is None else confidence
    signed_nets = tuple(float(value) for value in net_values)
    row = EdgeTestRow(
        scope="symbol",
        symbol=symbol,
        flag=flag_name,
        horizon=horizon,
        direction=direction,
        split=split,
        n_signals=n_signals,
        n_effective=n_effective,
        n_non_tied=non_tied_count,
        gross_mean_bps=float(gross_values.mean()) if len(gross_values) else None,
        net_mean_bps=float(net_values.mean()) if len(net_values) else None,
        hit_rate=hit_rate(signed_nets),
        binomial_p=binomial_p_value(signed_nets),
        circular_shift_p=_segmented_circular_shift_p_value(
            shift_groups,
            shifts=permutations,
            seed=_seed_for(seed, symbol, flag_name, horizon, direction, split, "shift"),
        ),
        bootstrap_low_bps=lower,
        bootstrap_high_bps=upper,
        minimum_detectable_hit_deviation_pct=minimum_detectable_hit_deviation(
            n_effective,
            alpha=alpha,
        ),
    )
    return _ComputedTest(
        row=row,
        gross_returns_bps=gross_values,
        net_returns_bps=net_values,
        bootstrap_groups=active_net,
        shift_groups=shift_groups,
    )


def _pool_tests(
    members: Sequence[_ComputedTest],
    *,
    flag_name: str,
    horizon: int,
    direction: DIRECTION,
    split: SPLIT,
    alpha: float,
    seed: int,
    bootstrap_reps: int,
) -> _ComputedTest:
    gross_values = (
        np.concatenate(
            [member.gross_returns_bps for member in members if len(member.gross_returns_bps)]
        )
        if any(len(member.gross_returns_bps) for member in members)
        else np.zeros(0, dtype=float)
    )
    net_values = (
        np.concatenate(
            [member.net_returns_bps for member in members if len(member.net_returns_bps)]
        )
        if any(len(member.net_returns_bps) for member in members)
        else np.zeros(0, dtype=float)
    )
    weights = [sqrt(member.row.n_effective) for member in members]
    stouffer_z, combined_p = _weighted_stouffer(
        [member.row.circular_shift_p for member in members],
        weights,
    )
    bootstrap_groups = [group for member in members for group in member.bootstrap_groups]
    confidence = _stationary_bootstrap_ci_groups(
        bootstrap_groups,
        seed=_seed_for(seed, "pooled", flag_name, horizon, direction, split, "ci"),
        replications=bootstrap_reps,
    )
    lower, upper = (None, None) if confidence is None else confidence
    n_effective = sum(member.row.n_effective for member in members)
    row = EdgeTestRow(
        scope="pooled",
        symbol="ALL",
        flag=flag_name,
        horizon=horizon,
        direction=direction,
        split=split,
        n_signals=sum(member.row.n_signals for member in members),
        n_effective=n_effective,
        n_non_tied=sum(value != 0.0 for value in net_values),
        gross_mean_bps=float(gross_values.mean()) if len(gross_values) else None,
        net_mean_bps=float(net_values.mean()) if len(net_values) else None,
        hit_rate=hit_rate(tuple(float(value) for value in net_values)),
        binomial_p=binomial_p_value(tuple(float(value) for value in net_values)),
        circular_shift_p=combined_p,
        stouffer_z=stouffer_z,
        bootstrap_low_bps=lower,
        bootstrap_high_bps=upper,
        minimum_detectable_hit_deviation_pct=minimum_detectable_hit_deviation(
            n_effective,
            alpha=alpha,
        ),
    )
    shifts = [group for member in members for group in member.shift_groups]
    return _ComputedTest(row, gross_values, net_values, bootstrap_groups, shifts)


def analyze_flag_series(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    flag_series_by_symbol: Mapping[str, Mapping[str, pd.Series[bool]]] | None = None,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    permutations: int = DEFAULT_PERMUTATIONS,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    product: ProductSpec | None = None,
) -> EdgeAnalysis:
    """Analyze causal flags in 70% discovery and 30% confirmation windows."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive candle counts")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    if permutations < DEFAULT_PERMUTATIONS:
        raise ValueError(f"permutations must be at least {DEFAULT_PERMUTATIONS}")
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be greater than zero")
    selected_product = product or _default_product()
    computed: dict[tuple[str, str, int, DIRECTION, SPLIT], _ComputedTest] = {}
    selected_symbols = sorted(candles_by_symbol)

    for symbol in selected_symbols:
        candles = list(candles_by_symbol[symbol])
        if not candles:
            continue
        if any(candle.symbol != symbol for candle in candles):
            raise ValueError(f"Candles for {symbol} contain a different symbol")
        epochs = pd.Index([candle.epoch for candle in candles])
        if any(left >= right for left, right in pairwise(epochs)):
            raise ValueError(f"Candles for {symbol} must be unique and chronological")
        if any(candle.granularity != GRANULARITY_SECONDS for candle in candles):
            raise ValueError("Flag edge analysis requires 60-second candles")
        if flag_series_by_symbol is None:
            symbol_flags = causal_flag_series(candles)
        else:
            symbol_flags = dict(flag_series_by_symbol.get(symbol, {}))
        if not symbol_flags:
            raise ValueError(f"No flag series supplied for {symbol}")
        for name, series in symbol_flags.items():
            if not series.index.equals(epochs):
                raise ValueError(f"Flag {name} index must match candles for {symbol}")
            symbol_flags[name] = series.fillna(False).astype(bool)

        discovery_end = int(len(candles) * 0.7)
        split_ranges: tuple[tuple[SPLIT, int, int], ...] = (
            ("discovery", 0, discovery_end),
            ("confirmation", discovery_end, len(candles)),
        )
        for split_name, start, stop in split_ranges:
            split_candles = candles[start:stop]
            for horizon in horizons:
                for direction in DIRECTIONS:
                    candidate_groups = _candidate_outcomes(
                        split_candles,
                        horizon=horizon,
                        direction=direction,
                        product=selected_product,
                    )
                    for flag_name, full_flags in symbol_flags.items():
                        flag_values = full_flags.iloc[start:stop]
                        signal_count = int(flag_values.sum())
                        computed[(symbol, flag_name, horizon, direction, split_name)] = (
                            _compute_symbol_test(
                                symbol=symbol,
                                flag_name=flag_name,
                                flag_values=flag_values,
                                candles=split_candles,
                                candidate_groups=candidate_groups,
                                n_signals=signal_count,
                                horizon=horizon,
                                direction=direction,
                                split=split_name,
                                alpha=alpha,
                                seed=seed,
                                permutations=permutations,
                                bootstrap_reps=bootstrap_reps,
                            )
                        )

    all_tests: list[_ComputedTest] = list(computed.values())
    for flag_name in sorted({key[1] for key in computed}):
        for horizon in horizons:
            for direction in DIRECTIONS:
                for split_name in ("discovery", "confirmation"):
                    members = [
                        computed[(symbol, flag_name, horizon, direction, split_name)]
                        for symbol in selected_symbols
                        if (symbol, flag_name, horizon, direction, split_name) in computed
                    ]
                    if members:
                        all_tests.append(
                            _pool_tests(
                                members,
                                flag_name=flag_name,
                                horizon=horizon,
                                direction=direction,
                                split=split_name,
                                alpha=alpha,
                                seed=seed,
                                bootstrap_reps=bootstrap_reps,
                            )
                        )

    rows = [test.row for test in all_tests]
    test_count = len(rows)
    circular_adjusted = benjamini_hochberg([row.circular_shift_p for row in rows])
    binomial_adjusted = benjamini_hochberg([row.binomial_p for row in rows])
    adjusted_rows: list[EdgeTestRow] = []
    bonferroni_count = 0
    for row, circular_p, binomial_p_value_adjusted in zip(
        rows,
        circular_adjusted,
        binomial_adjusted,
        strict=True,
    ):
        bonferroni_p = min(1.0, row.circular_shift_p * test_count)
        if bonferroni_p < alpha:
            bonferroni_count += 1
        adjusted_rows.append(
            row.model_copy(
                update={
                    "adjusted_p": circular_p,
                    "binomial_adjusted_p": binomial_p_value_adjusted,
                    "bonferroni_adjusted_p": bonferroni_p,
                }
            )
        )
    paired_rows = {
        (row.scope, row.symbol, row.flag, row.horizon, row.direction, row.split): row
        for row in adjusted_rows
    }
    confirmed_rows: list[EdgeTestRow] = []
    for row in adjusted_rows:
        discovery = paired_rows.get(
            (row.scope, row.symbol, row.flag, row.horizon, row.direction, "discovery")
        )
        confirmation = paired_rows.get(
            (row.scope, row.symbol, row.flag, row.horizon, row.direction, "confirmation")
        )
        confirmed = bool(
            discovery is not None
            and confirmation is not None
            and discovery.adjusted_p < alpha
            and confirmation.adjusted_p < alpha
            and discovery.net_mean_bps is not None
            and confirmation.net_mean_bps is not None
            and discovery.net_mean_bps > 0.0
            and confirmation.net_mean_bps > 0.0
        )
        confirmed_rows.append(row.model_copy(update={"confirmed": confirmed}))
    return EdgeAnalysis(
        rows=confirmed_rows,
        test_count=test_count,
        bonferroni_count=bonferroni_count,
        alpha=alpha,
    )


def load_stored_candles(
    database_path: Path | str,
    symbols: Sequence[str],
) -> dict[str, list[Candle]]:
    """Read 60-second candles without opening DuckDB in writable mode."""
    with DuckDBStorage(database_path, read_only=True) as storage:
        if not storage.has_candle_table(GRANULARITY_SECONDS):
            raise ValueError("missing candles_60 table")
        return {symbol: storage.read_candles(GRANULARITY_SECONDS, symbol) for symbol in symbols}


def _execute_null_replication(
    replication: int,
    config: _NullWorkerConfig,
) -> _NullReplicationSummary:
    started = time.perf_counter()
    simulated: dict[str, list[Candle]] = {}
    for symbol_position, (symbol, epochs) in enumerate(sorted(config.epochs_by_symbol.items())):
        simulated[symbol] = make_gbm_candles(
            symbol,
            epochs=epochs,
            stated_volatility=SYMBOLS[symbol].stated_vol_pct / 100.0,
            granularity=GRANULARITY_SECONDS,
            seed=_seed_for(config.seed, replication, symbol_position, symbol),
        )
    analysis = analyze_flag_series(
        simulated,
        horizons=config.horizons,
        alpha=config.alpha,
        seed=_seed_for(config.seed, replication, "analysis"),
        permutations=config.permutations,
        bootstrap_reps=config.bootstrap_reps,
        product=config.product,
    )
    return _NullReplicationSummary(
        replication=replication,
        elapsed_seconds=time.perf_counter() - started,
        test_count=analysis.test_count,
        raw_significant=sum(row.circular_shift_p < config.alpha for row in analysis.rows),
        bh_surviving=sum(row.adjusted_p < config.alpha for row in analysis.rows),
        confirmed_edges=sum(
            row.scope == "pooled" and row.split == "confirmation" and row.confirmed
            for row in analysis.rows
        ),
    )


def _initialize_null_worker(config: _NullWorkerConfig) -> None:
    global _NULL_WORKER_CONFIG
    _NULL_WORKER_CONFIG = config


def _run_null_replication_worker(replication: int) -> _NullReplicationSummary:
    if _NULL_WORKER_CONFIG is None:
        raise RuntimeError("Null-control worker was not initialized")
    return _execute_null_replication(replication, _NULL_WORKER_CONFIG)


def run_null_control(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    *,
    replications: int = DEFAULT_NULL_REPLICATIONS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    permutations: int = DEFAULT_PERMUTATIONS,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    product: ProductSpec | None = None,
    jobs: int | None = None,
) -> NullControlResult:
    """Run the full analysis on seeded driftless GBM datasets in deterministic workers."""
    if replications < DEFAULT_NULL_REPLICATIONS:
        raise ValueError(f"replications must be at least {DEFAULT_NULL_REPLICATIONS}")
    worker_count = _default_jobs() if jobs is None else jobs
    if worker_count <= 0:
        raise ValueError("jobs must be greater than zero")
    worker_count = min(worker_count, replications)
    config = _NullWorkerConfig(
        epochs_by_symbol={
            symbol: tuple(candle.epoch for candle in candles)
            for symbol, candles in candles_by_symbol.items()
        },
        horizons=tuple(horizons),
        alpha=alpha,
        seed=seed,
        permutations=permutations,
        bootstrap_reps=bootstrap_reps,
        product=product,
    )
    test_count = 0
    raw_significant = 0
    bh_surviving = 0
    confirmed_edges = 0
    completed_count = 0
    total_replication_seconds = 0.0
    run_started = time.perf_counter()
    pending: dict[Future[_NullReplicationSummary], float] = {}

    def accumulate(result: _NullReplicationSummary) -> None:
        nonlocal test_count, raw_significant, bh_surviving, confirmed_edges
        nonlocal completed_count, total_replication_seconds
        test_count += result.test_count
        raw_significant += result.raw_significant
        bh_surviving += result.bh_surviving
        confirmed_edges += result.confirmed_edges
        completed_count += 1
        total_replication_seconds += result.elapsed_seconds
        now = time.perf_counter()
        elapsed = now - run_started
        eta = _estimate_null_eta_seconds(
            replications=replications,
            completed_count=completed_count,
            worker_count=worker_count,
            average_replication_seconds=total_replication_seconds / completed_count,
            active_replication_seconds=sum(now - started for started in pending.values()),
        )
        _progress(
            f"null replication {result.replication + 1}/{replications} finished in "
            f"{result.elapsed_seconds:.1f}s; elapsed {elapsed:.1f}s; ETA {eta:.1f}s"
        )

    if worker_count == 1:
        for replication in range(replications):
            _progress(f"null replication {replication + 1}/{replications} started")
            accumulate(_execute_null_replication(replication, config))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_null_worker,
            initargs=(config,),
        ) as executor:
            next_replication = 0
            while next_replication < min(worker_count, replications):
                _progress(f"null replication {next_replication + 1}/{replications} started")
                started = time.perf_counter()
                future = executor.submit(_run_null_replication_worker, next_replication)
                pending[future] = started
                next_replication += 1
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                results = [future.result() for future in completed]
                for future in completed:
                    pending.pop(future)
                for _ in completed:
                    if next_replication < replications:
                        _progress(f"null replication {next_replication + 1}/{replications} started")
                        started = time.perf_counter()
                        new_future = executor.submit(
                            _run_null_replication_worker,
                            next_replication,
                        )
                        pending[new_future] = started
                        next_replication += 1
                for result in results:
                    accumulate(result)
    total_tests = test_count
    return NullControlResult(
        replications=replications,
        tests_per_replication=test_count // replications,
        raw_significant_fraction=raw_significant / total_tests if total_tests else 0.0,
        bh_surviving_fraction=bh_surviving / total_tests if total_tests else 0.0,
        confirmed_edges=confirmed_edges,
    )


def _report_paths(out_dir: Path) -> tuple[Path, Path]:
    date_text = datetime.now(UTC).strftime("%Y%m%d")
    return out_dir / f"flag_edge_{date_text}.csv", out_dir / f"flag_edge_{date_text}.md"


def write_reports(
    analysis: EdgeAnalysis,
    null_control: NullControlResult,
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    *,
    out_dir: Path,
    product: ProductSpec,
) -> tuple[Path, Path]:
    """Write the complete row-level CSV and a concise interpretation report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, markdown_path = _report_paths(out_dir)
    fieldnames = list(EdgeTestRow.model_fields)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in analysis.rows:
            writer.writerow(row.model_dump())

    pooled_confirmation = sorted(
        (row for row in analysis.rows if row.scope == "pooled" and row.split == "confirmation"),
        key=lambda row: (row.adjusted_p, row.flag, row.horizon, row.direction),
    )
    discovery_rows = {
        (row.flag, row.horizon, row.direction): row
        for row in analysis.rows
        if row.scope == "pooled" and row.split == "discovery"
    }
    confirmed = [row for row in pooled_confirmation if row.confirmed]
    bonferroni_threshold = analysis.alpha / analysis.test_count if analysis.test_count else 1.0
    lines = [
        "# Flag Edge Report",
        "",
        "> Descriptive analytics on random processes; no predictive edge is implied.",
        "",
        f"- UTC report date: {datetime.now(UTC).date().isoformat()}",
        f"- ProductSpec: multiplier={product.multiplier}, stake={product.stake}, "
        f"commission/trade={product.commission_per_trade}, "
        f"spread fraction={product.spread_fraction}.",
        f"- Alpha: {analysis.alpha}; total hypothesis rows per p-value family: "
        f"{analysis.test_count}; Bonferroni threshold: {bonferroni_threshold:.6g}; "
        f"Bonferroni discoveries: {analysis.bonferroni_count}.",
        "",
        "## Data Windows",
        "",
        "| Symbol | Candles | First candle epoch | Last candle epoch |",
        "|---|---:|---:|---:|",
    ]
    for symbol, candles in sorted(candles_by_symbol.items()):
        if candles:
            lines.append(
                f"| {symbol} | {len(candles)} | {candles[0].epoch} | {candles[-1].epoch} |"
            )
    lines.extend(
        [
            "",
            "## Flag Definitions",
            "",
        ]
    )
    for name in FLAG_NAMES:
        lines.append(f"- `{name}`: {FLAG_DEFINITIONS[name]}.")
    lines.extend(
        [
            "",
            "## Null Control",
            "",
            f"- Seeded driftless GBM replications: {null_control.replications}.",
            f"- Tests per replication: {null_control.tests_per_replication}.",
            f"- Raw circular-shift p < alpha: {null_control.raw_significant_fraction:.3%} "
            "(expected approximately alpha).",
            f"- BH survivors / all null tests: {null_control.bh_surviving_fraction:.3%} "
            "(expected near zero).",
            f"- Confirmed null edges: {null_control.confirmed_edges}.",
            "",
            "## Top 15 Pooled Tests",
            "",
            "Sorted by confirmation-window BH-adjusted circular-shift p-value. "
            "Pooled z uses Stouffer combination across symbols.",
            "",
            "| Flag | Horizon | Direction | Discovery net bp | Confirmation net bp | "
            "Confirmation q | Stouffer z | Effective n |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pooled_confirmation[:15]:
        discovery = discovery_rows[(row.flag, row.horizon, row.direction)]
        discovery_mean = (
            "n/a" if discovery.net_mean_bps is None else f"{discovery.net_mean_bps:.3f}"
        )
        confirmation_mean = "n/a" if row.net_mean_bps is None else f"{row.net_mean_bps:.3f}"
        stouffer = "n/a" if row.stouffer_z is None else f"{row.stouffer_z:.3f}"
        lines.append(
            f"| {row.flag} | {row.horizon} | {row.direction} | {discovery_mean} | "
            f"{confirmation_mean} | {row.adjusted_p:.5g} | {stouffer} | {row.n_effective} |"
        )
    lines.extend(["", "## Confirmed Edges", ""])
    if not confirmed:
        lines.append(
            "None. No pooled flag replicated with positive net return and adjusted p < alpha."
        )
    else:
        for row in confirmed:
            lines.append(
                f"- `{row.flag}` at {row.horizon} candles, {row.direction}: "
                f"confirmation q={row.adjusted_p:.5g}, net mean={row.net_mean_bps:.3f} bp."
            )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "Signals use only causal trailing features. Entry is at the next candle open; "
            "exit is at the close of candle t+h. The shared backtester applies ProductSpec "
            "costs and suppresses trades across gaps. Each symbol is split chronologically "
            "70% discovery / 30% confirmation. For h > 1, decisions are sampled every h "
            "candles within each contiguous segment. The primary p-value is the seeded "
            "circular-shift test; binomial hit-rate p-values are separately BH-adjusted. "
            "Confidence intervals use a stationary bootstrap. Non-significance means no "
            "detectable edge at this sample size, not proof of no edge.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path


def write_quick_report(
    analysis: EdgeAnalysis,
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    *,
    out_dir: Path,
) -> Path:
    """Write a reduced smoke artifact that is explicitly not statistical evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    time_text = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"quick_{time_text}.md"
    lines = [
        "# QUICK FLAG EDGE SMOKE TEST: NOT VALID EVIDENCE",
        "",
        "> This reduced run is for smoke testing only. It is NOT VALID EVIDENCE and "
        "must not be interpreted as a statistical conclusion.",
        "",
        f"- UTC run date: {datetime.now(UTC).date().isoformat()}",
        f"- Hypothesis rows: {analysis.test_count}.",
        "- Null control was skipped by request.",
        "",
        "## Data Windows",
        "",
        "| Symbol | Candles | First epoch | Last epoch |",
        "|---|---:|---:|---:|",
    ]
    for symbol, candles in sorted(candles_by_symbol.items()):
        lines.append(f"| {symbol} | {len(candles)} | {candles[0].epoch} | {candles[-1].epoch} |")
    lines.extend(["", "## Descriptive Results", ""])
    lines.extend(
        f"- `{row.scope}` `{row.flag}` h={row.horizon} {row.direction} "
        f"{row.split}: n_eff={row.n_effective}, net_mean_bps={row.net_mean_bps}."
        for row in analysis.rows[:15]
    )
    if not analysis.rows:
        lines.append("No rows were produced.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="DuckDB path; defaults to DB_PATH")
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=sorted(SYMBOLS),
        default=list(SYMBOLS),
        help="Volatility symbols to analyze (default: all 13)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="forward horizons in 60-second candles",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--commission-per-trade", type=float, default=0.0)
    parser.add_argument("--spread-fraction", type=float, default=0.0)
    parser.add_argument("--stop-out-fraction", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPLICATIONS)
    parser.add_argument("--null-reps", type=int, default=DEFAULT_NULL_REPLICATIONS)
    parser.add_argument("--jobs", type=int, default=_default_jobs())
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a reduced smoke test and write a NOT VALID EVIDENCE markdown file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline edge report and return its documented exit status."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between zero and one")
    if any(horizon <= 0 for horizon in args.horizons) or len(set(args.horizons)) != len(
        args.horizons
    ):
        parser.error("--horizons must be unique positive integers")
    if args.permutations < DEFAULT_PERMUTATIONS:
        parser.error(f"--permutations must be at least {DEFAULT_PERMUTATIONS}")
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")
    if args.null_reps < DEFAULT_NULL_REPLICATIONS:
        parser.error(f"--null-reps must be at least {DEFAULT_NULL_REPLICATIONS}")
    if args.jobs <= 0:
        parser.error("--jobs must be greater than zero")

    database_path = args.db if args.db is not None else Settings().db_path
    selected_symbols = list(args.symbols)
    selected_horizons = tuple(args.horizons)
    if args.quick:
        selected_symbols = selected_symbols[:2]
        selected_horizons = selected_horizons[:2]

    load_started = time.perf_counter()
    _progress("load started")
    try:
        candles_by_symbol = load_stored_candles(database_path, selected_symbols)
    except (OSError, ValueError, duckdb.Error) as exc:
        _progress(f"load finished in {time.perf_counter() - load_started:.1f}s (failed)")
        print(f"Flag edge report requires stored candles: {exc}")
        return 2
    _progress(f"load finished in {time.perf_counter() - load_started:.1f}s")
    too_little = [
        f"{symbol} ({len(candles_by_symbol[symbol])})"
        for symbol in selected_symbols
        if len(candles_by_symbol[symbol]) < MINIMUM_CANDLES
    ]
    if too_little:
        print(
            f"Flag edge report requires at least {MINIMUM_CANDLES} candles per symbol: "
            f"{', '.join(too_little)}"
        )
        return 2

    try:
        product = ProductSpec(
            contract_type="multiplier",
            stake=args.stake,
            multiplier=args.multiplier,
            stop_out_fraction=args.stop_out_fraction,
            commission_per_trade=args.commission_per_trade,
            spread_fraction=args.spread_fraction,
        )
    except ValidationError as exc:
        parser.error(f"invalid ProductSpec: {exc}")
    analysis_started = time.perf_counter()
    _progress("real-data analysis started")
    try:
        analysis = analyze_flag_series(
            candles_by_symbol,
            horizons=selected_horizons,
            alpha=args.alpha,
            seed=args.seed,
            permutations=args.permutations,
            bootstrap_reps=args.bootstrap_reps,
            product=product,
        )
    finally:
        _progress(f"real-data analysis finished in {time.perf_counter() - analysis_started:.1f}s")

    if args.quick:
        _progress("null control skipped in --quick mode")
        report_started = time.perf_counter()
        _progress("report writing started")
        try:
            quick_path = write_quick_report(
                analysis,
                candles_by_symbol,
                out_dir=args.out_dir,
            )
        finally:
            _progress(f"report writing finished in {time.perf_counter() - report_started:.1f}s")
        print(
            f"Quick smoke report: {len(candles_by_symbol)} symbols, "
            f"{analysis.test_count} hypothesis rows; NOT VALID EVIDENCE"
        )
        print(f"Quick Markdown: {quick_path}")
        return 0

    null_started = time.perf_counter()
    effective_jobs = min(args.jobs, args.null_reps)
    _progress(
        f"null control started: {args.null_reps} replications, "
        f"{effective_jobs} workers (requested jobs={args.jobs})"
    )
    try:
        null_control = run_null_control(
            candles_by_symbol,
            replications=args.null_reps,
            horizons=selected_horizons,
            alpha=args.alpha,
            seed=args.seed,
            permutations=args.permutations,
            bootstrap_reps=args.bootstrap_reps,
            product=product,
            jobs=args.jobs,
        )
    finally:
        _progress(f"null control finished in {time.perf_counter() - null_started:.1f}s")

    report_started = time.perf_counter()
    _progress("report writing started")
    try:
        csv_path, markdown_path = write_reports(
            analysis,
            null_control,
            candles_by_symbol,
            out_dir=args.out_dir,
            product=product,
        )
    finally:
        _progress(f"report writing finished in {time.perf_counter() - report_started:.1f}s")
    confirmed_count = sum(
        row.scope == "pooled" and row.split == "confirmation" and row.confirmed
        for row in analysis.rows
    )
    print(
        f"Flag edge report: {len(candles_by_symbol)} symbols, "
        f"{analysis.test_count} hypothesis rows, {confirmed_count} confirmed pooled edges"
    )
    print(
        f"Null control: raw p<alpha {null_control.raw_significant_fraction:.3%}; "
        f"BH survivors {null_control.bh_surviving_fraction:.3%}"
    )
    print(f"Bonferroni discoveries: {analysis.bonferroni_count}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    if null_control.bh_surviving_fraction > 2.0 * args.alpha:
        print("Null-control failure: BH-surviving rate exceeds 2 x alpha")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
