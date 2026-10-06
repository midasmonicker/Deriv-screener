"""Event-driven backtesting and bias-adjusted evaluation.

Deriv candle ``epoch`` values identify the candle open in UTC epoch seconds.
A signal indexed by a candle is available at its close (``epoch + granularity``),
enters at the next contiguous candle's open, and exits at the execution candle's
close or at the configured intrabar stop-out.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import ceil, isfinite, sqrt
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.stats import norm

from deriv_vol_lab.data.models import Candle


class ProductSpec(BaseModel):
    """Contract terms, with every friction represented as a configurable input.

    ``spread_fraction`` is the full round-trip spread as a fraction of
    multiplier notional for leveraged products, or of stake for fixed-payout
    contracts. ``commission_per_trade`` is charged once per completed contract.
    ``stop_out_fraction`` caps leveraged-contract gross losses as a fraction of
    stake. ``payout_fraction`` is the profit paid on a winning fixed-payout
    contract, expressed as a fraction of stake and constrained below 100%.
    """

    model_config = ConfigDict(frozen=True)

    contract_type: Literal["multiplier", "fixed_payout"]
    stake: float
    commission_per_trade: float = 0.0
    spread_fraction: float = 0.0
    multiplier: float | None = None
    stop_out_fraction: float | None = None
    payout_fraction: float | None = None

    @model_validator(mode="after")
    def validate_contract_terms(self) -> ProductSpec:
        for name in ("stake", "commission_per_trade", "spread_fraction"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.multiplier is not None and not isfinite(self.multiplier):
            raise ValueError("multiplier must be finite")
        if self.stake <= 0:
            raise ValueError("stake must be greater than zero")
        if self.commission_per_trade < 0:
            raise ValueError("commission_per_trade cannot be negative")
        if self.spread_fraction < 0:
            raise ValueError("spread_fraction cannot be negative")
        if self.contract_type == "multiplier":
            if self.multiplier is None or self.multiplier <= 0:
                raise ValueError("multiplier contracts require multiplier > 0")
            if self.stop_out_fraction is None or not 0 < self.stop_out_fraction <= 1:
                raise ValueError("multiplier contracts require stop_out_fraction in (0, 1]")
            if self.payout_fraction is not None:
                raise ValueError("multiplier contracts do not use payout_fraction")
        else:
            if self.payout_fraction is None or not 0 < self.payout_fraction < 1:
                raise ValueError("fixed-payout contracts require payout_fraction in (0, 1)")
            if self.multiplier is not None or self.stop_out_fraction is not None:
                raise ValueError(
                    "fixed-payout contracts do not use multiplier or stop_out_fraction"
                )
        return self


class TradeResult(BaseModel):
    """Settled contract event, including gross PnL and explicit costs."""

    symbol: str
    decision_epoch: int
    entry_epoch: int
    exit_epoch: int
    direction: Literal[-1, 1]
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    spread_cost: float
    net_pnl: float
    won: bool
    gross_won: bool | None = None
    exit_reason: Literal["bar_close", "stop_out"]


class BacktestReport(BaseModel):
    """Performance metrics computed from the supplied test-period trades."""

    observations: int
    period_observations: int = 0
    gap_count: int = 0
    net_pnl: float
    sharpe: float
    per_trade_mean_pnl: float = 0.0
    per_trade_std_pnl: float = 0.0
    max_drawdown: float
    win_rate: float
    expectancy_per_trade: float
    break_even_win_rate: float | None
    deflated_sharpe_ratio: float
    trades: list[TradeResult]


class WalkForwardSplit(BaseModel):
    """Positional train/test ranges, with end positions exclusive."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardWindow(BaseModel):
    """Backtest metrics for one out-of-sample fold."""

    split: WalkForwardSplit
    report: BacktestReport


class WalkForwardReport(BaseModel):
    """Fold-by-fold and pooled out-of-sample performance."""

    windows: list[WalkForwardWindow]
    out_of_sample: BacktestReport


class RealityCheckResult(BaseModel):
    """White's Reality Check statistic and seeded bootstrap p-value."""

    observed_max_mean: float
    p_value: float
    bootstrap_count: int
    strategy_count: int
    seed: int


def backtest(
    candles: Sequence[Candle],
    signal: pd.Series[float],
    product: ProductSpec,
    *,
    initial_equity: float = 10_000.0,
    strategy_trials: int = 1,
) -> BacktestReport:
    """Settle each nonzero close-time signal at the next candle open and close.

    A signal at index ``i`` cannot execute until candle ``i + 1`` opens.
    Signals are reduced to direction by sign; zero or missing values are flat.
    Each event is an independent one-candle contract, which makes exposure,
    fees, and stop-out settlement explicit and keeps the event loop causal.
    """
    if len(signal) != len(candles):
        raise ValueError("signal length must match candle count")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than zero")
    _validate_candle_sequence(candles)
    _positive_int(strategy_trials, "strategy_trials")
    if not signal.index.equals(pd.Index([item.epoch for item in candles])):
        raise ValueError("signal index must equal candle epochs in chronological order")

    trades: list[TradeResult] = []
    numeric_signals = signal.to_numpy(dtype=float)
    for decision_position in range(len(candles) - 1):
        raw_signal = numeric_signals[decision_position]
        if not np.isfinite(raw_signal) or raw_signal == 0:
            continue
        direction: Literal[-1, 1] = 1 if raw_signal > 0 else -1
        entry_bar = candles[decision_position + 1]
        if not _are_contiguous(candles[decision_position], entry_bar):
            continue
        trades.append(_settle_trade(candles[decision_position], entry_bar, direction, product))
    return _report(
        trades,
        candles,
        product,
        initial_equity=initial_equity,
        strategy_trials=strategy_trials,
    )


def walk_forward_splits(
    candle_count: int,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Create expanding-train, forward-only test splits with exclusive ends."""
    if candle_count < 0:
        raise ValueError("candle_count cannot be negative")
    _positive_int(train_size, "train_size")
    _positive_int(test_size, "test_size")
    step = test_size if step_size is None else step_size
    _positive_int(step, "step_size")
    splits: list[WalkForwardSplit] = []
    test_start = train_size
    while test_start < candle_count:
        test_end = min(test_start + test_size, candle_count)
        splits.append(
            WalkForwardSplit(
                train_start=0,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start += step
    return splits


def walk_forward_backtest(
    candles: Sequence[Candle],
    signal: pd.Series[float],
    product: ProductSpec,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_equity: float = 10_000.0,
    strategy_trials: int = 1,
) -> WalkForwardReport:
    """Report fold metrics and pooled results for out-of-sample trade events.

    Signals are assumed to have been generated causally from data available at
    their indexed candle close. The engine executes each at the next open. An
    event is included in a fold only if its execution candle is within that
    fold's test range; no training-period trade contributes to OOS metrics.
    """
    if len(signal) != len(candles):
        raise ValueError("signal length must match candle count")
    _validate_candle_sequence(candles)
    _positive_int(strategy_trials, "strategy_trials")
    if step_size is not None and step_size < test_size:
        raise ValueError("step_size must be greater than or equal to test_size")
    if not signal.index.equals(pd.Index([item.epoch for item in candles])):
        raise ValueError("signal index must equal candle epochs in chronological order")
    splits = walk_forward_splits(
        len(candles),
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    all_trades: list[TradeResult] = []
    all_test_candles: list[Candle] = []
    test_segments: list[Sequence[Candle]] = []
    windows: list[WalkForwardWindow] = []
    for split in splits:
        fold_trades: list[TradeResult] = []
        fold_candles = candles[split.test_start : split.test_end]
        all_test_candles.extend(fold_candles)
        test_segments.append(fold_candles)
        for decision_position in range(max(0, split.test_start - 1), split.test_end - 1):
            execution_position = decision_position + 1
            if execution_position < split.test_start or execution_position >= split.test_end:
                continue
            if not _are_contiguous(candles[decision_position], candles[execution_position]):
                continue
            raw_signal = float(signal.iloc[decision_position])
            if not np.isfinite(raw_signal) or raw_signal == 0:
                continue
            direction: Literal[-1, 1] = 1 if raw_signal > 0 else -1
            fold_trades.append(
                _settle_trade(
                    candles[decision_position],
                    candles[execution_position],
                    direction,
                    product,
                )
            )
        all_trades.extend(fold_trades)
        windows.append(
            WalkForwardWindow(
                split=split,
                report=_report(
                    fold_trades,
                    fold_candles,
                    product,
                    initial_equity=initial_equity,
                    strategy_trials=strategy_trials,
                ),
            )
        )
    return WalkForwardReport(
        windows=windows,
        out_of_sample=_report(
            all_trades,
            all_test_candles,
            product,
            initial_equity=initial_equity,
            strategy_trials=strategy_trials,
            timeline_segments=test_segments,
        ),
    )


def deflated_sharpe_ratio(returns: Sequence[float], strategy_trials: int = 1) -> float:
    """Estimate the probability the observed Sharpe exceeds the search-adjusted null.

    This implements the Bailey-López de Prado probabilistic/deflated Sharpe
    approximation using the expected maximum Sharpe among ``strategy_trials``.
    Searching many strategies inflates apparent performance; this correction
    is approximate, assumes independent trials, and is not a substitute for
    a genuinely untouched out-of-sample set.
    """
    _positive_int(strategy_trials, "strategy_trials")
    values = _finite_1d(returns, "returns", minimum=2)
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation <= np.finfo(float).eps * max(1.0, float(np.abs(values).max())):
        return 0.5
    sharpe = float(values.mean() / standard_deviation)
    centered = values - values.mean()
    skewness = float(np.mean(centered**3) / np.mean(centered**2) ** 1.5)
    kurtosis = float(np.mean(centered**4) / np.mean(centered**2) ** 2)
    if strategy_trials == 1:
        expected_max = 0.0
    else:
        euler_mascheroni = 0.5772156649015329
        expected_max = (1 - euler_mascheroni) * float(
            norm.ppf(1 - 1 / strategy_trials)
        ) + euler_mascheroni * float(norm.ppf(1 - 1 / (strategy_trials * np.e)))
    denominator_squared = 1 - skewness * sharpe + (kurtosis - 1) * sharpe**2 / 4
    if denominator_squared <= 0:
        raise ValueError("return moments make the deflated-Sharpe approximation undefined")
    z_score = (sharpe - expected_max) * sqrt(values.size - 1) / sqrt(denominator_squared)
    return float(norm.cdf(z_score))


def whites_reality_check(
    strategy_returns: pd.DataFrame,
    *,
    bootstrap_count: int = 999,
    seed: int,
    block_length: int = 1,
) -> RealityCheckResult:
    """Run White's Reality Check against zero benchmark via circular block bootstrap.

    Candidate strategies are centered under the null, then resampled jointly
    in circular time blocks. The maximum statistic accounts for selecting the
    best candidate from this supplied family. Omitted trials, repeated research
    choices, and dependence not captured by ``block_length`` remain sources of
    multiple-testing bias.
    """
    _positive_int(bootstrap_count, "bootstrap_count")
    _positive_int(block_length, "block_length")
    if strategy_returns.empty:
        raise ValueError("strategy_returns must contain at least one strategy")
    if strategy_returns.shape[0] < 2:
        raise ValueError("strategy_returns must contain at least two time observations")
    values = strategy_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("strategy_returns must contain only finite values")
    if block_length > values.shape[0]:
        raise ValueError("block_length cannot exceed the number of observations")

    means = values.mean(axis=0)
    observed = float(np.max(means))
    centered = values - means
    row_count = values.shape[0]
    generator = np.random.default_rng(seed)
    bootstrap_maxima = np.empty(bootstrap_count, dtype=float)
    blocks_per_sample = (row_count + block_length - 1) // block_length
    offsets = np.arange(block_length)
    for iteration in range(bootstrap_count):
        starts = generator.integers(0, row_count, size=blocks_per_sample)
        indices = ((starts[:, None] + offsets) % row_count).ravel()[:row_count]
        bootstrap_maxima[iteration] = float(np.max(centered[indices].mean(axis=0)))
    p_value = float((1 + np.count_nonzero(bootstrap_maxima >= observed)) / (bootstrap_count + 1))
    return RealityCheckResult(
        observed_max_mean=observed,
        p_value=p_value,
        bootstrap_count=bootstrap_count,
        strategy_count=values.shape[1],
        seed=seed,
    )


def _settle_trade(
    decision_bar: Candle,
    execution_bar: Candle,
    direction: Literal[-1, 1],
    product: ProductSpec,
) -> TradeResult:
    entry_price = float(execution_bar.open)
    close_price = float(execution_bar.close)
    exit_price = close_price
    if entry_price <= 0 or close_price <= 0:
        raise ValueError("Backtest candle prices must be greater than zero")
    commission = product.commission_per_trade
    notional = (
        product.stake * (product.multiplier or 0.0)
        if product.contract_type == "multiplier"
        else product.stake
    )
    spread_cost = notional * product.spread_fraction
    exit_reason: Literal["bar_close", "stop_out"] = "bar_close"

    if product.contract_type == "multiplier":
        multiplier = product.multiplier
        stop_out_fraction = product.stop_out_fraction
        if multiplier is None or stop_out_fraction is None:
            raise RuntimeError("Validated multiplier product is missing required terms")
        favorable_return = direction * (close_price - entry_price) / entry_price
        gross_pnl = product.stake * multiplier * favorable_return
        adverse_price = float(execution_bar.low if direction == 1 else execution_bar.high)
        adverse_return = direction * (adverse_price - entry_price) / entry_price
        if product.stake * multiplier * adverse_return <= -product.stake * stop_out_fraction:
            gross_pnl = -product.stake * stop_out_fraction
            exit_price = entry_price * (1.0 - direction * stop_out_fraction / multiplier)
            exit_reason = "stop_out"
        won = gross_pnl > 0
    else:
        payout_fraction = product.payout_fraction
        if payout_fraction is None:
            raise RuntimeError("Validated fixed-payout product is missing payout_fraction")
        won = direction * (close_price - entry_price) > 0
        gross_pnl = product.stake * payout_fraction if won else -product.stake

    net_pnl = gross_pnl - commission - spread_cost
    return TradeResult(
        symbol=execution_bar.symbol,
        decision_epoch=decision_bar.epoch + decision_bar.granularity,
        entry_epoch=execution_bar.epoch,
        exit_epoch=execution_bar.epoch + execution_bar.granularity,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl=gross_pnl,
        commission=commission,
        spread_cost=spread_cost,
        net_pnl=net_pnl,
        won=net_pnl > 0,
        gross_won=won,
        exit_reason=exit_reason,
    )


def _report(
    trades: Sequence[TradeResult],
    candles: Sequence[Candle],
    product: ProductSpec,
    *,
    initial_equity: float,
    strategy_trials: int,
    timeline_segments: Sequence[Sequence[Candle]] | None = None,
) -> BacktestReport:
    pnl = np.asarray([trade.net_pnl for trade in trades], dtype=float)
    segments = [candles] if timeline_segments is None else timeline_segments
    period_pnl = (
        np.concatenate([_time_aligned_pnl(trades, segment) for segment in segments])
        if segments
        else np.asarray([], dtype=float)
    )
    returns = period_pnl / product.stake
    period_count = returns.size
    if candles:
        granularity = candles[0].granularity
        periods_per_year = 365 * 24 * 60 * 60 / granularity
    else:
        periods_per_year = 0.0
    if pnl.size:
        equity = initial_equity + np.cumsum(pnl)
        peaks = np.maximum.accumulate(np.concatenate(([initial_equity], equity)))[1:]
        drawdowns = (peaks - equity) / peaks
        per_trade_mean = float(pnl.mean())
        per_trade_std = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0
        win_rate = float(np.count_nonzero(pnl > 0) / pnl.size)
        net_sum = float(pnl.sum())
        expectancy = float(pnl.mean())
        max_drawdown = float(np.max(drawdowns))
        dsr = (
            deflated_sharpe_ratio((pnl / product.stake).tolist(), strategy_trials)
            if pnl.size > 1
            else 0.5
        )
    else:
        per_trade_mean = 0.0
        per_trade_std = 0.0
        win_rate = 0.0
        net_sum = 0.0
        expectancy = 0.0
        max_drawdown = 0.0
        dsr = 0.5
    period_std = float(returns.std(ddof=1)) if period_count > 1 else 0.0
    sharpe = float(returns.mean() / period_std * sqrt(periods_per_year)) if period_std > 0 else 0.0

    break_even: float | None = None
    if product.contract_type == "fixed_payout":
        payout_fraction = product.payout_fraction
        if payout_fraction is None:
            raise RuntimeError("Validated fixed-payout product is missing payout_fraction")
        per_trade_cost = product.commission_per_trade + product.stake * product.spread_fraction
        break_even = (product.stake + per_trade_cost) / (product.stake * (1 + payout_fraction))
    return BacktestReport(
        observations=len(trades),
        period_observations=period_count,
        gap_count=sum(_gap_count(segment) for segment in segments),
        net_pnl=net_sum,
        sharpe=sharpe,
        per_trade_mean_pnl=per_trade_mean,
        per_trade_std_pnl=per_trade_std,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        expectancy_per_trade=expectancy,
        break_even_win_rate=break_even,
        deflated_sharpe_ratio=dsr,
        trades=list(trades),
    )


def _validate_candle_sequence(candles: Sequence[Candle]) -> None:
    if any(candle.granularity <= 0 for candle in candles):
        raise ValueError("Candle granularity must be greater than zero")
    symbols = {candle.symbol for candle in candles}
    granularities = {candle.granularity for candle in candles}
    if len(symbols) > 1:
        raise ValueError("One backtest run must contain a single symbol")
    if len(granularities) > 1:
        raise ValueError("One backtest run must contain a single granularity")
    if any(left.epoch >= right.epoch for left, right in pairwise(candles)):
        raise ValueError("Candles must be strictly increasing by epoch")
    if any(
        float(candle.open) <= 0
        or float(candle.high) <= 0
        or float(candle.low) <= 0
        or float(candle.close) <= 0
        for candle in candles
    ):
        raise ValueError("Backtest candle prices must be greater than zero")


def _are_contiguous(left: Candle, right: Candle) -> bool:
    return right.epoch - left.epoch == left.granularity


def _gap_count(candles: Sequence[Candle]) -> int:
    return sum(
        max(0, ceil((right.epoch - left.epoch) / left.granularity) - 1)
        for left, right in pairwise(candles)
    )


def _time_aligned_pnl(trades: Sequence[TradeResult], candles: Sequence[Candle]) -> np.ndarray:
    """Return per-candle PnL, inserting zero-PnL gap bars."""
    if not candles:
        return np.asarray([], dtype=float)

    pnl_by_entry = {trade.entry_epoch: trade.net_pnl for trade in trades}
    period_pnl: list[float] = []
    for position, candle in enumerate(candles):
        if position:
            previous = candles[position - 1]
            missing = max(0, ceil((candle.epoch - previous.epoch) / previous.granularity) - 1)
            period_pnl.extend([0.0] * missing)
        period_pnl.append(pnl_by_entry.get(candle.epoch, 0.0))
    return np.asarray(period_pnl, dtype=float)


def _finite_1d(values: Sequence[float], name: str, minimum: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
