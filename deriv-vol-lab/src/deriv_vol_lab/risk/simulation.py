"""Seeded null-process risk simulation and cross-index portfolio diagnostics."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from math import isfinite, sqrt
from typing import TextIO

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from deriv_vol_lab.backtest.engine import ProductSpec
from deriv_vol_lab.data.symbols import SYMBOLS
from deriv_vol_lab.features.indicators import SECONDS_PER_YEAR

TRADE_HORIZON_SECONDS = 60
FloatArray = NDArray[np.float64]


class MonteCarloRiskReport(BaseModel):
    """Drawdown and ruin estimates from fixed-stake null-process paths."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    stake: float
    contract_type: str
    trade_count: int
    simulations: int
    seed: int
    initial_equity: float
    ruin_equity_fraction: float
    risk_of_ruin_probability: float
    worst_drawdown_p95: float
    worst_drawdown_p99: float
    mean_worst_drawdown: float
    median_terminal_equity: float


class PortfolioRiskReport(BaseModel):
    """Observed return correlations and independent-process portfolio risk."""

    model_config = ConfigDict(frozen=True)

    symbols: tuple[str, ...]
    observations: int
    weights: dict[str, float]
    annual_volatility_if_independent: float
    correlation_matrix: dict[str, dict[str, float]]
    max_absolute_pairwise_correlation: float
    correlation_warning_threshold: float
    independence_warning: bool


def monte_carlo_risk_report(
    symbol: str,
    *,
    stake: float,
    product: ProductSpec,
    trade_count: int,
    seed: int,
    simulations: int = 2_000,
    initial_equity: float = 10_000.0,
    ruin_equity_fraction: float = 0.5,
) -> MonteCarloRiskReport:
    """Simulate 60-second null trades and estimate risk of ruin and drawdown.

    Underlying prices follow driftless geometric Brownian motion at the
    registry's stated annualized volatility and tick cadence. One trade opens
    per 60-second interval with an independent random long/short direction.
    OHLC extrema are simulated from the symbol's one- or two-second increments,
    allowing the backtest's multiplier stop-out rule to be applied. Stake is
    fixed across trades. Ruin means equity reaches or falls below the configured
    fraction of initial equity, after which the path stops trading.

    Drawdown quantiles are quantiles of each path's maximum peak-to-trough
    equity drawdown. They are model outputs under a driftless null, not loss
    guarantees or a forecast of strategy performance.
    """
    if symbol not in SYMBOLS:
        raise ValueError(f"Unsupported volatility symbol: {symbol}")
    _positive_finite(stake, "stake")
    _positive_finite(initial_equity, "initial_equity")
    _positive_int(trade_count, "trade_count")
    _positive_int(simulations, "simulations")
    if not isfinite(ruin_equity_fraction) or not 0.0 <= ruin_equity_fraction < 1.0:
        raise ValueError("ruin_equity_fraction must be finite and in [0, 1)")

    spec = product.model_copy(update={"stake": stake})
    definition = SYMBOLS[symbol]
    if TRADE_HORIZON_SECONDS % definition.tick_interval_sec:
        raise ValueError("Symbol tick interval must divide the 60-second trade horizon")
    steps_per_trade = TRADE_HORIZON_SECONDS // definition.tick_interval_sec
    annual_volatility = definition.stated_vol_pct / 100.0
    dt = definition.tick_interval_sec / SECONDS_PER_YEAR
    step_scale = annual_volatility * sqrt(dt)
    step_drift = -0.5 * annual_volatility**2 * dt
    ruin_equity = initial_equity * ruin_equity_fraction
    generator = np.random.default_rng(seed)
    worst_drawdowns = np.empty(simulations, dtype=float)
    terminal_equities = np.empty(simulations, dtype=float)
    ruin_events = 0

    for simulation in range(simulations):
        increments = generator.normal(
            step_drift,
            step_scale,
            size=(trade_count, steps_per_trade),
        )
        log_path = np.cumsum(increments, axis=1)
        close_returns = np.expm1(log_path[:, -1])
        high_returns = np.expm1(np.maximum(log_path.max(axis=1), 0.0))
        low_returns = np.expm1(np.minimum(log_path.min(axis=1), 0.0))
        directions = generator.choice(np.asarray([-1.0, 1.0]), size=trade_count)
        pnls = _settle_null_trades(
            close_returns,
            high_returns,
            low_returns,
            directions,
            spec,
        )
        equity = initial_equity + np.cumsum(pnls)
        breaches = np.flatnonzero(equity <= ruin_equity)
        if breaches.size:
            ruin_events += 1
            first_breach = int(breaches[0])
            pnls[first_breach + 1 :] = 0.0
            equity = initial_equity + np.cumsum(pnls)
        peaks = np.maximum.accumulate(np.concatenate(([initial_equity], equity)))[1:]
        drawdowns = np.maximum(0.0, (peaks - equity) / peaks)
        worst_drawdowns[simulation] = float(drawdowns.max(initial=0.0))
        terminal_equities[simulation] = float(equity[-1])

    return MonteCarloRiskReport(
        symbol=symbol,
        stake=stake,
        contract_type=spec.contract_type,
        trade_count=trade_count,
        simulations=simulations,
        seed=seed,
        initial_equity=initial_equity,
        ruin_equity_fraction=ruin_equity_fraction,
        risk_of_ruin_probability=ruin_events / simulations,
        worst_drawdown_p95=float(np.quantile(worst_drawdowns, 0.95)),
        worst_drawdown_p99=float(np.quantile(worst_drawdowns, 0.99)),
        mean_worst_drawdown=float(worst_drawdowns.mean()),
        median_terminal_equity=float(np.median(terminal_equities)),
    )


def print_risk_report(
    symbol: str,
    *,
    stake: float,
    product: ProductSpec,
    trade_count: int,
    seed: int,
    simulations: int = 2_000,
    initial_equity: float = 10_000.0,
    ruin_equity_fraction: float = 0.5,
    file: TextIO | None = None,
) -> MonteCarloRiskReport:
    """Write a seeded risk summary including 95% and 99% worst drawdowns."""
    report = monte_carlo_risk_report(
        symbol,
        stake=stake,
        product=product,
        trade_count=trade_count,
        seed=seed,
        simulations=simulations,
        initial_equity=initial_equity,
        ruin_equity_fraction=ruin_equity_fraction,
    )
    output = sys.stdout if file is None else file
    output.write(
        f"Null risk report: {report.symbol}, {report.trade_count} 60-second trades, "
        f"stake={report.stake:g}, simulations={report.simulations}\n"
        f"Risk of ruin (equity <= {report.ruin_equity_fraction:.0%} of initial): "
        f"{report.risk_of_ruin_probability:.2%}\n"
        f"Expected worst drawdown quantile: 95%={report.worst_drawdown_p95:.2%}, "
        f"99%={report.worst_drawdown_p99:.2%}\n"
    )
    return report


def portfolio_risk_view(
    returns: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    correlation_warning_threshold: float = 0.1,
) -> PortfolioRiskReport:
    """Report observed cross-index return correlations and independent risk.

    The input is a time-aligned DataFrame of close-to-close returns, one column
    per registered index. Rows with any missing return are omitted; no values
    are forward-filled. The theoretical annual portfolio volatility assumes
    independent log-return processes, so it is the root-sum-square of weighted
    stated volatilities. The empirical matrix is reported as a diagnostic of
    finite-sample co-movement, not used to claim dependence from noise.
    """
    symbols = tuple(SYMBOLS)
    if set(returns.columns) != set(symbols) or len(returns.columns) != len(symbols):
        raise ValueError("returns must contain exactly one column for each registered symbol")
    if (
        isinstance(returns.index, pd.MultiIndex)
        or not pd.api.types.is_integer_dtype(returns.index.dtype)
        or not returns.index.is_monotonic_increasing
        or not returns.index.is_unique
    ):
        raise ValueError("returns index must contain unique, increasing UTC epoch seconds")
    if not isfinite(correlation_warning_threshold) or not (
        0.0 <= correlation_warning_threshold <= 1.0
    ):
        raise ValueError("correlation_warning_threshold must be finite and in [0, 1]")
    aligned = returns.loc[:, list(symbols)].astype(float).dropna(how="any")
    if len(aligned) < 2 or not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("returns need at least two complete finite observations")
    if (aligned.std(ddof=1) == 0).any():
        raise ValueError("each symbol's returns must have nonzero variance")

    selected_weights = (
        {symbol: 1.0 / len(symbols) for symbol in symbols}
        if weights is None
        else {symbol: float(weights.get(symbol, 0.0)) for symbol in symbols}
    )
    if weights is not None and set(weights) != set(symbols):
        raise ValueError("weights must contain exactly one value for each registered symbol")
    if any(not isfinite(value) or value < 0 for value in selected_weights.values()):
        raise ValueError("portfolio weights must be finite and non-negative")
    if not np.isclose(sum(selected_weights.values()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("portfolio weights must sum to one")

    correlation = aligned.corr()
    correlation_values: FloatArray = correlation.to_numpy(dtype=np.float64)
    off_diagonal = correlation_values[~np.eye(len(symbols), dtype=bool)]
    max_correlation = float(np.max(np.abs(off_diagonal)))
    annual_volatility = sqrt(
        sum(
            (selected_weights[symbol] * SYMBOLS[symbol].stated_vol_pct / 100.0) ** 2
            for symbol in symbols
        )
    )
    correlation_dict = {
        symbol: {
            other: float(correlation_values[row_index, column_index])
            for column_index, other in enumerate(symbols)
        }
        for row_index, symbol in enumerate(symbols)
    }
    return PortfolioRiskReport(
        symbols=symbols,
        observations=len(aligned),
        weights=selected_weights,
        annual_volatility_if_independent=annual_volatility,
        correlation_matrix=correlation_dict,
        max_absolute_pairwise_correlation=max_correlation,
        correlation_warning_threshold=correlation_warning_threshold,
        independence_warning=max_correlation > correlation_warning_threshold,
    )


def _settle_null_trades(
    close_returns: np.ndarray,
    high_returns: np.ndarray,
    low_returns: np.ndarray,
    directions: np.ndarray,
    product: ProductSpec,
) -> np.ndarray:
    if product.contract_type == "multiplier":
        multiplier = product.multiplier
        stop_out = product.stop_out_fraction
        if multiplier is None or stop_out is None:
            raise ValueError("Multiplier product is missing stop-out parameters")
        favorable_returns = directions * close_returns
        gross = product.stake * multiplier * favorable_returns
        adverse_returns = np.where(directions > 0, low_returns, high_returns) * directions
        stopped = product.stake * multiplier * adverse_returns <= -product.stake * stop_out
        gross[stopped] = -product.stake * stop_out
        notional = product.stake * multiplier
    else:
        payout = product.payout_fraction
        if payout is None:
            raise ValueError("Fixed-payout product is missing payout_fraction")
        wins = directions * close_returns > 0
        gross = np.where(wins, product.stake * payout, -product.stake)
        notional = product.stake
    costs = product.commission_per_trade + notional * product.spread_fraction
    net_pnl: FloatArray = np.asarray(gross - costs, dtype=np.float64)
    return net_pnl


def _positive_finite(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
