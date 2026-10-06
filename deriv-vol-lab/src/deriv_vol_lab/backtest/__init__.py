"""Backtesting package."""

from deriv_vol_lab.backtest.engine import (
    BacktestReport,
    ProductSpec,
    RealityCheckResult,
    TradeResult,
    WalkForwardReport,
    WalkForwardSplit,
    WalkForwardWindow,
    backtest,
    deflated_sharpe_ratio,
    walk_forward_backtest,
    walk_forward_splits,
    whites_reality_check,
)

__all__ = [
    "BacktestReport",
    "ProductSpec",
    "RealityCheckResult",
    "TradeResult",
    "WalkForwardReport",
    "WalkForwardSplit",
    "WalkForwardWindow",
    "backtest",
    "deflated_sharpe_ratio",
    "walk_forward_backtest",
    "walk_forward_splits",
    "whites_reality_check",
]
