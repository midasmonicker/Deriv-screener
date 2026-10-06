"""Regression probes for quant and data-integrity audit findings."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from scipy.stats import binomtest

from deriv_vol_lab.backtest import ProductSpec, backtest, walk_forward_backtest
from deriv_vol_lab.data import Candle, Tick, build_quality_report
from deriv_vol_lab.features import screen_candles
from deriv_vol_lab.stats import evaluate_signal


def _candles(
    count: int,
    *,
    closes: list[float] | None = None,
    start_epoch: int = 100,
    granularity: int = 60,
) -> list[Candle]:
    close_values = closes or [100.0 + position for position in range(count)]
    return [
        Candle(
            symbol="R_75",
            epoch=start_epoch + position * granularity,
            open=Decimal("100"),
            high=Decimal(str(max(100.0, close_values[position]))),
            low=Decimal(str(min(100.0, close_values[position]))),
            close=Decimal(str(close_values[position])),
            granularity=granularity,
        )
        for position in range(count)
    ]


def _signal(candles: list[Candle], values: list[float] | None = None) -> pd.Series[float]:
    signal_values = values or [1.0] * len(candles)
    return pd.Series(
        signal_values,
        index=pd.Index([candle.epoch for candle in candles]),
        dtype=float,
    )


def test_pooled_walk_forward_does_not_count_overlapping_trades_twice() -> None:
    """Pooled OOS results reject overlapping folds while split generation allows them."""
    candles = _candles(12)
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )

    with pytest.raises(ValueError, match="step_size"):
        walk_forward_backtest(
            candles,
            _signal(candles),
            product,
            train_size=4,
            test_size=4,
            step_size=2,
        )


def test_sharpe_uses_annualized_candle_returns_not_trade_count_scaling() -> None:
    """The reported Sharpe matches the flat-inclusive per-candle annualized formula."""
    pnl_pattern = [1.0, -0.5, 2.0, -1.0, 0.0]
    single_candles = _candles(6, closes=[100.0, *(100.0 + value for value in pnl_pattern)])
    repeated_pnl = pnl_pattern * 4
    repeated_candles = _candles(21, closes=[100.0, *(100.0 + value for value in repeated_pnl)])
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )

    single = backtest(single_candles, _signal(single_candles), product)
    repeated = backtest(repeated_candles, _signal(repeated_candles), product)
    periods_per_year = 365 * 24 * 60 * 60 // 60
    single_returns = pd.Series([0.0, *pnl_pattern])
    repeated_returns = pd.Series([0.0, *repeated_pnl])
    expected_single = single_returns.mean() / single_returns.std(ddof=1) * periods_per_year**0.5
    expected_repeated = (
        repeated_returns.mean() / repeated_returns.std(ddof=1) * periods_per_year**0.5
    )

    assert single.period_observations == len(single_returns)
    assert repeated.period_observations == len(repeated_returns)
    assert single.sharpe == pytest.approx(expected_single)
    assert repeated.sharpe == pytest.approx(expected_repeated)


def test_trade_timestamps_use_candle_close_for_decision_and_exit() -> None:
    """A decision is recorded at its candle close, and exit at execution close."""
    candles = _candles(3, closes=[100.0, 101.0, 102.0], start_epoch=1_700_000_000)
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )

    trade = backtest(candles, _signal(candles), product).trades[0]

    assert trade.decision_epoch == candles[0].epoch + candles[0].granularity
    assert trade.exit_epoch == candles[1].epoch + candles[1].granularity


def test_backtest_splits_at_missing_candle_and_reports_gap_count() -> None:
    """A gap blocks cross-segment execution and appears in report metadata."""
    candles = _candles(60)
    candles[30:] = [
        candle.model_copy(update={"epoch": candle.epoch + candle.granularity})
        for candle in candles[30:]
    ]
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )

    report = backtest(candles, _signal(candles), product)
    screen = screen_candles(
        {"R_75": candles},
        latest_n=60,
        permutations=5,
        seed=1,
    )

    assert report.gap_count == 1
    assert report.observations == len(candles) - 2
    assert all(trade.entry_epoch == trade.decision_epoch for trade in report.trades)
    assert len(screen) == 1


def test_trade_won_flag_matches_net_win_rate_after_costs() -> None:
    """A gross winner made net-negative by fees is not a reported win."""
    candles = _candles(3, closes=[100.0, 101.0, 100.0])
    product = ProductSpec(
        contract_type="fixed_payout",
        stake=10,
        payout_fraction=0.5,
        commission_per_trade=6,
    )

    report = backtest(candles, _signal(candles), product)
    trade = report.trades[0]

    assert trade.gross_pnl > 0 > trade.net_pnl
    assert (trade.won, trade.gross_won, report.win_rate) == (False, True, 0.0)


def test_signal_binomial_test_excludes_zero_forward_return_trials() -> None:
    """The exact binomial calculation uses only non-tied signed returns."""
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 103.0], dtype=float)
    signal = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0], dtype=float)

    result = evaluate_signal(signal, close, horizons=(1,), permutations=19, seed=7)
    forward = close.shift(-1) / close - 1
    signed = signal * forward
    non_tied = signed.iloc[:-1][signed.iloc[:-1] != 0]
    expected_p_value = float(binomtest(int((non_tied > 0).sum()), len(non_tied), p=0.5).pvalue)

    assert len(non_tied) == 3
    assert result.horizons[0].tie_count == 1
    assert result.horizons[0].n_effective == 3
    assert result.horizons[0].binomial_p_value == pytest.approx(expected_p_value)


def test_quality_report_separates_tick_and_candle_series_for_symbol() -> None:
    """Ticks and candles for one symbol receive separate coverage series."""
    ticks = [
        Tick(symbol="R_75", epoch=epoch, quote=price)
        for epoch, price in ((0, 100), (2, 100), (4, 100), (6, 100))
    ]
    candle = Candle(
        symbol="R_75",
        epoch=60,
        open=100,
        high=101,
        low=99,
        close=100,
        granularity=60,
    )

    report = build_quality_report([*ticks, candle])

    assert {
        (coverage.record_kind, coverage.granularity, coverage.observed_records)
        for coverage in report.daily_coverage
    } == {("tick", 2, 4), ("candle", 60, 1)}


def test_quality_report_flags_jump_after_zero_variance_window() -> None:
    """A nonzero return after constant prices is not silently discarded."""
    records = [
        Tick(symbol="R_75", epoch=epoch, quote=price)
        for epoch, price in ((0, 100), (2, 100), (4, 100), (6, 100), (8, 110))
    ]

    report = build_quality_report(records, sigma_window=4)

    jump = next(issue for issue in report.abnormal_jumps if issue.epoch == 8)
    assert jump.local_sigma == 0
    assert jump.sigma_multiple is None
    assert jump.flag == "zero-variance-break"
