"""Event-driven backtest, walk-forward, and selection-bias tests."""

from __future__ import annotations

from decimal import Decimal
from itertools import product as cartesian_product
from math import e, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from deriv_vol_lab.backtest import (
    ProductSpec,
    backtest,
    deflated_sharpe_ratio,
    walk_forward_backtest,
    walk_forward_splits,
    whites_reality_check,
)
from deriv_vol_lab.data import Candle
from deriv_vol_lab.stats import simulate_driftless_gbm


def _candles(
    opens: list[float],
    closes: list[float],
    *,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
    start: int = 100,
    granularity: int = 60,
) -> list[Candle]:
    lows = closes if lows is None else lows
    highs = closes if highs is None else highs
    return [
        Candle(
            symbol="R_75",
            epoch=start + position * granularity,
            open=Decimal(str(open_price)),
            high=Decimal(str(max(highs[position], open_price, close_price))),
            low=Decimal(str(min(lows[position], open_price, close_price))),
            close=Decimal(str(close_price)),
            granularity=granularity,
        )
        for position, (open_price, close_price) in enumerate(zip(opens, closes, strict=True))
    ]


def _signals(candles: list[Candle], values: list[float]) -> pd.Series[float]:
    return pd.Series(values, index=pd.Index([candle.epoch for candle in candles]))


def _aggregate_one_second_ticks_to_candles(prices: np.ndarray) -> list[Candle]:
    """Aggregate one-second prices into complete, valid 60-second candles."""
    granularity = 60
    candle_count = (prices.size - 1) // granularity
    candles: list[Candle] = []
    for position in range(candle_count):
        start = position * granularity
        end = start + granularity
        interval_prices = prices[start : end + 1]
        candles.append(
            Candle(
                symbol="1HZ75V",
                epoch=start,
                open=Decimal(str(prices[start])),
                high=Decimal(str(float(interval_prices.max()))),
                low=Decimal(str(float(interval_prices.min()))),
                close=Decimal(str(prices[end])),
                granularity=granularity,
            )
        )
    return candles


def test_candle_model_rejects_subminute_granularity() -> None:
    """One-second samples cannot be labeled as a Deriv OHLC candle."""
    with pytest.raises(ValueError, match="at least 60 seconds"):
        Candle(
            symbol="1HZ75V",
            epoch=0,
            open=100,
            high=101,
            low=99,
            close=100,
            granularity=1,
        )


def test_multiplier_contract_costs_stopout_and_report_metrics() -> None:
    """Multiplier settlement caps intrabar loss and deducts configurable costs."""
    candles = _candles(
        [100, 100, 100, 100],
        [100, 101, 99, 100],
        lows=[100, 99, 80, 100],
        highs=[100, 102, 200, 100],
    )
    signal = _signals(candles, [1, -1, 1, 0])
    product = ProductSpec(
        contract_type="multiplier",
        stake=10,
        multiplier=5,
        stop_out_fraction=0.5,
        commission_per_trade=0.1,
        spread_fraction=0.002,
    )

    result = backtest(candles, signal, product)

    assert result.observations == 3
    assert result.trades[0].decision_epoch == candles[0].epoch + candles[0].granularity
    assert result.trades[0].entry_epoch == candles[1].epoch
    assert result.trades[0].exit_epoch == candles[1].epoch + candles[1].granularity
    assert result.trades[0].net_pnl == pytest.approx(0.3)
    assert result.trades[1].gross_pnl == pytest.approx(-5)
    assert result.trades[1].exit_reason == "stop_out"
    assert result.trades[1].net_pnl == pytest.approx(-5.2)
    assert result.break_even_win_rate is None
    assert result.net_pnl == pytest.approx(sum(trade.net_pnl for trade in result.trades))
    assert result.expectancy_per_trade == pytest.approx(result.net_pnl / 3)
    assert result.max_drawdown > 0
    assert 0 <= result.deflated_sharpe_ratio <= 1


def test_six_candle_sharpe_and_drawdown_known_answer() -> None:
    """Annualized Sharpe includes flat periods; drawdown uses the equity peak."""
    candles = _candles(
        [100, 100, 100, 100, 100, 100],
        [100, 101, 99, 102, 100, 100],
    )
    signal = _signals(candles, [1, 1, -1, 0, 0, 0])
    product = ProductSpec(
        contract_type="multiplier",
        stake=100,
        multiplier=1,
        stop_out_fraction=1,
    )

    report = backtest(candles, signal, product, initial_equity=1_000)
    period_returns = np.asarray([0.0, 0.01, -0.01, -0.02, 0.0, 0.0])
    expected_sharpe = (
        period_returns.mean() / period_returns.std(ddof=1) * np.sqrt(365 * 24 * 60 * 60 / 60)
    )

    assert report.period_observations == 6
    assert report.sharpe == pytest.approx(expected_sharpe)
    assert report.max_drawdown == pytest.approx(3 / 1_001)
    assert report.per_trade_mean_pnl == pytest.approx(-2 / 3)
    assert report.per_trade_std_pnl == pytest.approx(np.std([1, -1, -2], ddof=1))


def test_fixed_payout_settlement_and_break_even_win_rate() -> None:
    """Sub-100% payout includes fee effects in break-even hit rate."""
    candles = _candles([10, 10, 10, 10], [10, 11, 9, 10])
    signal = _signals(candles, [1, -1, 1, 1])
    product = ProductSpec(
        contract_type="fixed_payout",
        stake=20,
        payout_fraction=0.8,
        commission_per_trade=0.2,
        spread_fraction=0.01,
    )

    result = backtest(candles, signal, product)

    assert [trade.gross_pnl for trade in result.trades] == [16, 16, -20]
    assert all(trade.net_pnl == pytest.approx(trade.gross_pnl - 0.4) for trade in result.trades)
    assert result.break_even_win_rate == pytest.approx(20.4 / 36)
    assert result.win_rate == pytest.approx(2 / 3)


def test_next_open_execution_prevents_lookahead_and_requires_epoch_index() -> None:
    """A final-candle signal cannot execute and future values do not alter earlier trades."""
    candles = _candles([100, 100, 110, 100], [100, 110, 100, 120])
    signal = _signals(candles, [1, -1, 1, 1])
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )
    baseline = backtest(candles, signal, product)
    altered = signal.copy()
    altered.iloc[-1] = -1
    changed = backtest(candles, altered, product)

    assert len(baseline.trades) == 3
    assert [trade.net_pnl for trade in baseline.trades] == [
        trade.net_pnl for trade in changed.trades
    ]
    with pytest.raises(ValueError, match="signal index"):
        backtest(candles, pd.Series([1, 1, 1, 1]), product)


@pytest.mark.parametrize("epochs", [[100, 100, 220], [100, 220, 160]])
def test_backtest_rejects_duplicate_or_non_increasing_epochs(epochs: list[int]) -> None:
    """Duplicate and decreasing UTC candle opens are invalid, unlike gaps."""
    candles = _candles([100, 100, 100], [100, 100, 100])
    candles = [
        candle.model_copy(update={"epoch": epoch})
        for candle, epoch in zip(candles, epochs, strict=True)
    ]
    signal = _signals(candles, [1, 1, 1])
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        backtest(candles, signal, product)


def test_walk_forward_splits_and_out_of_sample_only_reporting() -> None:
    """Training trades are excluded and only events executed in test windows count."""
    splits = walk_forward_splits(10, train_size=4, test_size=3, step_size=2)
    assert [(split.train_end, split.test_start, split.test_end) for split in splits] == [
        (4, 4, 7),
        (6, 6, 9),
        (8, 8, 10),
    ]

    candles = _candles(
        [100] * 10,
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    )
    signal = _signals(candles, [1] * 10)
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=1,
    )
    report = walk_forward_backtest(
        candles,
        signal,
        product,
        train_size=4,
        test_size=3,
        step_size=3,
    )

    assert [window.report.observations for window in report.windows] == [3, 3]
    assert report.out_of_sample.observations == 6
    assert all(trade.entry_epoch >= candles[4].epoch for trade in report.out_of_sample.trades)


def test_deflated_sharpe_accounts_for_more_strategy_trials() -> None:
    """The same observed Sharpe is less convincing after searching more trials."""
    returns = [0.01, -0.002, 0.008, -0.001, 0.012, -0.003, 0.007, 0.002]
    one_trial = deflated_sharpe_ratio(returns, strategy_trials=1)
    many_trials = deflated_sharpe_ratio(returns, strategy_trials=100)

    assert 0 <= many_trials < one_trial <= 1
    assert deflated_sharpe_ratio([0.1, 0.1, 0.1], strategy_trials=5) == 0.5


def test_deflated_sharpe_matches_published_reference_formula() -> None:
    """The implementation matches the Bailey-López de Prado DSR expression."""
    returns = np.asarray([0.02, -0.01, 0.03, -0.015, 0.01, 0.025, -0.005, 0.012])
    trials = 12
    observed_sharpe = float(returns.mean() / returns.std(ddof=1))
    centered = returns - returns.mean()
    skewness = float(np.mean(centered**3) / np.mean(centered**2) ** 1.5)
    kurtosis = float(np.mean(centered**4) / np.mean(centered**2) ** 2)
    euler_mascheroni = 0.5772156649015329
    expected_max = (1 - euler_mascheroni) * norm.ppf(1 - 1 / trials) + (
        euler_mascheroni * norm.ppf(1 - 1 / (trials * e))
    )
    standard_error = sqrt(
        (1 - skewness * observed_sharpe + (kurtosis - 1) * observed_sharpe**2 / 4)
        / (returns.size - 1)
    )
    expected_probability = float(norm.cdf((observed_sharpe - expected_max) / standard_error))

    # Bailey & López de Prado (2014), "The Deflated Sharpe Ratio," SSRN 2460551.
    assert deflated_sharpe_ratio(returns.tolist(), strategy_trials=trials) == pytest.approx(
        expected_probability
    )


def test_whites_reality_check_is_reproducible_and_detects_null_family() -> None:
    """Centered zero-return strategies give a non-significant WRC p-value."""
    returns = pd.DataFrame(
        {
            "candidate_a": [0.01, -0.01, 0.02, -0.02] * 10,
            "candidate_b": [-0.01, 0.01, -0.02, 0.02] * 10,
        }
    )
    first = whites_reality_check(returns, bootstrap_count=199, seed=17, block_length=2)
    second = whites_reality_check(returns, bootstrap_count=199, seed=17, block_length=2)

    assert first == second
    assert first.observed_max_mean == pytest.approx(0)
    assert first.p_value >= 0.05
    with pytest.raises(ValueError, match="finite"):
        whites_reality_check(pd.DataFrame({"bad": [0.1, np.nan]}), bootstrap_count=9, seed=1)


def test_whites_reality_check_matches_enumerated_reference_bootstrap() -> None:
    """WRC p-value matches the max-mean bootstrap formula by exact enumeration."""
    values = np.asarray(
        [
            [0.04, -0.01],
            [-0.02, 0.03],
            [0.01, -0.02],
            [-0.01, 0.02],
        ],
        dtype=float,
    )
    means = values.mean(axis=0)
    observed_max_mean = float(means.max())
    centered = values - means
    bootstrap_maxima = []
    for sample_indices in cartesian_product(range(values.shape[0]), repeat=values.shape[0]):
        bootstrap_maxima.append(float(centered[list(sample_indices)].mean(axis=0).max()))
    exact_p_value = float(np.mean(np.asarray(bootstrap_maxima) >= observed_max_mean))

    result = whites_reality_check(
        pd.DataFrame(values, columns=["strategy_a", "strategy_b"]),
        bootstrap_count=20_000,
        seed=718,
        block_length=1,
    )

    # White (2000), "A Reality Check for Data Snooping," Econometrica:
    # bootstrap the maximum mean performance differential under the null.
    assert result.observed_max_mean == pytest.approx(observed_max_mean)
    assert result.p_value == pytest.approx(exact_p_value, abs=0.015)


def test_random_strategy_on_seeded_gbm_has_nonpositive_net_expectancy() -> None:
    """Random strategies on aggregated 1HZ75V null candles lose after costs."""
    simulation_seeds = (20261006, 20261007, 20261008, 20261009, 20261010)
    ticks_per_path = 120_000
    candle_returns: list[float] = []
    expectancies: list[float] = []
    product = ProductSpec(
        contract_type="multiplier",
        stake=1,
        multiplier=1,
        stop_out_fraction=0.9,
        commission_per_trade=0.002,
        spread_fraction=0.002,
    )

    for seed in simulation_seeds:
        path = simulate_driftless_gbm(
            "1HZ75V",
            n_paths=1,
            n_steps=ticks_per_path,
            seed=seed,
        )[0]
        candles = _aggregate_one_second_ticks_to_candles(path)
        assert candles and all(candle.granularity == 60 for candle in candles)
        close = np.asarray([float(candle.close) for candle in candles])
        candle_returns.extend(np.diff(np.log(close)).tolist())

        generator = np.random.default_rng(seed + 1_000)
        directions = generator.choice(np.asarray([-1.0, 1.0]), size=len(candles))
        signals = _signals(candles, directions.tolist())
        report = backtest(candles, signals, product)
        assert report.observations > 1_000
        expectancies.append(report.expectancy_per_trade)

    seconds_per_year = 365 * 24 * 60 * 60
    realized_annual_volatility = float(np.std(candle_returns, ddof=1) * sqrt(seconds_per_year / 60))
    # Five 2,000-candle samples make +/- 4% a conservative Monte Carlo tolerance.
    assert realized_annual_volatility == pytest.approx(0.75, abs=0.03)
    mean_expectancy = float(np.mean(expectancies))
    # Permit 0.0005 stake units for finite-sample Monte Carlo variation; costs are 0.004.
    assert mean_expectancy <= 0.0005


def test_product_spec_and_walk_forward_validation() -> None:
    """Product-specific terms and invalid split dimensions are checked."""
    with pytest.raises(ValueError, match="payout_fraction"):
        ProductSpec(contract_type="fixed_payout", stake=1, payout_fraction=1)
    with pytest.raises(ValueError, match="stop_out_fraction"):
        ProductSpec(
            contract_type="multiplier",
            stake=1,
            multiplier=2,
            stop_out_fraction=0,
        )
    with pytest.raises(ValueError, match="train_size"):
        walk_forward_splits(10, train_size=0, test_size=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stake", float("nan")),
        ("stake", float("inf")),
        ("stake", float("-inf")),
        ("commission_per_trade", float("nan")),
        ("commission_per_trade", float("inf")),
        ("commission_per_trade", float("-inf")),
        ("spread_fraction", float("nan")),
        ("spread_fraction", float("inf")),
        ("spread_fraction", float("-inf")),
        ("multiplier", float("nan")),
        ("multiplier", float("inf")),
        ("multiplier", float("-inf")),
    ],
)
def test_product_spec_rejects_non_finite_terms(field: str, value: float) -> None:
    """Every requested numeric contract term must be finite."""
    terms: dict[str, float | str] = {
        "contract_type": "multiplier",
        "stake": 1.0,
        "multiplier": 2.0,
        "stop_out_fraction": 0.5,
    }
    terms[field] = value

    with pytest.raises(ValueError, match="finite"):
        ProductSpec.model_validate(terms)
