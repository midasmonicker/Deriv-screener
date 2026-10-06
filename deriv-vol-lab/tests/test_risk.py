"""Known-answer tests for risk sizing, loss controls, and null simulations."""

from __future__ import annotations

from datetime import UTC, datetime
from math import sqrt

import numpy as np
import pandas as pd
import pytest

from deriv_vol_lab.backtest import ProductSpec
from deriv_vol_lab.data.symbols import SYMBOLS
from deriv_vol_lab.risk import (
    LossCircuitBreaker,
    fixed_fractional_size,
    fractional_kelly_size,
    maximum_loss_for_stake,
    monte_carlo_risk_report,
    portfolio_risk_view,
    print_risk_report,
    volatility_targeted_size,
)


def _multiplier_product(
    *,
    stake: float = 10.0,
    multiplier: float = 2.0,
    stop_out_fraction: float = 0.5,
    commission_per_trade: float = 0.0,
    spread_fraction: float = 0.0,
) -> ProductSpec:
    return ProductSpec(
        contract_type="multiplier",
        stake=stake,
        multiplier=multiplier,
        stop_out_fraction=stop_out_fraction,
        commission_per_trade=commission_per_trade,
        spread_fraction=spread_fraction,
    )


def test_volatility_targeted_size_matches_stated_volatility() -> None:
    """R_50 with multiplier two needs 1,000 stake for 10% annualized equity vol."""
    result = volatility_targeted_size(
        "R_50",
        equity=10_000,
        target_annual_volatility=0.10,
        product=_multiplier_product(stop_out_fraction=1),
        max_risk_fraction=0.1,
    )

    assert result.stake == pytest.approx(1_000)
    assert result.equity_fraction == pytest.approx(0.1)
    assert result.expected_annual_volatility == pytest.approx(0.10)
    assert result.risk_fraction == pytest.approx(0.1)
    assert not result.capped


def test_volatility_targeted_size_applies_max_risk_cap() -> None:
    """The configured maximum modeled loss caps a high-volatility target."""
    result = volatility_targeted_size(
        "R_50",
        equity=10_000,
        target_annual_volatility=0.10,
        product=_multiplier_product(stop_out_fraction=1),
        max_risk_fraction=0.02,
    )

    assert result.stake == pytest.approx(200)
    assert result.risk_fraction == pytest.approx(0.02)
    assert result.expected_annual_volatility == pytest.approx(0.02)
    assert result.capped


def test_volatility_targeting_rejects_fixed_payout_products() -> None:
    fixed_payout = ProductSpec(contract_type="fixed_payout", stake=10, payout_fraction=0.8)
    with pytest.raises(ValueError, match="multiplier product"):
        volatility_targeted_size(
            "R_75",
            equity=1_000,
            target_annual_volatility=0.1,
            product=fixed_payout,
        )


def test_maximum_loss_and_fixed_fractional_sizing_known_answer() -> None:
    """Maximum loss includes stop-out, spread, and fixed commission costs."""
    product = _multiplier_product(
        stake=10,
        multiplier=2,
        stop_out_fraction=0.5,
        commission_per_trade=0.2,
        spread_fraction=0.001,
    )

    assert maximum_loss_for_stake(10, product) == pytest.approx(5.22)
    result = fixed_fractional_size(
        equity=1_000,
        risk_fraction=0.01,
        product=product,
        max_risk_fraction=0.02,
    )

    expected_stake = (10 - 0.2) / (0.5 + 2 * 0.001)
    assert result.stake == pytest.approx(expected_stake)
    assert result.risk_fraction == pytest.approx(0.01)
    assert not result.capped


def test_fixed_fractional_size_caps_requested_risk() -> None:
    result = fixed_fractional_size(
        equity=1_000,
        risk_fraction=0.1,
        product=_multiplier_product(),
        max_risk_fraction=0.02,
    )

    assert result.stake == pytest.approx(40)
    assert result.risk_fraction == pytest.approx(0.02)
    assert result.capped


def test_fractional_kelly_uses_net_payoff_ratio_and_cap() -> None:
    """Half Kelly for p=.6 and net reward/risk 1.5 is capped at 2% equity risk."""
    result = fractional_kelly_size(
        equity=1_000,
        win_probability=0.6,
        net_win_loss_ratio=1.5,
        product=_multiplier_product(stop_out_fraction=1),
        kelly_fraction=0.5,
        max_risk_fraction=0.02,
    )

    assert result.requested_risk_fraction == pytest.approx(1 / 6)
    assert result.stake == pytest.approx(20)
    assert result.risk_fraction == pytest.approx(0.02)
    assert result.capped


def test_fractional_kelly_returns_zero_for_nonpositive_edge() -> None:
    result = fractional_kelly_size(
        equity=1_000,
        win_probability=0.4,
        net_win_loss_ratio=1,
        product=_multiplier_product(),
    )

    assert result.stake == 0
    assert result.risk_fraction == 0


def test_loss_circuit_breaker_daily_reset_and_loss_streak() -> None:
    """Daily losses reset at UTC midnight but loss-streak halts need manual reset."""
    midnight = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp())
    breaker = LossCircuitBreaker(daily_loss_limit=10, max_consecutive_losses=3)

    assert breaker.can_trade(midnight - 2)
    breaker.record_trade(midnight - 2, -6)
    daily_halt = breaker.record_trade(midnight - 1, -4)
    assert daily_halt.halted
    assert daily_halt.reasons == ("daily-loss-limit",)
    assert breaker.can_trade(midnight)
    streak_halt = breaker.record_trade(midnight + 1, -1)
    assert streak_halt.halted
    assert streak_halt.reasons == ("consecutive-loss-limit",)

    breaker.reset()
    assert breaker.can_trade(midnight + 2)


def test_loss_circuit_breaker_rejects_invalid_epoch_and_pnl() -> None:
    breaker = LossCircuitBreaker(daily_loss_limit=10, max_consecutive_losses=2)
    with pytest.raises(ValueError, match="epoch"):
        breaker.can_trade(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="net_pnl"):
        breaker.record_trade(0, float("nan"))


def test_null_risk_report_is_seeded_and_drawdown_quantiles_are_ordered() -> None:
    product = _multiplier_product(
        stake=1,
        multiplier=1,
        stop_out_fraction=0.5,
        commission_per_trade=0.002,
        spread_fraction=0.001,
    )
    arguments = {
        "stake": 1,
        "product": product,
        "trade_count": 80,
        "seed": 707,
        "simulations": 100,
        "initial_equity": 1_000,
    }

    first = monte_carlo_risk_report("1HZ75V", **arguments)
    second = monte_carlo_risk_report("1HZ75V", **arguments)

    assert first == second
    assert first.worst_drawdown_p95 <= first.worst_drawdown_p99
    assert 0 <= first.risk_of_ruin_probability <= 1
    assert first.worst_drawdown_p95 == pytest.approx(0.0002577898699833213)
    assert first.median_terminal_equity == pytest.approx(999.7579996721089)


def test_null_risk_report_models_fixed_payout_contract_costs() -> None:
    """Fixed-payout branches account for stake loss, payout, and commission."""
    product = ProductSpec(
        contract_type="fixed_payout",
        stake=1,
        payout_fraction=0.8,
        commission_per_trade=2,
    )

    report = monte_carlo_risk_report(
        "R_75",
        stake=1,
        product=product,
        trade_count=3,
        seed=51,
        simulations=10,
        initial_equity=100,
        ruin_equity_fraction=0.5,
    )

    assert report.median_terminal_equity < 100
    assert report.risk_of_ruin_probability == 0
    assert report.worst_drawdown_p95 > 0


def test_risk_report_prints_requested_tail_drawdowns(capsys: pytest.CaptureFixture[str]) -> None:
    report = print_risk_report(
        "R_75",
        stake=1,
        product=_multiplier_product(stake=1, multiplier=1),
        trade_count=10,
        seed=8,
        simulations=10,
    )

    output = capsys.readouterr().out
    assert "95%=" in output and "99%=" in output
    assert report.symbol == "R_75"


def test_portfolio_view_reports_correlations_and_independent_volatility() -> None:
    """Perfectly co-moving observations are reported against independent risk."""
    base_returns = np.arange(1.0, 13.0)
    returns = pd.DataFrame(
        {symbol: base_returns * (position + 1) for position, symbol in enumerate(SYMBOLS)},
        index=pd.Index(
            [1_700_000_000 + position * 60 for position in range(base_returns.size)],
            name="epoch",
        ),
    )
    weights = {symbol: 1 / len(SYMBOLS) for symbol in SYMBOLS}

    report = portfolio_risk_view(returns, weights=weights, correlation_warning_threshold=0.1)

    stated_vols = np.asarray([item.stated_vol_pct / 100 for item in SYMBOLS.values()])
    expected_volatility = sqrt(float(np.sum((stated_vols / len(SYMBOLS)) ** 2)))
    assert report.observations == 12
    assert report.annual_volatility_if_independent == pytest.approx(expected_volatility)
    assert report.max_absolute_pairwise_correlation == pytest.approx(1)
    assert report.correlation_matrix["R_10"]["1HZ100V"] == pytest.approx(1)
    assert report.independence_warning


def test_portfolio_view_rejects_missing_symbols_and_bad_weights() -> None:
    returns = pd.DataFrame(
        {symbol: np.arange(3.0) for symbol in SYMBOLS},
        index=pd.Index([1_700_000_000, 1_700_000_060, 1_700_000_120], name="epoch"),
    )
    incomplete = returns.drop(columns=["R_10"])
    with pytest.raises(ValueError, match="exactly one column"):
        portfolio_risk_view(incomplete)
    bad_weights = {symbol: 0.1 for symbol in SYMBOLS}
    bad_weights["R_10"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        portfolio_risk_view(returns, weights=bad_weights)

    returns.index = pd.Index([1_700_000_000.0, 1_700_000_060.0, 1_700_000_120.0])
    with pytest.raises(ValueError, match="UTC epoch seconds"):
        portfolio_risk_view(returns)
