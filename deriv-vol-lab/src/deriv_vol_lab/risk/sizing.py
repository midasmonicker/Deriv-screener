"""Position sizing from stated volatility and explicit contract risk."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict

from deriv_vol_lab.backtest.engine import ProductSpec
from deriv_vol_lab.data.symbols import SYMBOLS

SizingMethod = Literal["volatility_target", "fixed_fractional", "fractional_kelly"]


class PositionSize(BaseModel):
    """Sizing result; risk fraction is maximum modeled loss as a share of equity."""

    model_config = ConfigDict(frozen=True)

    method: SizingMethod
    stake: float
    equity_fraction: float
    risk_fraction: float
    requested_risk_fraction: float
    capped: bool
    expected_annual_volatility: float | None = None


def volatility_targeted_size(
    symbol: str,
    *,
    equity: float,
    target_annual_volatility: float,
    product: ProductSpec,
    max_risk_fraction: float = 0.02,
) -> PositionSize:
    """Size a linear multiplier contract to target annualized equity volatility.

    The known stated volatility is treated as the annualized standard deviation
    of log returns. This method is not meaningful for fixed-payout contracts.
    The output's risk fraction is separately capped using the product's
    stop-out and configured costs.
    """
    if symbol not in SYMBOLS:
        raise ValueError(f"Unsupported volatility symbol: {symbol}")
    _positive_finite(equity, "equity")
    _positive_finite(target_annual_volatility, "target_annual_volatility")
    _risk_cap(max_risk_fraction)
    if product.contract_type != "multiplier" or product.multiplier is None:
        raise ValueError("Volatility targeting requires a multiplier product")

    stated_volatility = SYMBOLS[symbol].stated_vol_pct / 100.0
    uncapped_stake = equity * target_annual_volatility / (stated_volatility * product.multiplier)
    stake, requested_risk_fraction, capped = _cap_stake(
        uncapped_stake,
        equity=equity,
        product=product,
        max_risk_fraction=max_risk_fraction,
    )
    return PositionSize(
        method="volatility_target",
        stake=stake,
        equity_fraction=stake / equity,
        risk_fraction=_risk_at_stake(stake, equity, product),
        requested_risk_fraction=requested_risk_fraction,
        capped=capped,
        expected_annual_volatility=(stake * product.multiplier * stated_volatility / equity),
    )


def fixed_fractional_size(
    *,
    equity: float,
    risk_fraction: float,
    product: ProductSpec,
    max_risk_fraction: float = 0.02,
) -> PositionSize:
    """Size stake so the modeled maximum loss risks a fixed share of equity."""
    _positive_finite(equity, "equity")
    _risk_cap(risk_fraction, allow_zero=False)
    _risk_cap(max_risk_fraction, allow_zero=False)
    requested = risk_fraction
    applied = min(requested, max_risk_fraction)
    stake = _stake_for_risk_budget(equity * applied, product)
    return PositionSize(
        method="fixed_fractional",
        stake=stake,
        equity_fraction=stake / equity,
        risk_fraction=_risk_at_stake(stake, equity, product),
        requested_risk_fraction=requested,
        capped=requested > max_risk_fraction,
    )


def fractional_kelly_size(
    *,
    equity: float,
    win_probability: float,
    net_win_loss_ratio: float,
    product: ProductSpec,
    kelly_fraction: float = 0.5,
    max_risk_fraction: float = 0.02,
) -> PositionSize:
    """Apply fractional Kelly to net outcomes, then cap maximum loss at equity.

    ``net_win_loss_ratio`` is the estimated net winning PnL divided by the
    absolute net losing PnL. It must be estimated out of sample; this function
    does not infer an edge from the synthetic null.
    """
    _positive_finite(equity, "equity")
    if not isfinite(win_probability) or not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be finite and between zero and one")
    _positive_finite(net_win_loss_ratio, "net_win_loss_ratio")
    if not isfinite(kelly_fraction) or not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be finite and in (0, 1]")
    _risk_cap(max_risk_fraction, allow_zero=False)

    full_kelly_risk_fraction = max(
        0.0,
        win_probability - (1.0 - win_probability) / net_win_loss_ratio,
    )
    requested = full_kelly_risk_fraction * kelly_fraction
    applied = min(requested, max_risk_fraction)
    if applied == 0:
        return PositionSize(
            method="fractional_kelly",
            stake=0.0,
            equity_fraction=0.0,
            risk_fraction=0.0,
            requested_risk_fraction=0.0,
            capped=False,
        )
    stake = _stake_for_risk_budget(equity * applied, product)
    return PositionSize(
        method="fractional_kelly",
        stake=stake,
        equity_fraction=stake / equity,
        risk_fraction=_risk_at_stake(stake, equity, product),
        requested_risk_fraction=requested,
        capped=requested > max_risk_fraction,
    )


def maximum_loss_for_stake(stake: float, product: ProductSpec) -> float:
    """Return the contract's maximum one-trade loss including stated costs."""
    _positive_finite(stake, "stake")
    if product.contract_type == "multiplier":
        if product.stop_out_fraction is None or product.multiplier is None:
            raise ValueError("Multiplier product is missing its stop-out terms")
        spread_loss = stake * product.multiplier * product.spread_fraction
        principal_loss = stake * product.stop_out_fraction
    else:
        principal_loss = stake
        spread_loss = stake * product.spread_fraction
    return principal_loss + spread_loss + product.commission_per_trade


def _stake_for_risk_budget(risk_budget: float, product: ProductSpec) -> float:
    loss_fraction = maximum_loss_for_stake(1.0, product) - product.commission_per_trade
    stake = (risk_budget - product.commission_per_trade) / loss_fraction
    if stake <= 0:
        raise ValueError("Risk budget must exceed commission_per_trade")
    return stake


def _cap_stake(
    uncapped_stake: float,
    *,
    equity: float,
    product: ProductSpec,
    max_risk_fraction: float,
) -> tuple[float, float, bool]:
    uncapped_risk = _risk_at_stake(uncapped_stake, equity, product)
    if uncapped_risk <= max_risk_fraction:
        return uncapped_stake, uncapped_risk, False
    return (
        _stake_for_risk_budget(equity * max_risk_fraction, product),
        uncapped_risk,
        True,
    )


def _risk_at_stake(stake: float, equity: float, product: ProductSpec) -> float:
    return maximum_loss_for_stake(stake, product) / equity


def _positive_finite(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _risk_cap(value: float, *, allow_zero: bool = True) -> None:
    lower_ok = value >= 0 if allow_zero else value > 0
    if not isfinite(value) or not lower_ok or value > 1:
        lower = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"Risk fraction must be finite and in {lower}")
