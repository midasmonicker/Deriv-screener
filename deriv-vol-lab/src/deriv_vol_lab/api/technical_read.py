"""Plain-language technical read of the latest screen. Descriptive, not a forecast."""

from __future__ import annotations

from pydantic import BaseModel

from deriv_vol_lab.api.ledger import REAL_STRATEGIES, Direction, direction_for
from deriv_vol_lab.api.snapshot import AnalyticsSnapshot, ScreenSnapshotRow

DISCLAIMER = (
    "Describes what already happened. The indices are random processes, so none of this "
    "predicts the next move; see the prediction ledger for the live test."
)


class CurrentCall(BaseModel):
    """What a ledger strategy would call right now."""

    strategy: str
    direction: Direction


class TechnicalRead(BaseModel):
    """One symbol's read."""

    symbol: str
    price: float | None
    trend: str
    momentum: str
    stretch: str
    volatility: str
    behaviour: str
    headline: str
    calls: list[CurrentCall]


class TechnicalReadResponse(BaseModel):
    """Technical reads for every symbol in the snapshot."""

    generated_at_epoch: int
    disclaimer: str
    reads: list[TechnicalRead]


def _trend(row: ScreenSnapshotRow) -> str:
    if row.ema20_50_trend.endswith("above_ema50"):
        return "up (short average above long)"
    if row.ema20_50_trend.endswith("below_ema50"):
        return "down (short average below long)"
    return "unclear"


def _momentum(rsi: float | None) -> str:
    if rsi is None:
        return "n/a"
    if rsi < 30:
        return f"oversold (RSI {rsi:.0f})"
    if rsi > 70:
        return f"overbought (RSI {rsi:.0f})"
    return f"neutral (RSI {rsi:.0f})"


def _stretch(z: float | None) -> str:
    if z is None:
        return "n/a"
    if z >= 2:
        return f"far above average (z {z:+.1f})"
    if z >= 1:
        return f"above average (z {z:+.1f})"
    if z <= -2:
        return f"far below average (z {z:+.1f})"
    if z <= -1:
        return f"below average (z {z:+.1f})"
    return f"near average (z {z:+.1f})"


def _volatility(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    if ratio > 1.15:
        return f"running hotter than stated ({ratio:.2f}x)"
    if ratio < 0.85:
        return f"running cooler than stated ({ratio:.2f}x)"
    return f"as stated ({ratio:.2f}x)"


def _behaviour(row: ScreenSnapshotRow) -> str:
    hurst, vr, ac = row.hurst, row.variance_ratio, row.rolling_autocorrelation
    if hurst is None or vr is None or ac is None:
        return "n/a"
    if 0.45 <= hurst <= 0.55 and 0.85 <= vr <= 1.15 and abs(ac) < 0.1:
        return "consistent with a random walk"
    return "small deviation from random walk (usually noise)"


def read_row(row: ScreenSnapshotRow) -> TechnicalRead:
    """Build the read for one screen row."""
    trend, momentum, stretch = _trend(row), _momentum(row.rsi), _stretch(row.bollinger_z)
    calls = [
        CurrentCall(strategy=name, direction=direction)
        for name in REAL_STRATEGIES
        if (direction := direction_for(name, row)) is not None
    ]
    return TechnicalRead(
        symbol=row.symbol,
        price=row.price,
        trend=trend,
        momentum=momentum,
        stretch=stretch,
        volatility=_volatility(row.vol_ratio),
        behaviour=_behaviour(row),
        headline=f"Trend {trend}; momentum {momentum}; price {stretch}.",
        calls=calls,
    )


def build_technical_read(snapshot: AnalyticsSnapshot) -> TechnicalReadResponse:
    """Technical reads for the whole snapshot."""
    return TechnicalReadResponse(
        generated_at_epoch=snapshot.generated_at_epoch,
        disclaimer=DISCLAIMER,
        reads=[read_row(row) for row in snapshot.screen],
    )
