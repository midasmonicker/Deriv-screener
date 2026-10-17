"""Forward-test ledger: record technical calls at candle close, score them later.

Every scheduled snapshot (1) scores earlier calls whose horizon has now closed and (2) records
new calls from the current screen. A call is fixed *before* its entry candle opens, and is scored
against the candles that follow, so the ledger is a genuine out-of-sample test. A ``coin`` control
that calls long/short by hash parity shows what pure chance looks like.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from deriv_vol_lab.api.snapshot import (
    AnalyticsSnapshot,
    ScreenSnapshotRow,
    default_snapshot_path,
)
from deriv_vol_lab.data.models import Candle

GRANULARITY_SECONDS = 60
HORIZON_CANDLES = 5
ASSUMED_PAYOUT = 0.95
PENDING_EXPIRY_SECONDS = 3 * 86_400
MIN_VERDICT_CALLS = 200
RECENT_LIMIT = 100
ALPHA = 0.05

Direction = Literal["long", "short"]
STRATEGIES: dict[str, tuple[str, str]] = {
    "trend": ("Trend follow", "Long when the 20-period average is above the 50, else short."),
    "reversion": (
        "Stretch reversion",
        "Fade price when it is 1+ standard deviations from its mean.",
    ),
    "rsi": ("RSI extremes", "Long when RSI is below 30, short when above 70."),
    "coin": ("Coin flip (control)", "Random direction every time. This is what chance looks like."),
}
REAL_STRATEGIES = tuple(name for name in STRATEGIES if name != "coin")


class Call(BaseModel):
    """A directional call fixed before its entry candle opens."""

    id: str
    symbol: str
    strategy: str
    direction: Direction
    signal_epoch: int
    recorded_epoch: int
    horizon: int = HORIZON_CANDLES


class ScoredCall(Call):
    """A call after its horizon closed."""

    outcome: Literal["win", "loss", "void"]
    entry_price: float | None = None
    exit_price: float | None = None


class Tally(BaseModel):
    """Running counts for one strategy."""

    calls: int = 0
    wins: int = 0
    voids: int = 0


class Ledger(BaseModel):
    """Persistent state. Bounded: pending calls, running tallies and a short recent list."""

    pending: list[Call] = Field(default_factory=list)
    tallies: dict[str, Tally] = Field(default_factory=dict)
    recent: list[ScoredCall] = Field(default_factory=list)
    first_recorded_epoch: int | None = None
    last_updated_epoch: int | None = None


class StrategyResult(BaseModel):
    """Scored performance of one strategy against the fixed-payout break-even."""

    strategy: str
    label: str
    description: str
    scored_calls: int
    wins: int
    hit_rate: float | None
    break_even: float
    p_value: float | None
    adjusted_p_value: float | None
    verdict: Literal["collecting data", "no edge detected", "edge detected", "control"]


class LedgerSummary(BaseModel):
    """Everything the dashboard needs about the ledger."""

    horizon_candles: int
    assumed_payout: float
    min_verdict_calls: int
    pending_calls: int
    first_recorded_epoch: int | None
    last_updated_epoch: int | None
    headline: str
    strategies: list[StrategyResult]
    recent: list[ScoredCall]


class LedgerUnavailableError(RuntimeError):
    """Raised when a ledger file exists but cannot be parsed."""


def default_ledger_path() -> Path:
    """The ledger lives beside the snapshot file."""
    override = os.environ.get("LEDGER_PATH")
    if override:
        return Path(override)
    return default_snapshot_path().with_name("ledger.json")


def load_ledger(path: Path) -> Ledger:
    """Load the ledger; a missing file is an empty ledger."""
    if not path.exists():
        return Ledger()
    try:
        return Ledger.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise LedgerUnavailableError(f"Ledger at {path} is unreadable: {exc}") from exc


def write_ledger(ledger: Ledger, path: Path) -> None:
    """Atomically persist the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".ledger-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(ledger.model_dump_json() + "\n")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def direction_for(strategy: str, row: ScreenSnapshotRow) -> Direction | None:
    """The call a strategy makes from a screen row, or None for no call."""
    if strategy == "trend":
        if row.ema20_50_trend.endswith("above_ema50"):
            return "long"
        if row.ema20_50_trend.endswith("below_ema50"):
            return "short"
        return None
    if strategy == "reversion":
        z = row.bollinger_z
        if z is None:
            return None
        if z <= -1.0:
            return "long"
        return "short" if z >= 1.0 else None
    if strategy == "rsi":
        if row.rsi is None:
            return None
        if row.rsi < 30.0:
            return "long"
        return "short" if row.rsi > 70.0 else None
    if strategy == "coin":
        digest = hashlib.sha256(f"{row.symbol}:{row.epoch}".encode()).digest()
        return "long" if digest[0] % 2 == 0 else "short"
    raise ValueError(f"unknown strategy: {strategy}")


def record_calls(ledger: Ledger, rows: Iterable[ScreenSnapshotRow], generated_at: int) -> int:
    """Register calls whose entry candle has not opened yet; return how many were added."""
    known = {call.id for call in ledger.pending} | {call.id for call in ledger.recent}
    added = 0
    for row in rows:
        if row.epoch + GRANULARITY_SECONDS <= generated_at:
            continue  # entry candle already open: recording now would be hindsight
        for strategy in STRATEGIES:
            direction = direction_for(strategy, row)
            if direction is None:
                continue
            call_id = f"{row.symbol}|{strategy}|{row.epoch}"
            if call_id in known:
                continue
            ledger.pending.append(
                Call(
                    id=call_id,
                    symbol=row.symbol,
                    strategy=strategy,
                    direction=direction,
                    signal_epoch=row.epoch,
                    recorded_epoch=generated_at,
                )
            )
            known.add(call_id)
            added += 1
    if added and ledger.first_recorded_epoch is None:
        ledger.first_recorded_epoch = generated_at
    return added


def _score(call: Call, index: Mapping[int, Candle], now: int) -> ScoredCall | None:
    entry_epoch = call.signal_epoch + GRANULARITY_SECONDS
    exit_epoch = call.signal_epoch + GRANULARITY_SECONDS * call.horizon
    if exit_epoch + GRANULARITY_SECONDS > now:
        return None  # horizon has not closed yet
    entry = index.get(entry_epoch)
    exit_ = index.get(exit_epoch)
    if entry is None or exit_ is None:
        return ScoredCall(**call.model_dump(), outcome="void")
    entry_price = float(entry.open)
    exit_price = float(exit_.close)
    moved = exit_price - entry_price
    won = moved > 0 if call.direction == "long" else moved < 0  # a flat result is a loss
    return ScoredCall(
        **call.model_dump(),
        outcome="win" if won else "loss",
        entry_price=entry_price,
        exit_price=exit_price,
    )


def score_pending(ledger: Ledger, candles_by_symbol: Mapping[str, list[Candle]], now: int) -> int:
    """Score every pending call whose horizon has closed; return how many were resolved."""
    indexes = {
        symbol: {candle.epoch: candle for candle in candles}
        for symbol, candles in candles_by_symbol.items()
    }
    still_pending: list[Call] = []
    resolved = 0
    for call in ledger.pending:
        result = _score(call, indexes.get(call.symbol, {}), now)
        if result is None and now - call.recorded_epoch > PENDING_EXPIRY_SECONDS:
            result = ScoredCall(**call.model_dump(), outcome="void")
        if result is None:
            still_pending.append(call)
            continue
        tally = ledger.tallies.setdefault(call.strategy, Tally())
        if result.outcome == "void":
            tally.voids += 1
        else:
            tally.calls += 1
            tally.wins += result.outcome == "win"
        ledger.recent.append(result)
        resolved += 1
    ledger.pending = still_pending
    ledger.recent = ledger.recent[-RECENT_LIMIT:]
    return resolved


def update_ledger(
    ledger: Ledger,
    snapshot: AnalyticsSnapshot,
    candles_by_symbol: Mapping[str, list[Candle]],
) -> Ledger:
    """Score what has closed, then record this snapshot's calls."""
    now = snapshot.generated_at_epoch
    score_pending(ledger, candles_by_symbol, now)
    record_calls(ledger, snapshot.screen, now)
    ledger.last_updated_epoch = now
    return ledger


def _p_value(wins: int, calls: int, break_even: float) -> float:
    if calls == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(wins, calls, break_even, alternative="greater").pvalue)


def summarise(ledger: Ledger, payout: float = ASSUMED_PAYOUT) -> LedgerSummary:
    """Turn raw tallies into per-strategy results with an honest verdict."""
    break_even = 1.0 / (1.0 + payout)
    results: list[StrategyResult] = []
    for strategy, (label, description) in STRATEGIES.items():
        tally = ledger.tallies.get(strategy, Tally())
        p_value = _p_value(tally.wins, tally.calls, break_even) if tally.calls else None
        adjusted = None if p_value is None else min(1.0, p_value * len(REAL_STRATEGIES))
        if strategy == "coin":
            verdict: Literal["collecting data", "no edge detected", "edge detected", "control"] = (
                "control"
            )
        elif tally.calls < MIN_VERDICT_CALLS:
            verdict = "collecting data"
        elif adjusted is not None and adjusted < ALPHA:
            verdict = "edge detected"
        else:
            verdict = "no edge detected"
        results.append(
            StrategyResult(
                strategy=strategy,
                label=label,
                description=description,
                scored_calls=tally.calls,
                wins=tally.wins,
                hit_rate=tally.wins / tally.calls if tally.calls else None,
                break_even=break_even,
                p_value=p_value,
                adjusted_p_value=adjusted,
                verdict=verdict,
            )
        )
    return LedgerSummary(
        horizon_candles=HORIZON_CANDLES,
        assumed_payout=payout,
        min_verdict_calls=MIN_VERDICT_CALLS,
        pending_calls=len(ledger.pending),
        first_recorded_epoch=ledger.first_recorded_epoch,
        last_updated_epoch=ledger.last_updated_epoch,
        headline=_headline(results),
        strategies=results,
        recent=ledger.recent[-20:][::-1],
    )


def _headline(results: list[StrategyResult]) -> str:
    real = [item for item in results if item.strategy in REAL_STRATEGIES]
    winners = [item.label for item in real if item.verdict == "edge detected"]
    if winners:
        return (
            "Statistically significant after correction: "
            + ", ".join(winners)
            + ". Treat as a lead to re-test, not as a trading signal."
        )
    scored = max((item.scored_calls for item in real), default=0)
    if scored < MIN_VERDICT_CALLS:
        return (
            f"Still collecting: {scored} of {MIN_VERDICT_CALLS} scored calls needed before any "
            "verdict. Hit rates this early are mostly luck."
        )
    return (
        "No call type has beaten the payout break-even after correction. "
        "Compare each hit rate with the coin-flip control."
    )
