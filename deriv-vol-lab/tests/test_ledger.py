"""Ledger rules: calls are fixed before entry, scored out-of-sample, and honestly summarised."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from deriv_vol_lab.api import ledger as ledger_module
from deriv_vol_lab.api.ledger import (
    HORIZON_CANDLES,
    MIN_VERDICT_CALLS,
    Call,
    Ledger,
    LedgerUnavailableError,
    Tally,
    direction_for,
    load_ledger,
    record_calls,
    score_pending,
    summarise,
    write_ledger,
)
from deriv_vol_lab.api.snapshot import ScreenSnapshotRow
from deriv_vol_lab.api.technical_read import build_technical_read, read_row
from deriv_vol_lab.data import Candle

T0 = 1_700_000_000 - 1_700_000_000 % 60


def _row(**overrides: object) -> ScreenSnapshotRow:
    values: dict[str, object] = {
        "symbol": "R_10",
        "epoch": T0,
        "calibration_candles": 1,
        "evaluation_candles": 1,
        "display_candles": 1,
        "price": 100.0,
        "rsi": 50.0,
        "bollinger_z": 0.0,
        "ema20": 1.0,
        "ema50": 1.0,
        "ema20_50_trend": "ema20_above_ema50",
        "realised_vol": 0.1,
        "stated_vol": 0.1,
        "vol_ratio": 1.0,
        "rolling_autocorrelation": 0.0,
        "hurst": 0.5,
        "variance_ratio": 1.0,
        "flags": [],
        "reliability": {},
    }
    values.update(overrides)
    return ScreenSnapshotRow.model_validate(values)


def _candles(path: list[float], start: int) -> list[Candle]:
    """Candles whose open is the previous close; ``path`` holds the closes."""
    candles = []
    previous = path[0]
    for index, close in enumerate(path[1:], start=0):
        candles.append(
            Candle(
                symbol="R_10",
                epoch=start + index * 60,
                open=Decimal(str(previous)),
                high=Decimal(str(max(previous, close))),
                low=Decimal(str(min(previous, close))),
                close=Decimal(str(close)),
                granularity=60,
            )
        )
        previous = close
    return candles


def test_direction_rules() -> None:
    assert direction_for("trend", _row()) == "long"
    assert direction_for("trend", _row(ema20_50_trend="ema20_below_ema50")) == "short"
    assert direction_for("trend", _row(ema20_50_trend="insufficient_data")) is None
    assert direction_for("reversion", _row(bollinger_z=-1.2)) == "long"
    assert direction_for("reversion", _row(bollinger_z=1.2)) == "short"
    assert direction_for("reversion", _row(bollinger_z=0.3)) is None
    assert direction_for("reversion", _row(bollinger_z=None)) is None
    assert direction_for("rsi", _row(rsi=25.0)) == "long"
    assert direction_for("rsi", _row(rsi=75.0)) == "short"
    assert direction_for("rsi", _row(rsi=50.0)) is None
    assert direction_for("rsi", _row(rsi=None)) is None
    assert direction_for("coin", _row()) in {"long", "short"}
    assert direction_for("coin", _row()) == direction_for("coin", _row())
    with pytest.raises(ValueError):
        direction_for("magic", _row())


def test_calls_are_only_recorded_before_the_entry_candle_opens() -> None:
    ledger = Ledger()
    assert record_calls(ledger, [_row()], generated_at=T0 + 30) > 0
    assert ledger.first_recorded_epoch == T0 + 30
    before = len(ledger.pending)
    assert record_calls(ledger, [_row()], generated_at=T0 + 31) == 0  # duplicate ids
    assert len(ledger.pending) == before
    late = Ledger()
    assert record_calls(late, [_row()], generated_at=T0 + 60) == 0  # entry already open


def _pending_long(signal: int = T0) -> Call:
    return Call(
        id=f"R_10|trend|{signal}",
        symbol="R_10",
        strategy="trend",
        direction="long",
        signal_epoch=signal,
        recorded_epoch=signal + 30,
    )


def test_scoring_uses_only_later_candles_and_counts_flat_as_loss() -> None:
    candles = _candles([100.0, 100.0, 100.5, 101.0, 101.5, 102.0, 102.5], T0)
    ledger = Ledger(pending=[_pending_long()])
    now = T0 + 60 * (HORIZON_CANDLES + 2)
    assert score_pending(ledger, {"R_10": candles}, now) == 1
    assert ledger.pending == []
    assert ledger.tallies["trend"] == Tally(calls=1, wins=1, voids=0)
    assert ledger.recent[-1].outcome == "win"

    flat = _candles([100.0] * 8, T0)
    ledger = Ledger(pending=[_pending_long()])
    score_pending(ledger, {"R_10": flat}, now)
    assert ledger.tallies["trend"].wins == 0
    assert ledger.recent[-1].outcome == "loss"


def test_unclosed_horizon_stays_pending_and_gaps_are_void() -> None:
    ledger = Ledger(pending=[_pending_long()])
    assert score_pending(ledger, {"R_10": []}, T0 + 120) == 0
    assert len(ledger.pending) == 1
    now = T0 + 60 * (HORIZON_CANDLES + 3)
    assert score_pending(ledger, {}, now) == 1
    assert ledger.tallies["trend"].voids == 1
    assert ledger.tallies["trend"].calls == 0

    stale = Ledger(pending=[_pending_long()])
    assert score_pending(stale, {}, T0 + 10 * 86_400) == 1


def test_short_calls_win_on_falling_prices() -> None:
    candles = _candles([100.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0], T0)
    call = _pending_long().model_copy(update={"direction": "short"})
    ledger = Ledger(pending=[call])
    score_pending(ledger, {"R_10": candles}, T0 + 60 * (HORIZON_CANDLES + 2))
    assert ledger.recent[-1].outcome == "win"


def test_summary_waits_for_enough_calls_then_corrects_for_multiplicity() -> None:
    empty = summarise(Ledger())
    assert {item.verdict for item in empty.strategies if item.strategy != "coin"} == {
        "collecting data"
    }
    assert "Still collecting" in empty.headline

    chance = Ledger(
        tallies={name: Tally(calls=1000, wins=500) for name in ledger_module.STRATEGIES}
    )
    summary = summarise(chance)
    assert "No call type has beaten" in summary.headline
    assert all(item.verdict == "no edge detected" for item in summary.strategies[:3])
    assert summary.strategies[-1].verdict == "control"

    lucky = Ledger(tallies={"trend": Tally(calls=1000, wins=600)})
    assert summarise(lucky).strategies[0].verdict == "edge detected"
    assert "re-test" in summarise(lucky).headline
    assert MIN_VERDICT_CALLS > 0


def test_ledger_round_trip_and_bad_file(tmp_path: Path) -> None:
    path = tmp_path / "snap" / "ledger.json"
    assert load_ledger(path) == Ledger()
    ledger = Ledger(pending=[_pending_long()], tallies={"trend": Tally(calls=2, wins=1)})
    write_ledger(ledger, path)
    assert load_ledger(path) == ledger
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LedgerUnavailableError):
        load_ledger(path)


def test_default_ledger_path_follows_snapshot_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SNAPSHOT_PATH", str(tmp_path / "x" / "latest.json"))
    monkeypatch.delenv("LEDGER_PATH", raising=False)
    assert ledger_module.default_ledger_path() == tmp_path / "x" / "ledger.json"
    monkeypatch.setenv("LEDGER_PATH", str(tmp_path / "other.json"))
    assert ledger_module.default_ledger_path() == tmp_path / "other.json"


def test_technical_read_is_descriptive() -> None:
    read = read_row(_row(rsi=25.0, bollinger_z=-2.3, vol_ratio=1.3, hurst=0.6))
    assert "oversold" in read.momentum
    assert "far below" in read.stretch
    assert "hotter" in read.volatility
    assert "deviation" in read.behaviour
    assert {call.strategy for call in read.calls} == {"trend", "reversion", "rsi"}
    quiet = read_row(_row(rsi=None, bollinger_z=None, vol_ratio=None, hurst=None))
    assert quiet.momentum == quiet.stretch == quiet.volatility == quiet.behaviour == "n/a"
    assert "random walk" in read_row(_row()).behaviour
    assert "overbought" in read_row(_row(rsi=80.0)).momentum
    assert "cooler" in read_row(_row(vol_ratio=0.7)).volatility
    assert "above average" in read_row(_row(bollinger_z=1.3)).stretch
    assert "far above" in read_row(_row(bollinger_z=2.5)).stretch
    assert "below average" in read_row(_row(bollinger_z=-1.3)).stretch
    assert read_row(_row(ema20_50_trend="x")).trend == "unclear"
    assert "down" in read_row(_row(ema20_50_trend="ema20_below_ema50")).trend
    assert build_technical_read.__doc__


def test_update_ledger_scores_then_records(tmp_path: Path) -> None:
    from deriv_vol_lab.api.ledger import update_ledger
    from deriv_vol_lab.api.snapshot import AnalyticsSnapshot

    row = _row(epoch=T0 + 600)
    snapshot = AnalyticsSnapshot.model_construct(
        generated_at_epoch=T0 + 630, granularity=60, candle_count=0, screen=[row]
    )
    ledger = Ledger(pending=[_pending_long()])
    candles = _candles([100.0, 100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0], T0)
    update_ledger(ledger, snapshot, {"R_10": candles})
    assert ledger.tallies["trend"].calls == 1
    assert ledger.last_updated_epoch == T0 + 630
    assert any(call.signal_epoch == T0 + 600 for call in ledger.pending)
