"""Offline tests for the network smoke-check command."""

from __future__ import annotations

import asyncio
import importlib.util
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from deriv_vol_lab.data import SYMBOLS, Candle

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_live.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("smoke_live", _SCRIPT_PATH)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load smoke script at {_SCRIPT_PATH}")
smoke_live = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(smoke_live)


class _FakeDerivClient:
    def __init__(self, app_id: int) -> None:
        assert app_id == 123
        self.requested_symbols: list[str] = []
        self.removed_candles = 0
        self.constant_symbols: set[str] = set()

    async def connect(self, *, validate_symbols: bool) -> None:
        assert validate_symbols

    async def backfill_candles(
        self,
        symbol: str,
        days: float,
        *,
        granularity: int,
        end_epoch: int,
    ) -> list[Candle]:
        assert days == smoke_live.BACKFILL_DAYS
        assert granularity == smoke_live.GRANULARITY_SECONDS
        self.requested_symbols.append(symbol)
        count = round(days * 86_400 / granularity)
        interval = SYMBOLS[symbol].stated_vol_pct / 100 * (granularity / 31_536_000) ** 0.5
        price = 100.0
        candles: list[Candle] = []
        for position in range(count):
            opened = price
            if symbol not in self.constant_symbols:
                price *= 1.0 + (interval if position % 2 == 0 else -interval)
            candles.append(
                Candle(
                    symbol=symbol,
                    epoch=end_epoch - (count - position - 1) * granularity,
                    open=Decimal(str(opened)),
                    high=Decimal(str(max(opened, price) * 1.0001)),
                    low=Decimal(str(min(opened, price) * 0.9999)),
                    close=Decimal(str(price)),
                    granularity=granularity,
                )
            )
        if symbol == "R_75" and self.removed_candles:
            return [
                candle
                for position, candle in enumerate(candles)
                if position not in range(self.removed_candles)
            ]
        return candles

    async def close(self) -> None:
        return None


def _prepare_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    removed_candles: int = 0,
) -> _FakeDerivClient:
    fake_client = _FakeDerivClient(123)
    fake_client.removed_candles = removed_candles
    monkeypatch.setattr(
        smoke_live,
        "Settings",
        lambda: SimpleNamespace(deriv_app_id=123, db_path=tmp_path / "smoke.duckdb"),
    )
    monkeypatch.setattr(smoke_live, "DerivWebSocketClient", lambda app_id: fake_client)
    monkeypatch.setattr(smoke_live, "BACKFILL_DAYS", 0.125)
    monkeypatch.setattr(smoke_live, "MONITOR_WINDOW", 2)
    monkeypatch.setattr(smoke_live.time, "time", lambda: 1_700_000_000)
    return fake_client


def test_smoke_live_backfills_all_symbols_and_reports_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = _prepare_smoke(monkeypatch, tmp_path)

    exit_code = asyncio.run(smoke_live.run_smoke())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert fake_client.requested_symbols == list(SYMBOLS)
    for symbol in SYMBOLS:
        assert f"{symbol}: gaps=0, coverage=100.000%" in output
    assert "vol ratio=" in output
    assert "99% CI=[" in output


def test_smoke_live_fails_when_coverage_is_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_smoke(monkeypatch, tmp_path, removed_candles=3)

    exit_code = asyncio.run(smoke_live.run_smoke())

    output = capsys.readouterr().out
    assert exit_code != 0
    assert "R_75: gaps=3, coverage=98.333%" in output
    assert "coverage 98.333% is below 99.0%" in output


def test_smoke_live_fails_when_stated_volatility_is_outside_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = _prepare_smoke(monkeypatch, tmp_path)
    fake_client.constant_symbols.add("R_75")

    exit_code = asyncio.run(smoke_live.run_smoke())

    output = capsys.readouterr().out
    assert exit_code != 0
    assert "R_75: gaps=0, coverage=100.000%, vol ratio=0.0000" in output
    assert "ALERT: stated volatility is outside the 99% confidence interval" in output
