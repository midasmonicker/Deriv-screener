"""Offline tests for the network smoke-check command."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Coroutine
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from deriv_vol_lab.data import SYMBOLS, Candle, HandshakeConnectionError
from deriv_vol_lab.stats.analysis import VolatilityWindowResult

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_live.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("smoke_live", _SCRIPT_PATH)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load smoke script at {_SCRIPT_PATH}")
smoke_live = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(smoke_live)


class _FakeDerivClient:
    def __init__(
        self,
        app_id: int | None,
        *,
        endpoint: str,
        max_handshake_attempts: int,
    ) -> None:
        assert app_id is None
        assert endpoint == "wss://api.derivws.com/trading/v1/options/ws/public"
        assert max_handshake_attempts == 5
        self.available_symbols = frozenset(SYMBOLS)
        self.requested_symbols: list[str] = []
        self.removed_candles = 0
        self.constant_symbols: set[str] = set()
        self.seed_by_symbol = {symbol: i for i, symbol in enumerate(SYMBOLS)}

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
        rng = np.random.default_rng(self.seed_by_symbol[symbol])
        price = 100.0
        candles: list[Candle] = []
        for position in range(count):
            opened = price
            if symbol not in self.constant_symbols:
                price *= float(np.exp(rng.normal(0.0, interval)))
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
    fake_client = _FakeDerivClient(
        None,
        endpoint="wss://api.derivws.com/trading/v1/options/ws/public",
        max_handshake_attempts=5,
    )
    fake_client.removed_candles = removed_candles
    monkeypatch.setattr(
        smoke_live,
        "Settings",
        lambda: SimpleNamespace(
            deriv_app_id=None,
            deriv_endpoint="wss://api.derivws.com/trading/v1/options/ws/public",
            deriv_handshake_max_attempts=5,
            db_path=tmp_path / "smoke.duckdb",
        ),
    )
    monkeypatch.setattr(
        smoke_live,
        "DerivWebSocketClient",
        lambda app_id, *, endpoint, max_handshake_attempts: (
            fake_client
            if app_id is None
            and endpoint == "wss://api.derivws.com/trading/v1/options/ws/public"
            and max_handshake_attempts == 5
            else pytest.fail("smoke client received unexpected connection settings")
        ),
    )
    monkeypatch.setattr(smoke_live, "BACKFILL_DAYS", 0.125)
    monkeypatch.setattr(smoke_live, "MONITOR_WINDOW", 20)
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
    assert "breaches=" in output
    assert "latest BH-adjusted p=" in output
    assert "Cross-symbol 1-minute log-return correlation matrix" in output
    assert "Cross-symbol max |corr|=" in output
    assert "Pairs with identical realized-vol ratios to 3 decimals=" in output


def test_smoke_live_max_seconds_aborts_during_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = _prepare_smoke(monkeypatch, tmp_path)

    async def never_finishes(
        symbol: str,
        days: float,
        *,
        granularity: int,
        end_epoch: int,
    ) -> list[Candle]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(fake_client, "backfill_candles", never_finishes)

    exit_code = asyncio.run(smoke_live.run_smoke(max_seconds=0.05))

    assert exit_code == 2
    assert capsys.readouterr().out == "Smoke check exceeded --max-seconds during fetch\n"


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


def test_smoke_live_fails_when_volatility_is_persistently_wrong(
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
    assert "overlapping, informational=160/160 (100.00%)" in output
    assert "non-overlapping breaches=8/8 (100.00%)" in output
    assert "latest BH-adjusted p=0" in output
    assert "latest volatility window is outside the BH-adjusted 99% CI" in output
    assert "non-overlapping breach rate is significantly above the 1% null rate" in output


def test_single_80_window_excursion_is_informational_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = _prepare_smoke(monkeypatch, tmp_path)
    fake_client.available_symbols = frozenset({"R_10"})

    def simulated_monitor(*args: object, **kwargs: object) -> SimpleNamespace:
        windows: list[VolatilityWindowResult] = []
        stated = SYMBOLS["R_10"].stated_vol_pct / 100
        for position in range(800):
            excursion = 100 <= position < 180
            windows.append(
                VolatilityWindowResult(
                    symbol="R_10",
                    window_end=position,
                    observations=100,
                    stated_volatility=stated,
                    status="ok",
                    realized_volatility=stated * (3.0 if excursion else 1.0),
                    confidence_level=0.99,
                    lower_confidence_bound=stated * 0.9,
                    upper_confidence_bound=stated * 1.1,
                    alert=excursion,
                )
            )
        return SimpleNamespace(windows=windows)

    monkeypatch.setattr(smoke_live, "MONITOR_WINDOW", 100)
    monkeypatch.setattr(smoke_live, "rolling_volatility_monitor", simulated_monitor)

    exit_code = asyncio.run(smoke_live.run_smoke())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "overlapping, informational" in output
    assert "non-overlapping breaches=1/8" in output
    assert "only 8 non-overlapping volatility windows; fewer than 10" in output
    assert "R_10: gaps=0, coverage=100.000%" in output
    assert "single-window excursion" not in output


@pytest.mark.parametrize(
    ("arguments", "expected_days", "expected_max_seconds"),
    [([], 7, None), (["--days", "3"], 3, None), (["--max-seconds", "30.5"], 7, 30.5)],
)
def test_smoke_cli_days_argument(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected_days: int,
    expected_max_seconds: float | None,
) -> None:
    monkeypatch.setattr(sys, "argv", ["smoke_live.py", *arguments])
    received_options: list[tuple[int | None, float | None]] = []

    def capture_run(coroutine: Coroutine[Any, Any, int]) -> int:
        assert coroutine.cr_frame is not None
        received_options.append(
            (
                coroutine.cr_frame.f_locals["days"],
                coroutine.cr_frame.f_locals["max_seconds"],
            )
        )
        coroutine.close()
        return 0

    monkeypatch.setattr(smoke_live.asyncio, "run", capture_run)

    with pytest.raises(SystemExit) as exit_error:
        smoke_live.main()

    assert exit_error.value.code == 0
    assert received_options == [(expected_days, expected_max_seconds)]


def test_cross_symbol_diagnostics_have_expected_correlations_and_ratio_pairs() -> None:
    increments = np.arange(1, 31, dtype=float) * 0.001
    epochs = pd.Index(np.arange(31) * 60, name="epoch")

    def close_series(returns: np.ndarray) -> pd.Series[float]:
        log_prices = np.concatenate(([0.0], np.cumsum(returns)))
        return pd.Series(np.exp(log_prices), index=epochs, dtype=float)

    matrix, maximum, threshold, exceeds_threshold, observations = (
        smoke_live._cross_symbol_diagnostics(
            {
                "A": close_series(increments),
                "B": close_series(increments * 2),
                "C": close_series(-increments),
            }
        )
    )

    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    assert matrix.loc["A", "C"] == pytest.approx(-1.0)
    assert maximum == pytest.approx(1.0)
    assert threshold == pytest.approx(4 / np.sqrt(30))
    assert exceeds_threshold
    assert observations == 30
    assert (
        smoke_live._count_identical_rounded_ratios(
            {"A": 1.0004, "B": 1.00049, "C": 1.0001, "D": 1.0006}
        )
        == 3
    )


def test_smoke_live_skips_symbol_missing_from_active_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = _prepare_smoke(monkeypatch, tmp_path)
    fake_client.available_symbols = frozenset(set(SYMBOLS) - {"R_75"})

    exit_code = asyncio.run(smoke_live.run_smoke())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "R_75: skipped (not present in active_symbols)" in output
    assert "R_75" not in fake_client.requested_symbols


def test_smoke_live_reports_handshake_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        smoke_live,
        "Settings",
        lambda: SimpleNamespace(
            deriv_app_id=None,
            deriv_endpoint="wss://api.derivws.com/trading/v1/options/ws/public",
            deriv_handshake_max_attempts=2,
            db_path=tmp_path / "smoke.duckdb",
        ),
    )

    class FailedClient:
        def __init__(
            self,
            app_id: int | None,
            *,
            endpoint: str,
            max_handshake_attempts: int,
        ) -> None:
            assert app_id is None
            assert endpoint == "wss://api.derivws.com/trading/v1/options/ws/public"
            assert max_handshake_attempts == 2

        async def connect(self, *, validate_symbols: bool) -> None:
            assert validate_symbols
            raise HandshakeConnectionError("ws.derivws.com", 2, 520)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(smoke_live, "DerivWebSocketClient", FailedClient)

    exit_code = asyncio.run(smoke_live.run_smoke())

    output = capsys.readouterr().out
    assert exit_code == 2
    assert output.strip() == (
        "WebSocket handshake failed after 2 attempts (host=ws.derivws.com, HTTP status=520)"
    )
