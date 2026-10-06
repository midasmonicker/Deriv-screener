"""Tests for read-only snapshot API endpoints and dashboard content."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from deriv_vol_lab import main as main_module
from deriv_vol_lab.api import ingestion
from deriv_vol_lab.api.app import app
from deriv_vol_lab.api.snapshot import build_snapshot, load_snapshot, write_snapshot
from deriv_vol_lab.data import SYMBOLS, Candle


def _api_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _candles_by_symbol() -> dict[str, list[Candle]]:
    epoch = 1_704_067_200
    return {
        symbol: [
            Candle(
                symbol=symbol,
                epoch=epoch + index * 60,
                open=Decimal(str(100 + index)),
                high=Decimal(str(101 + index)),
                low=Decimal(str(99 + index)),
                close=Decimal(str(100 + index)),
                granularity=60,
            )
            for index in range(3)
        ]
        for symbol in SYMBOLS
    }


def test_snapshot_round_trip_is_atomic_and_json_safe(tmp_path: Path) -> None:
    snapshot = build_snapshot(_candles_by_symbol(), generated_at_epoch=1_704_067_320)
    snapshot_path = tmp_path / "snapshots" / "latest.json"

    write_snapshot(snapshot, snapshot_path)

    assert load_snapshot(snapshot_path) == snapshot
    assert len(snapshot.screen) == len(SYMBOLS)
    assert len(snapshot.quality_by_symbol) == len(SYMBOLS)
    assert snapshot.granularity == 60
    assert not list(snapshot_path.parent.glob("*.tmp"))


async def test_api_serves_all_read_only_snapshot_views(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "latest.json"
    snapshot = build_snapshot(_candles_by_symbol(), generated_at_epoch=1_704_067_320)
    write_snapshot(snapshot, snapshot_path)
    monkeypatch.setenv("SNAPSHOT_PATH", str(snapshot_path))

    async with _api_client() as client:
        health = await client.get("/health")
        screen = await client.get("/screen")
        quality = await client.get("/data-quality")
        volatility = await client.get("/stats/vol-monitor")

    assert health.status_code == 200
    assert health.json()["snapshot_available"]
    assert screen.status_code == 200
    assert len(screen.json()["rows"]) == len(SYMBOLS)
    assert screen.json()["granularity"] == 60
    assert quality.status_code == 200
    assert len(quality.json()["by_symbol"]) == len(SYMBOLS)
    assert quality.json()["by_symbol"][0]["status"] in {"healthy", "warning"}
    assert volatility.status_code == 200
    assert volatility.json()["report"]["confidence_level"] == pytest.approx(0.99)
    assert volatility.json()["report"]["windows"] == []


async def test_api_reports_missing_snapshot_without_fabricating_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SNAPSHOT_PATH", str(tmp_path / "missing.json"))
    async with _api_client() as client:
        assert (await client.get("/health")).json()["status"] == "degraded"
        response = await client.get("/screen")

    assert response.status_code == 503
    assert "snapshot does not exist" in response.json()["detail"]


async def test_dashboard_keeps_required_banner_visible() -> None:
    async with _api_client() as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert (
        "Descriptive analytics on random processes. No predictive edge is implied." in response.text
    )
    assert "position: sticky" in response.text
    assert 'fetch("/screen")' in response.text
    assert "qualityData.by_symbol.map" in response.text


def test_scheduled_ingestion_fetches_all_symbols_and_writes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The short-lived ingestion entry point persists one validated snapshot."""
    candles_by_symbol = _candles_by_symbol()
    snapshot_path = tmp_path / "latest.json"
    client_calls: list[str] = []

    class FakeClient:
        def __init__(self, app_id: int) -> None:
            assert app_id == 123

        async def connect(self, *, validate_symbols: bool) -> None:
            assert validate_symbols

        async def backfill_candles(
            self,
            symbol: str,
            days: float,
            *,
            granularity: int,
        ) -> list[Candle]:
            assert days == 1
            assert granularity == 60
            client_calls.append(symbol)
            return candles_by_symbol[symbol]

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        ingestion,
        "Settings",
        lambda: SimpleNamespace(deriv_app_id=123, snapshot_path=snapshot_path),
    )
    monkeypatch.setattr(ingestion, "DerivWebSocketClient", FakeClient)

    candle_count = asyncio.run(ingestion.ingest_latest_snapshot(days=1, seed=11))

    assert client_calls == list(SYMBOLS)
    assert candle_count == len(SYMBOLS) * 3
    assert load_snapshot(snapshot_path).generated_at_epoch > 0


def test_ingest_snapshot_cli_dispatches_requested_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[float, int]] = []

    async def fake_ingest(*, days: float, seed: int) -> int:
        calls.append((days, seed))
        return 30

    monkeypatch.setattr(main_module, "ingest_latest_snapshot", fake_ingest)
    main_module.main(["ingest-snapshot", "--days", "1.5", "--seed", "9"])

    assert calls == [(1.5, 9)]
    assert "30 candles" in capsys.readouterr().out
