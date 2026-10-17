"""Tests for the resumable candle backfill command using a fake WebSocket server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from scripts import backfill
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from deriv_vol_lab.data import SYMBOLS, Candle, DuckDBStorage

_PROTOCOL_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "deriv_public_ws_responses.json").read_text(
        encoding="utf-8"
    )
)
_END_EPOCH = 1_800_000_000 // 60 * 60 - 60
_START_EPOCH = _END_EPOCH - 300 + 60


def _reply(request: dict[str, Any], msg_type: str, **fields: Any) -> str:
    return json.dumps(
        {
            "echo_req": request,
            "msg_type": msg_type,
            "req_id": request["req_id"],
            **fields,
        }
    )


def _candle(symbol: str, epoch: int) -> Candle:
    return Candle(
        symbol=symbol,
        epoch=epoch,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        granularity=60,
    )


@pytest_asyncio.fixture
async def history_server() -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                response = json.loads(json.dumps(_PROTOCOL_FIXTURES["active_symbols"]))
                response["req_id"] = request["req_id"]
                response["echo_req"] = request
                await websocket.send(json.dumps(response))
                continue
            requests.append(request)
            epochs = [
                request["end"] - offset * 60
                for offset in range(2)
                if request["end"] - offset * 60 >= _START_EPOCH
            ]
            candles = [
                {
                    "epoch": epoch,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                }
                for epoch in epochs
            ]
            await websocket.send(_reply(request, "candles", candles=candles))

    server = await serve(handler, "127.0.0.1", 0)
    socket = server.sockets[0]
    yield f"ws://127.0.0.1:{socket.getsockname()[1]}", requests
    server.close()
    await server.wait_closed()


def _configure_backfill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    endpoint: str,
) -> Path:
    database_path = tmp_path / "backfill.duckdb"
    monkeypatch.setattr(backfill, "SYMBOLS", {"R_75": SYMBOLS["R_75"]})
    monkeypatch.setattr(backfill, "SECONDS_PER_DAY", 300)
    monkeypatch.setattr("scripts.backfill.time.time", lambda: _END_EPOCH + 60)
    monkeypatch.setattr(
        backfill,
        "Settings",
        lambda: SimpleNamespace(
            deriv_app_id=None,
            deriv_endpoint=endpoint,
            deriv_handshake_max_attempts=1,
            db_path=database_path,
        ),
    )
    return database_path


@pytest.mark.asyncio
async def test_resume_fills_missing_candles_and_rerun_is_idempotent(
    history_server: tuple[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint, requests = history_server
    database_path = _configure_backfill(monkeypatch, tmp_path, endpoint)
    with DuckDBStorage(database_path) as storage:
        storage.write_candles([_candle("R_75", _END_EPOCH - 60), _candle("R_75", _END_EPOCH)])

    assert await backfill.run_backfill(days=1) == 0
    with DuckDBStorage(database_path) as storage:
        first_run = storage.read_candles(
            60,
            "R_75",
            start_epoch=_START_EPOCH,
            end_epoch=_END_EPOCH,
        )
    assert [candle.epoch for candle in first_run] == list(range(_START_EPOCH, _END_EPOCH + 1, 60))
    assert [request["end"] for request in requests] == [
        _END_EPOCH,
        _END_EPOCH - 120,
        _END_EPOCH - 240,
    ]
    progress = capsys.readouterr()
    assert progress.out == ""
    assert progress.err.count("R_75: page ") == 3
    assert "R_75: quality gaps=0, duplicates=0, coverage=100.000%" in progress.err

    assert await backfill.run_backfill(days=1) == 0
    with DuckDBStorage(database_path) as storage:
        second_run = storage.read_candles(
            60,
            "R_75",
            start_epoch=_START_EPOCH,
            end_epoch=_END_EPOCH,
        )
    assert second_run == first_run
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_backfill_stops_on_empty_history_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[dict[str, Any]] = []

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                response = json.loads(json.dumps(_PROTOCOL_FIXTURES["active_symbols"]))
                response["req_id"] = request["req_id"]
                response["echo_req"] = request
                await websocket.send(json.dumps(response))
            else:
                requests.append(request)
                await websocket.send(_reply(request, "candles", candles=[]))

    server = await serve(handler, "127.0.0.1", 0)
    socket = server.sockets[0]
    endpoint = f"ws://127.0.0.1:{socket.getsockname()[1]}"
    database_path = _configure_backfill(monkeypatch, tmp_path, endpoint)
    try:
        assert await backfill.run_backfill(days=1) == 0
    finally:
        server.close()
        await server.wait_closed()
    assert len(requests) == 1
    assert "R_75: page 1 empty" in capsys.readouterr().err
    with DuckDBStorage(database_path) as storage:
        assert storage.read_candles(60, "R_75") == []


@pytest.mark.asyncio
async def test_backfill_handshake_failure_returns_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def reject_handshake(connection: ServerConnection, request: Any) -> Response:
        return Response(503, "Service Unavailable", Headers(), b"offline")

    async def handler(websocket: ServerConnection) -> None:
        await websocket.wait_closed()

    server: Server = await serve(
        handler,
        "127.0.0.1",
        0,
        process_request=reject_handshake,
    )
    socket = server.sockets[0]
    endpoint = f"ws://127.0.0.1:{socket.getsockname()[1]}"
    database_path = _configure_backfill(monkeypatch, tmp_path, endpoint)
    try:
        assert await backfill.run_backfill(days=1) == 2
    finally:
        server.close()
        await server.wait_closed()
    assert not database_path.exists()


@pytest.mark.asyncio
async def test_max_seconds_deadline_returns_exit_code_two(
    history_server: tuple[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint, requests = history_server
    database_path = _configure_backfill(monkeypatch, tmp_path, endpoint)

    assert await backfill.run_backfill(days=1, max_seconds=1e-12) == 2
    assert requests == []
    assert "exceeded --max-seconds during connect" in capsys.readouterr().err
    assert not database_path.exists()
