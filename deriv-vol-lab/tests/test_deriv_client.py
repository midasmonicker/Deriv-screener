"""Tests for the Deriv WebSocket market-data client."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal
from time import monotonic
from typing import Any, TypeAlias

import pytest
import pytest_asyncio
from websockets.asyncio.server import Server, ServerConnection, serve

from deriv_vol_lab.data import (
    SYMBOLS,
    Candle,
    ClientClosedError,
    ConnectionLostError,
    DerivAPIError,
    DerivWebSocketClient,
    Tick,
)

Handler: TypeAlias = Callable[[ServerConnection], Awaitable[None]]
ServerFactory: TypeAlias = Callable[[Handler], Awaitable[str]]


@pytest_asyncio.fixture
async def fake_websocket_server() -> AsyncIterator[ServerFactory]:
    """Start local WebSocket handlers and close all servers after each test."""
    servers: list[Server] = []

    async def start(handler: Handler) -> str:
        server = await serve(handler, "127.0.0.1", 0)
        servers.append(server)
        socket = server.sockets[0]
        return f"ws://127.0.0.1:{socket.getsockname()[1]}"

    yield start
    for server in servers:
        server.close()
        await server.wait_closed()


def _client(uri: str, **kwargs: Any) -> DerivWebSocketClient:
    options: dict[str, Any] = {
        "endpoint": uri,
        "requests_per_second": 1_000_000,
        "reconnect_initial_delay": 0.01,
        "reconnect_max_delay": 0.02,
    }
    options.update(kwargs)
    return DerivWebSocketClient(12345, **options)


def _active_symbols_response(request: dict[str, Any]) -> str:
    return json.dumps(
        {
            "req_id": request["req_id"],
            "active_symbols": [{"symbol": symbol} for symbol in SYMBOLS],
        }
    )


@pytest.mark.asyncio
async def test_client_and_history_reject_invalid_arguments(
    fake_websocket_server: ServerFactory,
) -> None:
    """Configuration and backfill bounds fail explicitly."""
    with pytest.raises(ValueError, match="requests_per_second"):
        _client("ws://localhost", requests_per_second=0)
    with pytest.raises(ValueError, match="heartbeat_interval"):
        _client("ws://localhost", heartbeat_interval=0)
    with pytest.raises(ValueError, match="Reconnect delays"):
        _client(
            "ws://localhost",
            reconnect_initial_delay=0.2,
            reconnect_max_delay=0.1,
        )
    with pytest.raises(ValueError, match="request_timeout"):
        _client("ws://localhost", request_timeout=0)

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            await websocket.send(_active_symbols_response(request))

    client = _client(await fake_websocket_server(handler))
    await client.connect()
    with pytest.raises(ValueError, match="days"):
        await client.backfill_ticks("R_75", 0)
    with pytest.raises(ValueError, match="page_size"):
        await client.backfill_ticks("R_75", 1, page_size=5001)
    with pytest.raises(ValueError, match="granularity"):
        await client.backfill_candles("R_75", 1, granularity=0)
    with pytest.raises(ValueError, match="granularity"):
        await client.subscribe_ohlc("R_75", 0)
    await client.close()
    await client.close()
    with pytest.raises(ClientClosedError, match="closed"):
        await client.connect()
    with pytest.raises(ClientClosedError, match="closed"):
        await client.request({"ping": 1})


@pytest.mark.asyncio
async def test_requests_are_correlated_when_responses_arrive_out_of_order(
    fake_websocket_server: ServerFactory,
) -> None:
    """Parallel API calls receive their own responses rather than arrival order."""
    received: list[dict[str, Any]] = []
    both_received = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            received.append(request)
            if len(received) == 2:
                both_received.set()
                await asyncio.wait_for(both_received.wait(), timeout=1)
                for item in reversed(received):
                    await websocket.send(
                        json.dumps(
                            {
                                "req_id": item["req_id"],
                                "value": item["value"],
                            }
                        )
                    )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect(validate_symbols=False)
        first, second = await asyncio.gather(
            client.request({"value": "first"}),
            client.request({"value": "second"}),
        )
        assert first["value"] == "first"
        assert second["value"] == "second"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_first_request_connects_and_validates_registry(
    fake_websocket_server: ServerFactory,
) -> None:
    """The request API establishes the connection and runs startup validation."""

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            else:
                await websocket.send(json.dumps({"req_id": request["req_id"], "ping": "pong"}))

    client = _client(await fake_websocket_server(handler))
    try:
        response = await client.request({"ping": 1})
        assert response["ping"] == "pong"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_symbol_registry_warns_when_active_symbols_differ(
    fake_websocket_server: ServerFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Startup emits a warning when supported synthetic indices are missing."""

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            await websocket.send(
                json.dumps(
                    {
                        "req_id": request["req_id"],
                        "active_symbols": [{"symbol": "R_999"}],
                    }
                )
            )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        output = capsys.readouterr().out
        assert "symbol_registry_mismatch" in output
        assert "R_999" in output
        assert "R_75" in output
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deriv_error_response_raises_typed_error(
    fake_websocket_server: ServerFactory,
) -> None:
    """A Deriv error payload is surfaced with its API code and message."""

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            await websocket.send(
                json.dumps(
                    {
                        "req_id": request["req_id"],
                        "error": {"code": "InvalidSymbol", "message": "Unknown symbol"},
                    }
                )
            )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect(validate_symbols=False)
        with pytest.raises(DerivAPIError, match="InvalidSymbol: Unknown symbol") as error:
            await client.request({"ticks": "NOT_A_SYMBOL"})
        assert error.value.code == "InvalidSymbol"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rate_limit_spaces_requests(
    fake_websocket_server: ServerFactory,
) -> None:
    """Concurrent requests are spaced by the configured client-side limiter."""
    received_at: list[float] = []

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            received_at.append(monotonic())
            await websocket.send(json.dumps({"req_id": request["req_id"]}))

    client = DerivWebSocketClient(
        12345,
        endpoint=await fake_websocket_server(handler),
        requests_per_second=20,
    )
    try:
        await client.connect(validate_symbols=False)
        await asyncio.gather(client.request({"ping": 1}), client.request({"ping": 1}))
        assert len(received_at) == 2
        assert received_at[1] - received_at[0] >= 0.04
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tick_history_paginates_backwards_with_server_limit(
    fake_websocket_server: ServerFactory,
) -> None:
    """Tick history pages walk back until the requested start epoch."""
    history = [100_000, 90_000, 80_000, 70_000, 60_000, 50_000, 40_000, 30_000, 20_000, 10_000]
    requests: list[dict[str, Any]] = []

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            requests.append(request)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
                continue
            page = [epoch for epoch in history if epoch <= request["end"]][: request["count"]]
            await websocket.send(
                json.dumps(
                    {
                        "req_id": request["req_id"],
                        "history": {"times": page, "prices": [str(epoch / 1000) for epoch in page]},
                    }
                )
            )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        ticks = await client.backfill_ticks("R_75", 1, end_epoch=100_000, page_size=2)
        assert [tick.epoch for tick in ticks] == [
            20_000,
            30_000,
            40_000,
            50_000,
            60_000,
            70_000,
            80_000,
            90_000,
            100_000,
        ]
        history_requests = [request for request in requests if "ticks_history" in request]
        assert [request["end"] for request in history_requests] == [
            100_000,
            89_999,
            69_999,
            49_999,
            29_999,
        ]
        assert all(request["count"] <= 5000 for request in history_requests)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_candle_history_uses_candles_style_and_typed_model(
    fake_websocket_server: ServerFactory,
) -> None:
    """The candles history request is parsed into the Candle contract."""

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            else:
                await websocket.send(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "candles": [
                                {
                                    "epoch": 100,
                                    "open": "1.1",
                                    "high": "1.4",
                                    "low": "1.0",
                                    "close": "1.3",
                                }
                            ],
                        }
                    )
                )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        candles = await client.backfill_candles(
            "1HZ75V",
            1,
            granularity=60,
            end_epoch=100,
        )
        assert candles == [
            Candle(
                symbol="1HZ75V",
                epoch=100,
                open="1.1",
                high="1.4",
                low="1.0",
                close="1.3",
                granularity=60,
            )
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_malformed_history_response_raises_value_error(
    fake_websocket_server: ServerFactory,
) -> None:
    """History response shape and model validation errors are not suppressed."""

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            elif request["style"] == "ticks":
                await websocket.send(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "history": {"times": [100], "prices": []},
                        }
                    )
                )
            else:
                await websocket.send(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "candles": [{"epoch": "invalid"}],
                        }
                    )
                )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        with pytest.raises(ValueError, match="different lengths"):
            await client.backfill_ticks("R_75", 1, end_epoch=100)
        with pytest.raises(ValueError, match="malformed history"):
            await client.backfill_candles("R_75", 1, granularity=60, end_epoch=100)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_reconnect_resubscribes_and_delivers_live_ticks(
    fake_websocket_server: ServerFactory,
) -> None:
    """An active tick stream is re-established after the socket reconnects."""
    connections = 0
    resubscribed = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        nonlocal connections
        connections += 1
        connection_number = connections
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            elif "ticks" in request:
                sub_id = f"ticks-{connection_number}"
                await websocket.send(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "subscription": {"id": sub_id},
                        }
                    )
                )
                if connection_number == 1:
                    await websocket.close(code=1012, reason="restart")
                    return
                await websocket.send(
                    json.dumps(
                        {
                            "msg_type": "tick",
                            "tick": {"symbol": "R_75", "epoch": 2, "quote": "123.45"},
                            "subscription": {"id": sub_id},
                        }
                    )
                )
                resubscribed.set()
            elif "ping" in request:
                await websocket.send(json.dumps({"req_id": request["req_id"], "ping": "pong"}))
            elif "forget" in request:
                await websocket.send(json.dumps({"req_id": request["req_id"]}))

    client = _client(
        await fake_websocket_server(handler),
        heartbeat_interval=0.03,
    )
    try:
        await client.connect()
        stream = await client.subscribe_ticks("R_75")
        tick = await asyncio.wait_for(stream.get(), timeout=2)
        assert isinstance(tick, Tick)
        assert tick.epoch == 2
        assert tick.quote == Decimal("123.45")
        await asyncio.wait_for(resubscribed.wait(), timeout=2)
        await stream.close()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_reconnect_backoff_grows_for_short_lived_connections(
    fake_websocket_server: ServerFactory,
) -> None:
    """Repeated short-lived sockets increase the exponential reconnect delay."""
    connected_at: list[float] = []
    third_connection = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        connected_at.append(monotonic())
        if len(connected_at) >= 3:
            third_connection.set()
        await websocket.close(code=1012, reason="reconnect test")

    client = DerivWebSocketClient(
        12345,
        endpoint=await fake_websocket_server(handler),
        requests_per_second=1_000_000,
        reconnect_initial_delay=0.05,
        reconnect_max_delay=0.2,
        heartbeat_interval=5,
        jitter_seed=0,
    )
    try:
        await client.connect(validate_symbols=False)
        await asyncio.wait_for(third_connection.wait(), timeout=2)
        first_gap = connected_at[1] - connected_at[0]
        second_gap = connected_at[2] - connected_at[1]
        assert second_gap > first_gap
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ohlc_subscription_yields_typed_candle(
    fake_websocket_server: ServerFactory,
) -> None:
    """OHLC subscription updates include symbol and granularity metadata."""

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            elif request.get("style") == "candles":
                subscription_id = "ohlc-1"
                await websocket.send(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "subscription": {"id": subscription_id},
                        }
                    )
                )
                await websocket.send(
                    json.dumps(
                        {
                            "msg_type": "ohlc",
                            "ohlc": {
                                "epoch": 60,
                                "open": "1",
                                "high": "2",
                                "low": "0.5",
                                "close": "1.5",
                            },
                            "subscription": {"id": subscription_id},
                        }
                    )
                )
            elif "forget" in request:
                await websocket.send(json.dumps({"req_id": request["req_id"]}))

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        stream = await client.subscribe_ohlc("R_75", 60)
        candle = await asyncio.wait_for(stream.get(), timeout=1)
        assert isinstance(candle, Candle)
        assert candle.symbol == "R_75"
        assert candle.granularity == 60
        await stream.close()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_unblocks_request_waiting_for_reconnect(
    fake_websocket_server: ServerFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing in the disconnected reconnect state completes connectivity waiters."""

    async def handler(websocket: ServerConnection) -> None:
        await websocket.close(code=1012, reason="force reconnect")

    client = _client(
        await fake_websocket_server(handler),
        reconnect_initial_delay=0.5,
        reconnect_max_delay=0.5,
    )
    disconnected = asyncio.Event()
    original_clear = client._connected.clear

    def record_disconnect() -> None:
        original_clear()
        disconnected.set()

    monkeypatch.setattr(client._connected, "clear", record_disconnect)
    request_task: asyncio.Task[dict[str, Any]] | None = None
    try:
        await client.connect(validate_symbols=False)
        await asyncio.wait_for(disconnected.wait(), timeout=1)
        request_task = asyncio.create_task(client.request({"ping": 1}))
        await asyncio.sleep(0)
        await client.close()

        completed, _ = await asyncio.wait({request_task}, timeout=0.05)
        assert request_task in completed
        with pytest.raises(ClientClosedError, match="closed"):
            await request_task
    finally:
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_failed_resubscription_terminates_and_unregisters_stream(
    fake_websocket_server: ServerFactory,
) -> None:
    """An API error while restoring a subscription must end its iterator."""
    connection_count = 0
    restore_error_sent = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        nonlocal connection_count
        connection_count += 1
        current_connection = connection_count
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            elif "ticks" in request:
                if current_connection == 1:
                    await websocket.send(
                        json.dumps(
                            {
                                "req_id": request["req_id"],
                                "subscription": {"id": "initial-subscription"},
                            }
                        )
                    )
                    await websocket.close(code=1012, reason="force resubscription")
                    return
                await websocket.send(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "error": {
                                "code": "InvalidSymbol",
                                "message": "Subscription restore rejected",
                            },
                        }
                    )
                )
                restore_error_sent.set()

    client = _client(
        await fake_websocket_server(handler),
        heartbeat_interval=5,
    )
    next_task: asyncio.Task[Tick | Candle] | None = None
    try:
        await client.connect()
        stream = await client.subscribe_ticks("R_75")
        await asyncio.wait_for(restore_error_sent.wait(), timeout=2)
        with pytest.raises(DerivAPIError, match="InvalidSymbol"):
            await asyncio.wait_for(stream.__anext__(), timeout=1)

        next_task = asyncio.create_task(stream.__anext__())
        completed, _ = await asyncio.wait({next_task}, timeout=0.05)
        terminated = bool(
            completed
            and next_task.exception() is not None
            and isinstance(next_task.exception(), StopAsyncIteration)
        )
        unregistered = id(stream) not in client._subscriptions

        assert terminated and unregistered, (
            f"iterator_terminated={terminated}, handle_unregistered={unregistered}"
        )
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_disconnect_fails_in_flight_request_with_retryable_error(
    fake_websocket_server: ServerFactory,
) -> None:
    """A dropped socket completes pending requests with a retryable exception."""
    connection_count = 0

    async def handler(websocket: ServerConnection) -> None:
        nonlocal connection_count
        connection_count += 1
        current_connection = connection_count
        async for raw in websocket:
            request = json.loads(raw)
            if "ping" in request:
                if current_connection == 1:
                    await websocket.close(code=1012, reason="server restart")
                    return
                await websocket.send(json.dumps({"req_id": request["req_id"], "ping": "pong"}))

    client = _client(await fake_websocket_server(handler), heartbeat_interval=5)
    try:
        await client.connect(validate_symbols=False)
        with pytest.raises(ConnectionLostError, match="disconnected") as error:
            await asyncio.wait_for(client.request({"ping": 1}), timeout=1)
        assert error.value.retryable

        await asyncio.wait_for(client._connected.wait(), timeout=1)
        response = await client.request({"ping": 1})
        assert response["ping"] == "pong"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unexpected_manager_failure_is_logged_surfaced_and_retried(
    fake_websocket_server: ServerFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A manager exception fails active requests and reconnects with a fresh socket."""
    connection_count = 0
    second_connection = asyncio.Event()
    first_connection_closed = asyncio.Event()
    request_received = asyncio.Event()
    allow_manager_crash = asyncio.Event()
    restore_started = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        nonlocal connection_count
        connection_count += 1
        current_connection = connection_count
        if current_connection >= 2:
            second_connection.set()
        async for raw in websocket:
            request = json.loads(raw)
            if "ping" in request:
                if current_connection == 1:
                    request_received.set()
                    continue
                await websocket.send(json.dumps({"req_id": request["req_id"], "ping": "pong"}))
        if current_connection == 1:
            first_connection_closed.set()

    client = _client(await fake_websocket_server(handler))
    original_restore = client._restore_subscriptions
    restore_calls = 0

    async def fail_first_restore() -> None:
        nonlocal restore_calls
        restore_calls += 1
        if restore_calls == 1:
            restore_started.set()
            await allow_manager_crash.wait()
            raise RuntimeError("injected manager failure")
        await original_restore()

    monkeypatch.setattr(client, "_restore_subscriptions", fail_first_restore)
    request_task: asyncio.Task[dict[str, Any]] | None = None
    try:
        await client.connect(validate_symbols=False)
        await asyncio.wait_for(restore_started.wait(), timeout=1)
        request_task = asyncio.create_task(client.request({"ping": 1}))
        await asyncio.wait_for(request_received.wait(), timeout=1)
        allow_manager_crash.set()

        with pytest.raises(ConnectionLostError, match="manager failed") as error:
            await asyncio.wait_for(request_task, timeout=1)
        assert error.value.retryable
        assert isinstance(error.value.__cause__, RuntimeError)

        await asyncio.wait_for(first_connection_closed.wait(), timeout=1)
        await asyncio.wait_for(second_connection.wait(), timeout=1)
        await asyncio.wait_for(client._connected.wait(), timeout=1)
        response = await client.request({"ping": 1})
        assert response["ping"] == "pong"
        assert "deriv_connection_manager_unexpected_error" in capsys.readouterr().out
    finally:
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        await client.close()
