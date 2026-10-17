"""Tests for the Deriv WebSocket market-data client."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any, TypeAlias

import pytest
import pytest_asyncio
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import WebSocketException
from websockets.http11 import Response

import deriv_vol_lab.data.client as client_module
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
_PROTOCOL_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "deriv_public_ws_responses.json").read_text(
        encoding="utf-8"
    )
)


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
    response = json.loads(json.dumps(_PROTOCOL_FIXTURES["active_symbols"]))
    response["req_id"] = request["req_id"]
    response["echo_req"] = request
    return json.dumps(response)


def _fixture_response(name: str, request: dict[str, Any]) -> str:
    response = json.loads(json.dumps(_PROTOCOL_FIXTURES[name]))
    response["req_id"] = request["req_id"]
    response["echo_req"] = request
    return json.dumps(response)


def test_live_confirmed_1hz_symbols_match_active_symbol_display_names() -> None:
    expected = {
        "1HZ15V": ("Volatility 15 (1s) Index", 15),
        "1HZ30V": ("Volatility 30 (1s) Index", 30),
        "1HZ90V": ("Volatility 90 (1s) Index", 90),
    }
    active_rows = {
        row["underlying_symbol"]: row["underlying_symbol_name"]
        for row in _PROTOCOL_FIXTURES["active_symbols"]["active_symbols"]
    }

    for code, (display_name, stated_vol_pct) in expected.items():
        definition = SYMBOLS[code]
        assert definition.code == code
        assert definition.family == "1HZ"
        assert definition.stated_vol_pct == stated_vol_pct
        assert definition.tick_interval_sec == 1
        assert active_rows[code] == display_name


def _reply(request: dict[str, Any], msg_type: str, **fields: Any) -> str:
    return json.dumps(
        {
            "echo_req": request,
            "msg_type": msg_type,
            "req_id": request["req_id"],
            **fields,
        }
    )


@pytest.mark.asyncio
async def test_handshake_failure_exposes_network_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handshake errors expose status and safe server-side diagnostic headers."""

    class Response:
        def __init__(self) -> None:
            self.status_code = 520
            self.reason_phrase = "Origin Error"
            self.headers = {
                "server": "cloudflare",
                "cf-ray": "example-ray-id",
                "content-type": "text/html",
            }

    class HandshakeRejected(WebSocketException):
        def __init__(self) -> None:
            self.response = Response()
            super().__init__("Handshake rejected")

    connection_options: list[dict[str, Any]] = []

    async def reject_handshake(*args: Any, **kwargs: Any) -> Any:
        connection_options.append(kwargs)
        raise HandshakeRejected()

    monkeypatch.setattr(client_module, "websocket_connect", reject_handshake)
    client = DerivWebSocketClient(
        12345,
        endpoint="wss://ws.derivws.com/websockets/v3",
        max_handshake_attempts=1,
        reconnect_initial_delay=10,
        reconnect_max_delay=10,
    )
    try:
        with pytest.raises(client_module.HandshakeConnectionError) as error:
            await client.connect(validate_symbols=False)
        assert error.value.host == "ws.derivws.com"
        assert error.value.status_code == 520
        assert error.value.attempts == 1
        assert connection_options[0]["compression"] is None
        assert "app_id" not in str(error.value)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_520_handshake_stops_after_configured_attempts() -> None:
    """An offline server can reject every handshake; retries stop at the configured limit."""
    attempts = 0

    def reject_handshake(
        connection: ServerConnection,
        request: Any,
    ) -> Response:
        nonlocal attempts
        attempts += 1
        return Response(520, "Origin Error", Headers(), b"handshake rejected")

    async def handler(websocket: ServerConnection) -> None:
        await websocket.wait_closed()

    server = await serve(handler, "127.0.0.1", 0, process_request=reject_handshake)
    socket = server.sockets[0]
    client = _client(
        f"ws://127.0.0.1:{socket.getsockname()[1]}",
        max_handshake_attempts=2,
        reconnect_initial_delay=0.001,
        reconnect_max_delay=0.002,
    )
    try:
        with pytest.raises(client_module.HandshakeConnectionError) as error:
            await client.connect(validate_symbols=False, connection_timeout=5)
        assert attempts == 2
        assert error.value.status_code == 520
        assert error.value.host == "127.0.0.1"
        assert error.value.attempts == 2
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


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
    with pytest.raises(ValueError, match="valid ws:// or wss:// URL"):
        _client("wss://://binaryws.com")

    default_client = DerivWebSocketClient()
    assert default_client._endpoint_host == "api.derivws.com"
    assert default_client._uri == "wss://api.derivws.com/trading/v1/options/ws/public"


@pytest.mark.asyncio
async def test_app_id_is_sent_only_as_deriv_header(
    fake_websocket_server: ServerFactory,
) -> None:
    """Optional app IDs use the protocol header and never alter the endpoint URI."""
    observed: list[tuple[str, str | None]] = []
    connected = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        observed.append(
            (
                websocket.request.path,
                websocket.request.headers.get("Deriv-App-ID"),
            )
        )
        connected.set()
        await websocket.wait_closed()

    endpoint = await fake_websocket_server(handler)
    client = _client(endpoint)
    try:
        await client.connect(validate_symbols=False)
        await asyncio.wait_for(connected.wait(), timeout=1)
        assert observed == [("/", "12345")]
        assert "app_id" not in client._uri
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_client_without_app_id_sends_no_app_id_header(
    fake_websocket_server: ServerFactory,
) -> None:
    observed: list[tuple[str, str | None]] = []
    connected = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        observed.append(
            (
                websocket.request.path,
                websocket.request.headers.get("Deriv-App-ID"),
            )
        )
        connected.set()
        await websocket.wait_closed()

    endpoint = await fake_websocket_server(handler)
    client = DerivWebSocketClient(endpoint=endpoint)
    try:
        await client.connect(validate_symbols=False)
        await asyncio.wait_for(connected.wait(), timeout=1)
        assert observed == [("/", None)]
        assert "app_id" not in client._uri
    finally:
        await client.close()

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
                                "echo_req": item,
                                "msg_type": "test_response",
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
    active_request: dict[str, Any] | None = None

    async def handler(websocket: ServerConnection) -> None:
        nonlocal active_request
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                active_request = request
                await websocket.send(_active_symbols_response(request))
            else:
                await websocket.send(_fixture_response("ping", request))

    client = _client(await fake_websocket_server(handler))
    try:
        response = await client.request({"ping": 1})
        assert response["ping"] == "pong"
        assert active_request == {"active_symbols": "brief", "req_id": 1}
        assert client.available_symbols == frozenset(SYMBOLS)
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
                _reply(
                    request,
                    "active_symbols",
                    active_symbols=[
                        {
                            "underlying_symbol": "R_999",
                            "underlying_symbol_name": "Volatility 999 Index",
                        }
                    ],
                )
            )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        output = capsys.readouterr().out
        assert "symbol_registry_mismatch" in output
        assert "R_999" in output
        assert "R_75" in output
        assert client.available_symbols == frozenset()
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
                _reply(
                    request,
                    "tick",
                    error={
                        "code": "InvalidSymbol",
                        "details": {},
                        "message": "Unknown symbol",
                    },
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
            await websocket.send(_reply(request, "ping", ping="pong"))

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
                _reply(
                    request,
                    "history",
                    history={
                        "times": page,
                        "prices": [str(epoch / 1000) for epoch in page],
                    },
                    pip_size=2,
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
        assert all("adjust_start" not in request for request in history_requests)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_tick_history_fixture_parses_required_arrays(
    fake_websocket_server: ServerFactory,
) -> None:
    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            else:
                await websocket.send(_fixture_response("ticks_history_ticks", request))

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        ticks = await client.backfill_ticks(
            "R_100",
            1,
            end_epoch=1791467372,
            page_size=5000,
        )
        assert [(tick.epoch, tick.quote) for tick in ticks] == [
            (1791467368, Decimal("628.53")),
            (1791467370, Decimal("628.65")),
            (1791467372, Decimal("628.82")),
        ]
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
                await websocket.send(_fixture_response("ticks_history_candles", request))

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        candles = await client.backfill_candles(
            "1HZ75V",
            1,
            granularity=60,
            end_epoch=1791467340,
        )
        assert [candle.epoch for candle in candles] == [
            1791467220,
            1791467280,
            1791467340,
        ]
        assert candles[-1] == Candle(
            symbol="1HZ75V",
            epoch=1791467340,
            open="626.99",
            high="628.82",
            low="626.8",
            close="628.82",
            granularity=60,
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_candle_backfill_caps_pages_and_stops_at_time_boundary(
    fake_websocket_server: ServerFactory,
) -> None:
    """Candle pages contain at most 1,000 rows and cursor back to the range start."""
    requests: list[dict[str, Any]] = []
    end_epoch = 1_800_000_000
    start_epoch = end_epoch - 1000 * 60

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
                continue
            requests.append(request)
            epochs = list(
                range(
                    request["end"] // 60 * 60,
                    max(start_epoch, request["end"] - 1000 * 60) - 1,
                    -60,
                )
            )[: request["count"]]
            candles = [
                {
                    "epoch": epoch,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                }
                for epoch in epochs[: min(request["count"], 1000)]
            ]
            await websocket.send(_reply(request, "candles", candles=candles))

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        candles = await client.backfill_candles(
            "R_100",
            days=1000 * 60 / 86_400,
            granularity=60,
            end_epoch=end_epoch,
            page_size=5000,
        )

        assert len(candles) == 1001
        assert candles[0].epoch == start_epoch
        assert candles[-1].epoch == end_epoch
        assert len(requests) == 2
        assert all(request["count"] == 5000 for request in requests)
        assert [request["end"] for request in requests] == [
            end_epoch,
            end_epoch - 999 * 60 - 1,
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_seven_day_candle_backfill_uses_at_most_three_history_requests(
    fake_websocket_server: ServerFactory,
) -> None:
    """Seven days of 60-second history are fetched in three or fewer API calls."""
    requests: list[dict[str, Any]] = []
    end_epoch = 1_800_000_000 // 60 * 60
    start_epoch = end_epoch - 7 * 86_400

    async def handler(websocket: ServerConnection) -> None:
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
                continue
            requests.append(request)
            first_epoch = request["end"] // 60 * 60
            page = list(
                range(
                    first_epoch,
                    max(start_epoch, first_epoch - request["count"] * 60) - 1,
                    -60,
                )
            )[: request["count"]]
            candles = [
                {
                    "epoch": epoch,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                }
                for epoch in reversed(page)
            ]
            await websocket.send(_reply(request, "candles", candles=candles))

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect(validate_symbols=False)
        candles = await client.backfill_candles(
            "R_50",
            7,
            granularity=60,
            end_epoch=end_epoch,
            page_size=5000,
        )

        assert len(candles) == 7 * 24 * 60 + 1
        assert len(requests) <= 3
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_malformed_history_response_raises_value_error(
    fake_websocket_server: ServerFactory,
) -> None:
    """History response shape and model validation errors are not suppressed."""
    tick_requests = 0

    async def handler(websocket: ServerConnection) -> None:
        nonlocal tick_requests
        async for raw in websocket:
            request = json.loads(raw)
            if "active_symbols" in request:
                await websocket.send(_active_symbols_response(request))
            elif request["style"] == "ticks":
                if tick_requests == 0:
                    await websocket.send(
                        _reply(
                            request,
                            "history",
                            history={"times": [100], "prices": []},
                        )
                    )
                else:
                    await websocket.send(_reply(request, "history", history={"times": [100]}))
                tick_requests += 1
            else:
                await websocket.send(
                    _reply(
                        request,
                        "candles",
                        candles=[{"epoch": "invalid"}],
                    )
                )

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        with pytest.raises(ValueError, match="different lengths"):
            await client.backfill_ticks("R_75", 1, end_epoch=100)
        with pytest.raises(ValueError, match="malformed history"):
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
                response = json.loads(json.dumps(_PROTOCOL_FIXTURES["tick_subscription"]))
                response["req_id"] = request["req_id"]
                response["echo_req"] = request
                response["subscription"]["id"] = sub_id
                response["tick"]["id"] = sub_id
                response["tick"]["symbol"] = request["ticks"]
                await websocket.send(json.dumps(response))
                if connection_number == 1:
                    await websocket.close(code=1012, reason="restart")
                    return
                resubscribed.set()
            elif "ping" in request:
                await websocket.send(_fixture_response("ping", request))
            elif "forget" in request:
                await websocket.send(_reply(request, "forget", forget=request["forget"]))

    client = _client(
        await fake_websocket_server(handler),
        heartbeat_interval=0.03,
    )
    try:
        await client.connect()
        stream = await client.subscribe_ticks("R_75")
        initial_tick = await asyncio.wait_for(stream.get(), timeout=2)
        await asyncio.wait_for(resubscribed.wait(), timeout=2)
        tick = await asyncio.wait_for(stream.get(), timeout=2)
        assert isinstance(initial_tick, Tick)
        assert isinstance(tick, Tick)
        assert tick.epoch == 1791467374
        assert tick.quote == Decimal("628.49")
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
                    _reply(
                        request,
                        "candles",
                        candles=[],
                        subscription={"id": subscription_id},
                    )
                )
                update = json.loads(json.dumps(_PROTOCOL_FIXTURES["ohlc_update"]))
                update["req_id"] = request["req_id"]
                update["echo_req"] = request
                update["subscription"]["id"] = subscription_id
                update["ohlc"]["id"] = subscription_id
                update["ohlc"]["symbol"] = request["ticks_history"]
                await websocket.send(json.dumps(update))
            elif "forget" in request:
                await websocket.send(_reply(request, "forget", forget=request["forget"]))

    client = _client(await fake_websocket_server(handler))
    try:
        await client.connect()
        stream = await client.subscribe_ohlc("R_75", 60)
        candle = await asyncio.wait_for(stream.get(), timeout=1)
        assert isinstance(candle, Candle)
        assert candle.symbol == "R_75"
        assert candle.epoch == 1791467580
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
                        _reply(
                            request,
                            "tick",
                            subscription={"id": "initial-subscription"},
                            tick={
                                "symbol": request["ticks"],
                                "epoch": 1791467374,
                                "quote": "628.49",
                                "id": "initial-subscription",
                                "ask": "628.59",
                                "bid": "628.39",
                                "pip_size": 2,
                            },
                        )
                    )
                    await websocket.close(code=1012, reason="force resubscription")
                    return
                await websocket.send(
                    _reply(
                        request,
                        "tick",
                        error={
                            "code": "InvalidSymbol",
                            "details": {},
                            "message": "Subscription restore rejected",
                        },
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
        initial_tick = await asyncio.wait_for(stream.get(), timeout=1)
        assert isinstance(initial_tick, Tick)
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
                await websocket.send(_fixture_response("ping", request))

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
                await websocket.send(_fixture_response("ping", request))
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


@pytest.mark.asyncio
async def test_close_during_heartbeat_request_is_quiet_and_does_not_reconnect(
    fake_websocket_server: ServerFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Shutdown completes an in-flight heartbeat without logging or reconnecting."""
    connection_count = 0
    heartbeat_received = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        nonlocal connection_count
        connection_count += 1
        async for raw in websocket:
            request = json.loads(raw)
            if "ping" in request:
                heartbeat_received.set()

    client = _client(
        await fake_websocket_server(handler),
        heartbeat_interval=0.01,
    )
    try:
        await client.connect(validate_symbols=False)
        await asyncio.wait_for(heartbeat_received.wait(), timeout=1)
        await client.close()
        await asyncio.sleep(0)

        output = capsys.readouterr().out
        assert "deriv_connection_manager_unexpected_error" not in output
        assert "deriv_websocket_connection_failed" not in output
        assert connection_count == 1
    finally:
        await client.close()
