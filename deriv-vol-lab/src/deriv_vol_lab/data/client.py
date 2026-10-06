"""Async market-data client for the Deriv public WebSocket API."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import AsyncIterator, Mapping
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import urlencode

import structlog
from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from deriv_vol_lab.data.models import Candle, Tick
from deriv_vol_lab.data.symbols import SYMBOLS

logger = structlog.get_logger(__name__)

DEFAULT_ENDPOINT = "wss://ws.derivws.com/websockets/v3"
MAX_HISTORY_COUNT = 5000
_VOLATILITY_CODE = re.compile(r"^(?:R_\d+|1HZ\d+V)$")
_StreamItem = Tick | Candle | BaseException
_StreamKind = Literal["ticks", "ohlc"]


class DerivAPIError(RuntimeError):
    """An error returned in a Deriv API response."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class ClientClosedError(RuntimeError):
    """Raised when an operation is attempted on a closed client."""


class ConnectionLostError(ConnectionError):
    """A retryable failure caused by losing the WebSocket connection."""

    retryable = True


class SubscriptionClosed(RuntimeError):
    """Raised when reading from a closed subscription."""


class SubscriptionHandle:
    """Queue-backed async stream for one live market-data subscription."""

    def __init__(
        self,
        client: DerivWebSocketClient,
        kind: _StreamKind,
        symbol: str,
        granularity: int | None,
    ) -> None:
        self._client = client
        self.kind = kind
        self.symbol = symbol
        self.granularity = granularity
        self.subscription_id: str | None = None
        self._queue: asyncio.Queue[_StreamItem] = asyncio.Queue()
        self._closed = False

    async def get(self) -> Tick | Candle:
        """Wait for the next update, raising queued errors to the consumer."""
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def __aiter__(self) -> AsyncIterator[Tick | Candle]:
        return self

    async def __anext__(self) -> Tick | Candle:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        try:
            return await self.get()
        except SubscriptionClosed:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        """Unsubscribe from the stream and stop its async iterator."""
        if not self._closed:
            await self._client._unsubscribe(self)

    def _enqueue(self, item: _StreamItem) -> None:
        if not self._closed:
            self._queue.put_nowait(item)

    def _stop(self, error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(error or SubscriptionClosed())


class DerivWebSocketClient:
    """Correlated, rate-limited WebSocket access to Deriv market data."""

    def __init__(
        self,
        app_id: int,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        requests_per_second: float = 5.0,
        heartbeat_interval: float = 30.0,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        request_timeout: float = 30.0,
        jitter_seed: int = 0,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than zero")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be greater than zero")
        if reconnect_initial_delay <= 0 or reconnect_max_delay < reconnect_initial_delay:
            raise ValueError("Reconnect delays must be positive and max must be >= initial")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        separator = "&" if "?" in endpoint else "?"
        self._uri = f"{endpoint}{separator}{urlencode({'app_id': app_id})}"
        self._minimum_request_interval = 1 / requests_per_second
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._request_timeout = request_timeout
        self._rng = random.Random(jitter_seed)

        self._websocket: ClientConnection | None = None
        self._manager_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._closed = False
        self._next_request_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._pending_subscriptions: dict[int, SubscriptionHandle] = {}
        self._subscriptions: dict[int, SubscriptionHandle] = {}
        self._subscription_ids: dict[str, SubscriptionHandle] = {}
        self._send_lock = asyncio.Lock()
        self._rate_limit_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._validation_lock = asyncio.Lock()
        self._symbols_validated = False

    async def connect(
        self,
        *,
        connection_timeout: float = 15.0,
        validate_symbols: bool = True,
    ) -> None:
        """Start the connection manager and validate the symbol registry."""
        if self._closed:
            raise ClientClosedError("Deriv WebSocket client is closed")
        if connection_timeout <= 0:
            raise ValueError("connection_timeout must be greater than zero")
        if self._manager_task is None:
            self._manager_task = asyncio.create_task(self._connection_manager())
        async with asyncio.timeout(connection_timeout):
            await self._connected.wait()
        if self._closed:
            raise ClientClosedError("Deriv WebSocket client is closed")
        if validate_symbols and not self._symbols_validated:
            async with self._validation_lock:
                if not self._symbols_validated:
                    await self.validate_symbol_registry()
                    self._symbols_validated = True

    async def close(self) -> None:
        """Close all streams and stop reconnecting."""
        if self._closed:
            return
        self._closed = True
        self._connected.set()
        for handle in tuple(self._subscriptions.values()):
            handle._stop()
        self._subscriptions.clear()
        self._subscription_ids.clear()
        self._fail_pending(ClientClosedError("Deriv WebSocket client was closed"))
        websocket = self._websocket
        if websocket is not None:
            await websocket.close()
        task = self._manager_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._manager_task = None

    async def __aenter__(self) -> DerivWebSocketClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Send one API request and return its response by request ID."""
        if self._closed:
            raise ClientClosedError("Deriv WebSocket client is closed")
        if self._manager_task is None:
            await self.connect()
        return await self._request(payload)

    async def _request(
        self,
        payload: Mapping[str, Any],
        *,
        subscription_handle: SubscriptionHandle | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ClientClosedError("Deriv WebSocket client is closed")
        await self._apply_rate_limit()
        while True:
            if self._closed:
                raise ClientClosedError("Deriv WebSocket client is closed")
            await self._connected.wait()
            if self._closed:
                raise ClientClosedError("Deriv WebSocket client is closed")
            if self._connected.is_set():
                break

        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        if subscription_handle is not None:
            self._pending_subscriptions[request_id] = subscription_handle
        message = dict(payload)
        message["req_id"] = request_id

        try:
            async with self._send_lock:
                if self._closed:
                    raise ClientClosedError("Deriv WebSocket client is closed")
                websocket = self._websocket
                if websocket is None:
                    raise ConnectionLostError("WebSocket disconnected before request could be sent")
                try:
                    await websocket.send(json.dumps(message))
                except (ConnectionClosed, ConnectionError, OSError, WebSocketException) as exc:
                    raise ConnectionLostError(
                        "WebSocket disconnected while sending request"
                    ) from exc
            return await asyncio.wait_for(future, timeout=self._request_timeout)
        finally:
            self._pending.pop(request_id, None)
            self._pending_subscriptions.pop(request_id, None)
            if future.done() and not future.cancelled():
                future.exception()

    async def validate_symbol_registry(self) -> None:
        """Compare the local volatility-index registry with active_symbols."""
        response = await self.request({"active_symbols": "brief", "product_type": "basic"})
        active_symbols = response.get("active_symbols")
        if not isinstance(active_symbols, list):
            raise ValueError("Deriv active_symbols response did not contain a symbol list")

        active_codes = {
            symbol["symbol"]
            for symbol in active_symbols
            if isinstance(symbol, dict)
            and isinstance(symbol.get("symbol"), str)
            and _VOLATILITY_CODE.fullmatch(symbol["symbol"])
        }
        registered_codes = set(SYMBOLS)
        missing = sorted(registered_codes - active_codes)
        unregistered = sorted(active_codes - registered_codes)
        if missing or unregistered:
            logger.warning(
                "symbol_registry_mismatch",
                missing_from_active_symbols=missing,
                missing_from_registry=unregistered,
            )

    async def backfill_ticks(
        self,
        symbol: str,
        days: float,
        *,
        end_epoch: int | None = None,
        page_size: int = MAX_HISTORY_COUNT,
    ) -> list[Tick]:
        """Fetch up to ``days`` of tick history by paging backwards."""
        items = await self._backfill(
            symbol,
            days,
            style="ticks",
            granularity=None,
            end_epoch=end_epoch,
            page_size=page_size,
        )
        if not all(isinstance(item, Tick) for item in items):
            raise TypeError("Tick history returned a non-tick item")
        return [item for item in items if isinstance(item, Tick)]

    async def backfill_candles(
        self,
        symbol: str,
        days: float,
        *,
        granularity: int,
        end_epoch: int | None = None,
        page_size: int = MAX_HISTORY_COUNT,
    ) -> list[Candle]:
        """Fetch up to ``days`` of OHLC history by paging backwards."""
        if granularity <= 0:
            raise ValueError("granularity must be greater than zero")
        items = await self._backfill(
            symbol,
            days,
            style="candles",
            granularity=granularity,
            end_epoch=end_epoch,
            page_size=page_size,
        )
        if not all(isinstance(item, Candle) for item in items):
            raise TypeError("Candle history returned a non-candle item")
        return [item for item in items if isinstance(item, Candle)]

    async def subscribe_ticks(self, symbol: str) -> SubscriptionHandle:
        """Subscribe to live ticks for a symbol."""
        return await self._subscribe(symbol, "ticks", None)

    async def subscribe_ohlc(self, symbol: str, granularity: int) -> SubscriptionHandle:
        """Subscribe to live OHLC updates for a symbol."""
        if granularity <= 0:
            raise ValueError("granularity must be greater than zero")
        return await self._subscribe(symbol, "ohlc", granularity)

    async def _backfill(
        self,
        symbol: str,
        days: float,
        *,
        style: Literal["ticks", "candles"],
        granularity: int | None,
        end_epoch: int | None,
        page_size: int,
    ) -> list[Tick | Candle]:
        if days <= 0:
            raise ValueError("days must be greater than zero")
        if not 1 <= page_size <= MAX_HISTORY_COUNT:
            raise ValueError(f"page_size must be between 1 and {MAX_HISTORY_COUNT}")

        end = int(time.time()) if end_epoch is None else end_epoch
        start = end - int(days * 86_400)
        cursor = end
        results: dict[tuple[int, Decimal] | int, Tick | Candle] = {}
        previous_oldest: int | None = None

        while cursor >= start:
            payload: dict[str, Any] = {
                "ticks_history": symbol,
                "style": style,
                "count": page_size,
                "end": cursor,
                "adjust_start": 1,
            }
            if granularity is not None:
                payload["granularity"] = granularity

            response = await self.request(payload)
            page = self._parse_history(response, symbol, style, granularity)
            if not page:
                break

            epochs = [item.epoch for item in page]
            oldest = min(epochs)
            if previous_oldest is not None and oldest >= previous_oldest:
                raise RuntimeError(
                    f"History pagination did not move backwards for {symbol}: {oldest}"
                )
            previous_oldest = oldest

            for item in page:
                if not start <= item.epoch <= end:
                    continue
                key: tuple[int, Decimal] | int
                if isinstance(item, Tick):
                    key = (item.epoch, item.quote)
                else:
                    key = item.epoch
                results[key] = item

            if oldest <= start or len(page) < page_size:
                break
            cursor = oldest - 1

        return sorted(results.values(), key=lambda item: item.epoch)

    @staticmethod
    def _parse_history(
        response: Mapping[str, Any],
        symbol: str,
        style: Literal["ticks", "candles"],
        granularity: int | None,
    ) -> list[Tick] | list[Candle]:
        try:
            if style == "candles":
                candles = response.get("candles")
                if not isinstance(candles, list):
                    raise ValueError("Deriv candles response did not contain candle data")
                assert granularity is not None
                return [
                    Candle.model_validate({**item, "symbol": symbol, "granularity": granularity})
                    for item in candles
                ]

            history = response.get("history")
            if not isinstance(history, dict):
                raise ValueError("Deriv ticks response did not contain history data")
            times = history.get("times")
            prices = history.get("prices")
            if not isinstance(times, list) or not isinstance(prices, list):
                raise ValueError("Deriv tick history must contain times and prices")
            if len(times) != len(prices):
                raise ValueError("Deriv tick history times and prices have different lengths")
            return [
                Tick(symbol=symbol, epoch=epoch, quote=quote)
                for epoch, quote in zip(times, prices, strict=True)
            ]
        except (TypeError, ValidationError) as exc:
            raise ValueError("Deriv returned malformed history data") from exc

    async def _subscribe(
        self,
        symbol: str,
        kind: _StreamKind,
        granularity: int | None,
    ) -> SubscriptionHandle:
        if self._closed:
            raise ClientClosedError("Deriv WebSocket client is closed")
        await self.connect()
        handle = SubscriptionHandle(self, kind, symbol, granularity)
        self._subscriptions[id(handle)] = handle
        try:
            await self._send_subscription(handle)
        except BaseException:
            self._subscriptions.pop(id(handle), None)
            raise
        return handle

    async def _send_subscription(self, handle: SubscriptionHandle) -> None:
        payload: dict[str, Any]
        if handle.kind == "ticks":
            payload = {"ticks": handle.symbol, "subscribe": 1}
        else:
            payload = {
                "ticks_history": handle.symbol,
                "style": "candles",
                "granularity": handle.granularity,
                "subscribe": 1,
            }

        response = await self._request(payload, subscription_handle=handle)
        subscription = response.get("subscription")
        subscription_id = subscription.get("id") if isinstance(subscription, dict) else None
        if not isinstance(subscription_id, (str, int)):
            raise ValueError("Deriv subscription response did not contain a subscription ID")
        handle.subscription_id = str(subscription_id)
        self._subscription_ids[handle.subscription_id] = handle

    async def _unsubscribe(self, handle: SubscriptionHandle) -> None:
        handle._stop()
        self._remove_subscription(handle)
        if handle.subscription_id is not None:
            await self.request({"forget": handle.subscription_id})
            handle.subscription_id = None

    def _remove_subscription(self, handle: SubscriptionHandle) -> None:
        self._subscriptions.pop(id(handle), None)
        if handle.subscription_id is not None:
            self._subscription_ids.pop(handle.subscription_id, None)

    def _fail_pending(self, error: BaseException) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)

    async def _connection_manager(self) -> None:
        delay = self._reconnect_initial_delay
        while not self._closed:
            websocket: ClientConnection | None = None
            reader_task: asyncio.Task[None] | None = None
            heartbeat_task: asyncio.Task[None] | None = None
            connection_started: float | None = None
            disconnect_error = ConnectionLostError("Deriv WebSocket disconnected")
            try:
                websocket = await websocket_connect(self._uri, ping_interval=None)
                self._websocket = websocket
                self._subscription_ids.clear()
                connection_started = asyncio.get_running_loop().time()
                self._connected.set()
                reader_task = asyncio.create_task(self._read_messages(websocket))
                heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                await self._restore_subscriptions()
                done, _ = await asyncio.wait(
                    (reader_task, heartbeat_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for finished_task in done:
                    finished_task.result()
                raise ConnectionError("Deriv WebSocket connection ended")
            except (
                ConnectionClosed,
                ConnectionError,
                OSError,
                WebSocketException,
                TimeoutError,
            ) as exc:
                if not self._closed:
                    logger.warning(
                        "deriv_websocket_disconnected",
                        error=str(exc),
                        reconnect_delay_seconds=delay,
                    )
            except ValueError as exc:
                disconnect_error = ConnectionLostError(f"Deriv WebSocket protocol failure: {exc}")
                disconnect_error.__cause__ = exc
                logger.error("deriv_websocket_protocol_error", error=str(exc))
            except Exception as exc:
                disconnect_error = ConnectionLostError(
                    f"Connection manager failed unexpectedly: {exc}"
                )
                disconnect_error.__cause__ = exc
                logger.error(
                    "deriv_connection_manager_unexpected_error",
                    error=str(exc),
                    exc_info=True,
                    reconnect_delay_seconds=delay,
                )
            finally:
                if (
                    connection_started is not None
                    and asyncio.get_running_loop().time() - connection_started
                    >= self._heartbeat_interval
                ):
                    delay = self._reconnect_initial_delay
                self._connected.clear()
                self._websocket = None
                self._subscription_ids.clear()
                if not self._closed:
                    self._fail_pending(disconnect_error)
                for child_task in (reader_task, heartbeat_task):
                    if child_task is not None and not child_task.done():
                        child_task.cancel()
                child_tasks = [
                    child_task
                    for child_task in (reader_task, heartbeat_task)
                    if child_task is not None
                ]
                if child_tasks:
                    await asyncio.gather(*child_tasks, return_exceptions=True)
                if not self._closed and websocket is not None:
                    try:
                        await websocket.close()
                    except Exception as exc:
                        logger.error(
                            "deriv_websocket_cleanup_failed",
                            error=str(exc),
                            exc_info=True,
                        )

            if not self._closed:
                await asyncio.sleep(self._rng.uniform(0, delay))
                delay = min(delay * 2, self._reconnect_max_delay)

    async def _read_messages(self, websocket: ClientConnection) -> None:
        async for raw_message in websocket:
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                raise ValueError("Deriv sent invalid JSON") from exc
            if not isinstance(message, dict):
                raise ValueError("Deriv sent a non-object WebSocket message")
            self._dispatch(message)
        raise ConnectionError("Deriv WebSocket closed")

    def _dispatch(self, message: dict[str, Any]) -> None:
        request_id = message.get("req_id")
        if isinstance(request_id, int) and not isinstance(request_id, bool):
            subscription = message.get("subscription")
            subscription_id = subscription.get("id") if isinstance(subscription, dict) else None
            pending_handle = self._pending_subscriptions.get(request_id)
            if pending_handle is not None and isinstance(subscription_id, (str, int)):
                pending_handle.subscription_id = str(subscription_id)
                self._subscription_ids[str(subscription_id)] = pending_handle
            future = self._pending.get(request_id)
            if future is not None and not future.done():
                error = message.get("error")
                if isinstance(error, dict):
                    future.set_exception(
                        DerivAPIError(
                            str(error.get("code", "UnknownError")),
                            str(error.get("message", "Deriv API request failed")),
                        )
                    )
                else:
                    future.set_result(message)

        subscription = message.get("subscription")
        subscription_id = subscription.get("id") if isinstance(subscription, dict) else None
        if subscription_id is None:
            return
        handle = self._subscription_ids.get(str(subscription_id))
        if handle is None:
            return
        event = message.get("tick") if handle.kind == "ticks" else message.get("ohlc")
        if not isinstance(event, dict):
            error = message.get("error")
            if isinstance(error, dict):
                handle._enqueue(
                    DerivAPIError(
                        str(error.get("code", "UnknownError")),
                        str(error.get("message", "Deriv subscription failed")),
                    )
                )
            return
        try:
            item: Tick | Candle
            if handle.kind == "ticks":
                item = Tick.model_validate(event)
            else:
                assert handle.granularity is not None
                item = Candle.model_validate(
                    {
                        **event,
                        "symbol": handle.symbol,
                        "granularity": handle.granularity,
                    }
                )
            handle._enqueue(item)
        except ValidationError as exc:
            handle._enqueue(ValueError("Deriv sent a malformed subscription update"))
            logger.error("invalid_deriv_subscription_update", error=str(exc))

    async def _heartbeat_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._heartbeat_interval)
            await self.request({"ping": 1})

    async def _restore_subscriptions(self) -> None:
        for handle in tuple(self._subscriptions.values()):
            if handle._closed:
                continue
            try:
                await self._send_subscription(handle)
            except (DerivAPIError, ValueError) as exc:
                self._remove_subscription(handle)
                handle.subscription_id = None
                handle._stop(exc)
                error_code = exc.code if isinstance(exc, DerivAPIError) else "InvalidResponse"
                error_message = exc.message if isinstance(exc, DerivAPIError) else str(exc)
                logger.error(
                    "deriv_subscription_restore_failed",
                    symbol=handle.symbol,
                    error_code=error_code,
                    error_message=error_message,
                )

    async def _apply_rate_limit(self) -> None:
        async with self._rate_limit_lock:
            now = asyncio.get_running_loop().time()
            wait = max(0.0, self._next_request_at - now)
            if wait:
                await asyncio.sleep(wait)
            self._next_request_at = max(now, self._next_request_at) + self._minimum_request_interval
