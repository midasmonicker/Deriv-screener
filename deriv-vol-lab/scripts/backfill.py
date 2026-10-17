"""Resumable backfill of 60-second Deriv volatility-index candles."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from deriv_vol_lab.data import (
    SYMBOLS,
    Candle,
    DerivWebSocketClient,
    DuckDBStorage,
    HandshakeConnectionError,
)
from deriv_vol_lab.settings import Settings

GRANULARITY_SECONDS = 60
SECONDS_PER_DAY = 86_400
PAGE_SIZE = 1_000
RESUME_OVERLAP_CANDLES = 2
_T = TypeVar("_T")


class _BackfillDeadlineExceeded(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _progress(message: str) -> None:
    print(f"[backfill] {message}", file=sys.stderr, flush=True)


async def _await_with_deadline(
    operation: Callable[[], Awaitable[_T]],
    deadline: float | None,
    stage: str,
) -> _T:
    if deadline is None:
        return await operation()
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise _BackfillDeadlineExceeded(stage)
    try:
        return await asyncio.wait_for(operation(), timeout=remaining)
    except TimeoutError as exc:
        raise _BackfillDeadlineExceeded(stage) from exc


def _check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise _BackfillDeadlineExceeded(stage)


def _coverage_percentage(
    storage: DuckDBStorage,
    symbol: str,
    start_epoch: int,
    end_epoch: int,
) -> None:
    report = storage.quality_report(
        symbol=symbol,
        granularity=GRANULARITY_SECONDS,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )
    gaps = storage.find_gaps(
        symbol,
        start_epoch,
        end_epoch,
        granularity=GRANULARITY_SECONDS,
    )
    first_expected_epoch = (
        (start_epoch + GRANULARITY_SECONDS - 1) // GRANULARITY_SECONDS * GRANULARITY_SECONDS
    )
    expected = (
        (end_epoch - first_expected_epoch) // GRANULARITY_SECONDS + 1
        if first_expected_epoch <= end_epoch
        else 0
    )
    observed = expected - len(gaps)
    coverage = 100.0 * observed / expected if expected else 0.0
    _progress(
        f"{symbol}: quality gaps={len(gaps)}, duplicates={len(report.duplicate_epochs)}, "
        f"coverage={coverage:.3f}%"
    )


async def _backfill_symbol(
    client: DerivWebSocketClient,
    storage: DuckDBStorage,
    symbol: str,
    *,
    start_epoch: int,
    end_epoch: int,
    deadline: float | None,
) -> None:
    _check_deadline(deadline, f"read {symbol}")
    stored = (
        storage.read_candles(
            GRANULARITY_SECONDS,
            symbol,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
        )
        if storage.has_candle_table(GRANULARITY_SECONDS)
        else []
    )
    known_epochs = {candle.epoch for candle in stored}
    missing_epochs = storage.find_gaps(
        symbol,
        start_epoch,
        end_epoch,
        granularity=GRANULARITY_SECONDS,
    )
    latest_stored_epoch = max(known_epochs, default=None)
    resume_epoch = (
        start_epoch
        if latest_stored_epoch is None
        else max(
            start_epoch,
            latest_stored_epoch - RESUME_OVERLAP_CANDLES * GRANULARITY_SECONDS,
        )
    )
    target_start = min(resume_epoch, min(missing_epochs, default=resume_epoch))
    _progress(
        f"{symbol}: started; stored={len(stored)}, missing={len(missing_epochs)}, "
        f"resume_start={target_start}"
    )

    page_number = 0
    cursor = end_epoch
    while missing_epochs and cursor >= target_start:
        page_started = time.perf_counter()

        async def request_page(request_end: int = cursor) -> dict[str, Any]:
            return await client.request(
                {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": GRANULARITY_SECONDS,
                    "count": PAGE_SIZE,
                    "end": request_end,
                }
            )

        response = await _await_with_deadline(
            request_page,
            deadline,
            f"fetch {symbol}",
        )
        raw_candles = response.get("candles")
        if not isinstance(raw_candles, list):
            raise ValueError(f"Deriv candles response for {symbol} did not contain candle data")
        page_number += 1
        if not raw_candles:
            _progress(
                f"{symbol}: page {page_number} empty; stopped after "
                f"{time.perf_counter() - page_started:.2f}s"
            )
            break

        candles: list[Candle] = []
        for item in raw_candles:
            if not isinstance(item, dict):
                raise ValueError(f"Deriv candles response for {symbol} contained invalid data")
            candle = Candle.model_validate(
                {**item, "symbol": symbol, "granularity": GRANULARITY_SECONDS}
            )
            if start_epoch <= candle.epoch <= end_epoch and candle.epoch not in known_epochs:
                candles.append(candle)
                known_epochs.add(candle.epoch)

        page_epochs = [
            int(item["epoch"]) for item in raw_candles if isinstance(item, dict) and "epoch" in item
        ]
        if not page_epochs:
            raise ValueError(f"Deriv candles response for {symbol} had no candle epochs")
        oldest_epoch = min(page_epochs)
        if oldest_epoch > cursor:
            raise RuntimeError(
                f"History pagination did not move backwards for {symbol}: {oldest_epoch}"
            )
        if candles:
            storage.write_candles(candles)
        _check_deadline(deadline, f"write {symbol}")
        _progress(
            f"{symbol}: page {page_number} fetched={len(raw_candles)}, "
            f"added={len(candles)}, oldest={oldest_epoch}, "
            f"elapsed={time.perf_counter() - page_started:.2f}s"
        )
        if oldest_epoch <= target_start:
            break
        cursor = oldest_epoch - GRANULARITY_SECONDS

    _check_deadline(deadline, f"quality report {symbol}")
    _coverage_percentage(storage, symbol, start_epoch, end_epoch)
    _check_deadline(deadline, f"quality report {symbol}")
    _progress(f"{symbol}: finished; pages={page_number}")


async def run_backfill(
    days: int = 60,
    *,
    max_seconds: float | None = None,
) -> int:
    """Fetch and persist missing 60-second candles for every registered symbol."""
    if days <= 0:
        raise ValueError("days must be a positive integer")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be a positive number")

    settings = Settings()
    deadline = time.perf_counter() + max_seconds if max_seconds is not None else None
    end_epoch = int(time.time()) // GRANULARITY_SECONDS * GRANULARITY_SECONDS
    end_epoch -= GRANULARITY_SECONDS
    start_epoch = end_epoch - days * SECONDS_PER_DAY + GRANULARITY_SECONDS
    client = DerivWebSocketClient(
        settings.deriv_app_id,
        endpoint=settings.deriv_endpoint,
        max_handshake_attempts=settings.deriv_handshake_max_attempts,
    )
    try:
        await _await_with_deadline(
            lambda: client.connect(validate_symbols=True),
            deadline,
            "connect",
        )
        with DuckDBStorage(settings.db_path) as storage:
            for symbol in SYMBOLS:
                _check_deadline(deadline, f"start {symbol}")
                await _backfill_symbol(
                    client,
                    storage,
                    symbol,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    deadline=deadline,
                )
    except HandshakeConnectionError as exc:
        _progress(f"handshake failed: {exc}")
        return 2
    except _BackfillDeadlineExceeded as exc:
        _progress(f"exceeded --max-seconds during {exc.stage}")
        return 2
    finally:
        await client.close()
    return 0


def main() -> None:
    """Run the resumable backfill command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="number of days to backfill (default: 60)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="abort when the total runtime exceeds this many seconds",
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be a positive number")
    raise SystemExit(asyncio.run(run_backfill(args.days, max_seconds=args.max_seconds)))


if __name__ == "__main__":
    main()
