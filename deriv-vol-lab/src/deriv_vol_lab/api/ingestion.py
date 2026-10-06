"""Short-lived snapshot ingestion job; never starts a persistent WebSocket worker."""

from __future__ import annotations

import time

from deriv_vol_lab.api.snapshot import build_snapshot, write_snapshot
from deriv_vol_lab.data import SYMBOLS, DerivWebSocketClient
from deriv_vol_lab.data.models import Candle
from deriv_vol_lab.settings import Settings


async def ingest_latest_snapshot(*, days: float = 2.0, seed: int = 0) -> int:
    """Fetch recent 60-second candles and atomically persist the current snapshot."""
    if days <= 0:
        raise ValueError("days must be greater than zero")
    settings = Settings()
    if settings.deriv_app_id is None:
        raise ValueError("DERIV_APP_ID must be configured for scheduled ingestion")

    client = DerivWebSocketClient(settings.deriv_app_id)
    candles_by_symbol: dict[str, list[Candle]] = {}
    try:
        await client.connect(validate_symbols=True)
        for symbol in SYMBOLS:
            candles_by_symbol[symbol] = await client.backfill_candles(
                symbol,
                days,
                granularity=60,
            )
    finally:
        await client.close()

    snapshot = build_snapshot(
        candles_by_symbol,
        generated_at_epoch=int(time.time()),
        seed=seed,
    )
    write_snapshot(snapshot, settings.snapshot_path)
    return snapshot.candle_count
