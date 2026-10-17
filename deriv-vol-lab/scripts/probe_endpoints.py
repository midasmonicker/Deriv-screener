r"""
Standalone Deriv WebSocket probe v2 (no project imports, read-only, no token).
Run from the project folder:
    .\.venv\Scripts\python.exe scripts\probe_endpoints.py
Tests the NEW public market-data endpoint and, for comparison, the legacy hosts.
Prints the first part of each raw reply so you can see the real message shapes.
"""

import asyncio
import json

import websockets

TARGETS = [
    # (label, url, extra headers)
    ("NEW public, no headers", "wss://api.derivws.com/trading/v1/options/ws/public", {}),
    (
        "NEW public + Deriv-App-ID 1089",
        "wss://api.derivws.com/trading/v1/options/ws/public",
        {"Deriv-App-ID": "1089"},
    ),
    ("LEGACY ws.derivws.com", "wss://ws.derivws.com/websockets/v3", {}),
    ("LEGACY ws.binaryws.com", "wss://ws.binaryws.com/websockets/v3", {}),
]

CANDLES = {
    "ticks_history": "R_100",
    "adjust_start_time": 1,
    "count": 3,
    "end": "latest",
    "granularity": 60,
    "style": "candles",
}


async def connect(url: str, headers: dict):
    try:
        return await websockets.connect(url, open_timeout=15, additional_headers=headers)
    except TypeError:  # older websockets versions
        return await websockets.connect(url, open_timeout=15, extra_headers=headers)


async def probe(label: str, url: str, headers: dict) -> None:
    try:
        ws = await connect(url, headers)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"FAIL {label}: {type(exc).__name__} status={status}")
        return
    try:
        for name, msg in (("ping", {"ping": 1}), ("candles", CANDLES)):
            await ws.send(json.dumps(msg))
            raw = await asyncio.wait_for(ws.recv(), 15)
            print(f"OK   {label} [{name}] {str(raw)[:240]}")
    except Exception as exc:
        print(f"WARN {label}: connected but {type(exc).__name__}: {exc}")
    finally:
        await ws.close()


async def main() -> None:
    for label, url, headers in TARGETS:
        await probe(label, url, headers)


if __name__ == "__main__":
    asyncio.run(main())
