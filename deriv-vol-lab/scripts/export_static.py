"""Export the dashboard and its JSON endpoints as a static site for Vercel.

The FastAPI app serves read-only data from a snapshot file and a frozen research
summary. Neither needs a server at request time, so this script renders every
endpoint once into ``public/`` and Vercel serves the result as plain files.

Usage:
    python scripts/export_static.py [--snapshot PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deriv_vol_lab.api.app import (
    DataQualityResponse,
    ScreenResponse,
    VolatilityMonitorResponse,
)
from deriv_vol_lab.api.ledger import (
    LedgerUnavailableError,
    load_ledger,
    summarise,
)
from deriv_vol_lab.api.research import ResearchUnavailableError, load_research_summary
from deriv_vol_lab.api.snapshot import (
    SnapshotUnavailableError,
    default_snapshot_path,
    load_snapshot,
)
from deriv_vol_lab.api.technical_read import build_technical_read

# Public URL path (as the dashboard fetches it) -> static file written to disk.
ROUTES: dict[str, str] = {
    "/screen": "screen.json",
    "/data-quality": "data-quality.json",
    "/stats/vol-monitor": "vol-monitor.json",
    "/research/flag-edge": "flag-edge.json",
    "/technical-read": "technical-read.json",
    "/ledger": "ledger.json",
}


def _dump(path: Path, model: BaseModel) -> None:
    payload: Any = json.loads(model.model_dump_json())
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def export(
    out_dir: Path,
    snapshot_path: Path,
    ledger_path: Path | None = None,
) -> list[str]:
    """Write the dashboard and all available JSON files; return the names written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    html = files("deriv_vol_lab.api").joinpath("dashboard.html").read_text(encoding="utf-8")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    written.append("index.html")

    try:
        snapshot = load_snapshot(snapshot_path)
    except SnapshotUnavailableError as exc:
        print(f"warning: no snapshot ({exc}); live-data files not refreshed", file=sys.stderr)
    else:
        builders: dict[str, Callable[[], BaseModel]] = {
            "screen.json": lambda: ScreenResponse(
                generated_at_epoch=snapshot.generated_at_epoch,
                granularity=snapshot.granularity,
                rows=snapshot.screen,
            ),
            "data-quality.json": lambda: DataQualityResponse(
                generated_at_epoch=snapshot.generated_at_epoch,
                by_symbol=snapshot.quality_by_symbol,
                report=snapshot.quality,
            ),
            "vol-monitor.json": lambda: VolatilityMonitorResponse(
                generated_at_epoch=snapshot.generated_at_epoch,
                report=snapshot.volatility_monitor,
            ),
        }
        builders["technical-read.json"] = lambda: build_technical_read(snapshot)
        for name, build in builders.items():
            _dump(out_dir / name, build())
            written.append(name)

    try:
        ledger = load_ledger(ledger_path or snapshot_path.with_name("ledger.json"))
        _dump(out_dir / "ledger.json", summarise(ledger))
        written.append("ledger.json")
    except LedgerUnavailableError as exc:
        print(f"warning: ledger unreadable ({exc})", file=sys.stderr)

    try:
        _dump(out_dir / "flag-edge.json", load_research_summary())
        written.append("flag-edge.json")
    except ResearchUnavailableError as exc:
        print(f"warning: no research summary ({exc})", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("public"))
    args = parser.parse_args(argv)
    written = export(args.out, args.snapshot or default_snapshot_path())
    print("wrote: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
