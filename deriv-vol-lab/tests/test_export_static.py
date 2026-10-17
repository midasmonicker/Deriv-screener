"""The static export must match the live API byte-for-meaning and Vercel must route to it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from scripts import export_static
from tests.test_api import _candles_by_symbol

from deriv_vol_lab.api.app import app
from deriv_vol_lab.api.snapshot import build_snapshot, write_snapshot

ROOT = Path(__file__).resolve().parents[1]


def _fetch(path: str) -> object:
    async def run() -> object:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(path)
            assert response.status_code == 200
            return response.json()

    return asyncio.run(run())


def test_export_matches_live_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot_path = tmp_path / "latest.json"
    write_snapshot(
        build_snapshot(_candles_by_symbol(), generated_at_epoch=1_704_067_320), snapshot_path
    )
    monkeypatch.setenv("SNAPSHOT_PATH", str(snapshot_path))
    out = tmp_path / "public"

    written = export_static.export(out, snapshot_path)

    assert sorted(written) == sorted(["index.html", *export_static.ROUTES.values()])
    assert "<title>" in (out / "index.html").read_text(encoding="utf-8")
    for url, name in export_static.ROUTES.items():
        assert json.loads((out / name).read_text(encoding="utf-8")) == _fetch(url)


def test_missing_snapshot_still_exports_dashboard_and_research(tmp_path: Path) -> None:
    out = tmp_path / "public"

    written = export_static.export(out, tmp_path / "absent.json")

    assert sorted(written) == ["flag-edge.json", "index.html", "ledger.json"]
    assert not (out / "screen.json").exists()


def test_vercel_rewrites_cover_every_route() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}
    assert rewrites == {url: "/" + name for url, name in export_static.ROUTES.items()}
    assert config["outputDirectory"] == "public"
    assert "builds" not in config
    assert not (ROOT / "api").exists()
