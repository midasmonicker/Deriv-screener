"""Tests for DuckDB persistence, gap backfills, and data-quality reports."""

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from deriv_vol_lab.data import (
    Candle,
    DuckDBStorage,
    Tick,
    build_quality_report,
)


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[DuckDBStorage]:
    """Provide an isolated DuckDB database for a test."""
    database = DuckDBStorage(tmp_path / "market.duckdb")
    yield database
    database.close()


def test_ticks_and_candles_upsert_by_symbol_epoch(storage: DuckDBStorage) -> None:
    """Repeated keys update records without increasing table row counts."""
    original = Tick(symbol="R_75", epoch=100, quote=Decimal("10.25"), id="first")
    replacement = Tick(symbol="R_75", epoch=100, quote=Decimal("11.5"), id="second")
    assert storage.write_ticks([original]) == 1
    assert storage.write_ticks([replacement]) == 1
    assert storage.read_ticks() == [replacement]

    candle = Candle(
        symbol="R_75",
        epoch=120,
        open=Decimal("10"),
        high=Decimal("12"),
        low=Decimal("9"),
        close=Decimal("11"),
        granularity=60,
    )
    changed_candle = candle.model_copy(update={"close": Decimal("11.5")})
    assert storage.write_candles([candle]) == 1
    assert storage.write_candles([changed_candle]) == 1
    assert storage.read_candles(60) == [changed_candle]
    assert storage._connection.execute("SELECT count(*) FROM candles_60").fetchone() == (1,)


def test_candle_write_requires_one_granularity(storage: DuckDBStorage) -> None:
    """A mixed candle batch cannot be written into one cadence table."""
    first = Candle(
        symbol="R_75",
        epoch=60,
        open=1,
        high=2,
        low=0.5,
        close=1.5,
        granularity=60,
    )
    second = first.model_copy(update={"epoch": 120, "granularity": 120})
    with pytest.raises(ValueError, match="share a granularity"):
        storage.write_candles([first, second])


def test_parquet_exports_ticks_and_candles(storage: DuckDBStorage, tmp_path: Path) -> None:
    """Both market-data table types can be read back from exported Parquet."""
    tick = Tick(symbol="R_75", epoch=100, quote=Decimal("12.5"))
    storage.write_ticks([tick])
    ticks_path = storage.export_parquet("ticks", tmp_path / "export" / "ticks.parquet")
    tick_rows = (
        duckdb.connect()
        .execute("SELECT symbol, epoch, quote FROM read_parquet(?)", [str(ticks_path)])
        .fetchall()
    )
    assert tick_rows == [("R_75", 100, Decimal("12.5"))]

    candle = Candle(
        symbol="1HZ75V",
        epoch=120,
        open=1,
        high=2,
        low=0.5,
        close=1.5,
        granularity=60,
    )
    storage.write_candles([candle])
    candle_path = storage.export_parquet(60, tmp_path / "candles.parquet")
    assert duckdb.connect().execute(
        "SELECT symbol, epoch FROM read_parquet(?)", [str(candle_path)]
    ).fetchall() == [("1HZ75V", 120)]


def test_gap_detector_finds_missing_tick_and_candle_epochs(storage: DuckDBStorage) -> None:
    """The detector finds only expected timestamps absent from each table."""
    storage.write_ticks(
        [
            Tick(symbol="R_75", epoch=100, quote=10),
            Tick(symbol="R_75", epoch=104, quote=11),
            Tick(symbol="R_75", epoch=110, quote=12),
        ]
    )
    assert storage.find_gaps("R_75", 100, 110) == [102, 106, 108]

    storage.write_candles(
        [
            Candle(
                symbol="R_75",
                epoch=120,
                open=1,
                high=2,
                low=0.5,
                close=1,
                granularity=60,
            ),
            Candle(
                symbol="R_75",
                epoch=240,
                open=1,
                high=2,
                low=0.5,
                close=1,
                granularity=60,
            ),
        ]
    )
    assert storage.find_gaps("R_75", 120, 240, granularity=60) == [180]


@pytest.mark.asyncio
async def test_gap_backfill_fetches_and_persists_only_remaining_gaps(
    storage: DuckDBStorage,
) -> None:
    """Backfill results are stored and the detector rechecks unresolved points."""
    storage.write_ticks([Tick(symbol="R_75", epoch=100, quote=10)])

    class FakeClient:
        async def backfill_ticks(self, symbol: str, days: float, *, end_epoch: int) -> list[Tick]:
            assert symbol == "R_75"
            assert days > 0
            return [
                Tick(symbol=symbol, epoch=100, quote=10),
                Tick(symbol=symbol, epoch=102, quote=11),
                Tick(symbol=symbol, epoch=106, quote=12),
            ]

    remaining = await storage.backfill_gaps(FakeClient(), "R_75", 100, 106)  # type: ignore[arg-type]
    assert remaining == [104]
    assert [tick.epoch for tick in storage.read_ticks("R_75")] == [100, 102, 106]


@pytest.mark.asyncio
async def test_candle_gap_backfill_uses_granularity_and_persists(
    storage: DuckDBStorage,
) -> None:
    """Candle gaps invoke the candle endpoint and upsert into its own table."""
    storage.write_candles(
        [
            Candle(
                symbol="R_75",
                epoch=120,
                open=1,
                high=2,
                low=0.5,
                close=1,
                granularity=60,
            )
        ]
    )

    class FakeClient:
        async def backfill_candles(
            self,
            symbol: str,
            days: float,
            *,
            granularity: int,
            end_epoch: int,
        ) -> list[Candle]:
            assert symbol == "R_75"
            assert granularity == 60
            return [
                Candle(
                    symbol=symbol,
                    epoch=epoch,
                    open=1,
                    high=2,
                    low=0.5,
                    close=1,
                    granularity=granularity,
                )
                for epoch in (120, 180, 240)
            ]

    remaining = await storage.backfill_gaps(
        FakeClient(),  # type: ignore[arg-type]
        "R_75",
        120,
        240,
        granularity=60,
    )
    assert remaining == []
    assert [candle.epoch for candle in storage.read_candles(60, "R_75")] == [120, 180, 240]


def test_quality_report_detects_duplicate_order_prices_jumps_and_coverage() -> None:
    """Injected bad observations produce actionable quality findings."""
    epochs = [100 + 2 * index for index in range(12)]
    returns = [0.001, -0.001, 0.001, -0.001, 0.001, 0.12, -0.001, 0.001, -0.001, 0.001, -0.001]
    price = 100.0
    prices = [price]
    for change in returns:
        price *= 2.718281828459045**change
        prices.append(price)
    ticks = [
        Tick(symbol="R_75", epoch=epoch, quote=Decimal(str(value)))
        for epoch, value in zip(epochs, prices, strict=True)
    ]
    del ticks[4]
    injected = [
        *ticks,
        Tick(symbol="R_75", epoch=200, quote=0),
        Tick(symbol="R_75", epoch=202, quote=-1),
        ticks[1],
        ticks[0],
    ]

    report = build_quality_report(
        injected,
        start_epoch=100,
        end_epoch=122,
        sigma_window=4,
    )

    assert [(item.epoch, item.occurrences) for item in report.duplicate_epochs] == [
        (100, 2),
        (102, 2),
    ]
    assert report.non_monotonic_timestamps
    assert {(issue.epoch, issue.price) for issue in report.invalid_prices} == {
        (200, 0.0),
        (202, -1.0),
    }
    assert any(item.epoch == 112 and item.sigma_multiple > 8 for item in report.abnormal_jumps)
    assert report.daily_coverage[0].expected_records == 12
    assert report.daily_coverage[0].observed_records == 11
    assert report.daily_coverage[0].coverage_pct == pytest.approx(100 * 11 / 12)


def test_quality_report_validates_inputs_and_unsupported_symbols() -> None:
    """Bad report parameters and unregistered tick cadence are explicit errors."""
    with pytest.raises(ValueError, match="sigma_threshold"):
        build_quality_report([], sigma_threshold=0)
    with pytest.raises(ValueError, match="sigma_window"):
        build_quality_report([], sigma_window=1)
    with pytest.raises(ValueError, match="start_epoch"):
        build_quality_report([], start_epoch=2, end_epoch=1)
    with pytest.raises(ValueError, match="No expected tick interval"):
        build_quality_report([Tick(symbol="unknown", epoch=1, quote=1)])


def test_quality_report_coverage_splits_at_utc_midnight() -> None:
    """Daily tick coverage is grouped on UTC midnight for its own series."""
    midnight = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp())
    records = [
        Tick(symbol="R_75", epoch=epoch, quote=100)
        for epoch in (midnight - 2, midnight, midnight + 2)
    ]

    report = build_quality_report(records)
    coverage = report.daily_coverage

    observed_by_day = [
        (item.day.isoformat(), item.expected_records, item.observed_records) for item in coverage
    ]
    assert observed_by_day == [
        ("2024-01-01", 1, 1),
        ("2024-01-02", 2, 2),
    ]
    assert all(item.record_kind == "tick" and item.granularity == 2 for item in coverage)


def test_storage_rejects_invalid_granularity_and_closed_connection(tmp_path: Path) -> None:
    """Invalid cadence names and use after close fail explicitly."""
    database = DuckDBStorage(tmp_path / "market.duckdb")
    with pytest.raises(ValueError, match="granularity"):
        database.find_gaps("R_75", 0, 100, granularity=0)
    database.close()
    with pytest.raises(RuntimeError, match="closed"):
        database.read_ticks()


def test_parquet_export_rejects_invalid_table_name(storage: DuckDBStorage, tmp_path: Path) -> None:
    """Only the ticks table or numeric candle granularities may be exported."""
    with pytest.raises(ValueError, match="table must"):
        storage.export_parquet("ticks; DROP TABLE ticks", tmp_path / "invalid.parquet")  # type: ignore[arg-type]
