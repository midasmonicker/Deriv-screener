"""DuckDB market-data storage, gap detection, and Parquet export."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import duckdb

from deriv_vol_lab.data.models import Candle, Tick
from deriv_vol_lab.data.quality import DataQualityReport, MarketRecord, build_quality_report
from deriv_vol_lab.data.symbols import SYMBOLS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deriv_vol_lab.data.client import DerivWebSocketClient


class DuckDBStorage:
    """Synchronous DuckDB storage for ticks and multiple candle intervals."""

    def __init__(self, database_path: Path | str = "data/deriv-vol-lab.duckdb") -> None:
        self.database_path = Path(database_path)
        if str(database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(database_path))
        self._closed = False
        self._create_ticks_table()

    def close(self) -> None:
        """Close the database connection."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> DuckDBStorage:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write_ticks(self, ticks: Sequence[Tick]) -> int:
        """Idempotently insert or update ticks keyed by (symbol, epoch)."""
        self._ensure_open()
        if not ticks:
            return 0
        connection = self._connection.cursor()
        connection.begin()
        try:
            connection.executemany(
                """
                INSERT INTO ticks (symbol, epoch, quote, tick_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (symbol, epoch) DO UPDATE SET
                    quote = excluded.quote,
                    tick_id = excluded.tick_id
                """,
                [(tick.symbol, tick.epoch, tick.quote, tick.id) for tick in ticks],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return len(ticks)

    def write_candles(self, candles: Sequence[Candle]) -> int:
        """Idempotently insert or update candles in their granularity table."""
        self._ensure_open()
        if not candles:
            return 0
        granularities = {candle.granularity for candle in candles}
        if len(granularities) != 1:
            raise ValueError("All candles in one write must share a granularity")
        granularity = granularities.pop()
        table = self._candle_table(granularity)
        self._create_candles_table(granularity)
        connection = self._connection.cursor()
        connection.begin()
        try:
            connection.executemany(
                f"""
                INSERT INTO {table}
                    (symbol, epoch, open, high, low, close, granularity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, epoch) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close
                """,
                [
                    (
                        candle.symbol,
                        candle.epoch,
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.granularity,
                    )
                    for candle in candles
                ],
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return len(candles)

    def read_ticks(
        self,
        symbol: str | None = None,
        *,
        start_epoch: int | None = None,
        end_epoch: int | None = None,
    ) -> list[Tick]:
        """Read ticks in chronological order, optionally constrained by range."""
        self._ensure_open()
        conditions, parameters = self._range_conditions(symbol, start_epoch, end_epoch)
        rows = self._connection.execute(
            f"SELECT symbol, epoch, quote, tick_id FROM ticks{conditions} ORDER BY symbol, epoch",
            parameters,
        ).fetchall()
        return [Tick(symbol=row[0], epoch=row[1], quote=row[2], id=row[3]) for row in rows]

    def read_candles(
        self,
        granularity: int,
        symbol: str | None = None,
        *,
        start_epoch: int | None = None,
        end_epoch: int | None = None,
    ) -> list[Candle]:
        """Read candles in chronological order for one granularity."""
        self._ensure_open()
        table = self._candle_table(granularity)
        conditions, parameters = self._range_conditions(symbol, start_epoch, end_epoch)
        rows = self._connection.execute(
            f"""
            SELECT symbol, epoch, open, high, low, close, granularity
            FROM {table}{conditions}
            ORDER BY symbol, epoch
            """,
            parameters,
        ).fetchall()
        return [
            Candle(
                symbol=row[0],
                epoch=row[1],
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                granularity=row[6],
            )
            for row in rows
        ]

    def has_candle_table(self, granularity: int) -> bool:
        """Return whether the database contains a candle table for this cadence."""
        self._ensure_open()
        table = self._candle_table(granularity)
        result = self._connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema = 'main'",
            [table],
        ).fetchone()
        return result is not None and result[0] > 0

    def find_gaps(
        self,
        symbol: str,
        start_epoch: int,
        end_epoch: int,
        *,
        granularity: int | None = None,
    ) -> list[int]:
        """Find expected cadence-aligned timestamps missing from storage."""
        self._ensure_open()
        if start_epoch > end_epoch:
            raise ValueError("start_epoch must be less than or equal to end_epoch")
        if granularity is None:
            definition = SYMBOLS.get(symbol)
            if definition is None:
                raise ValueError(f"No expected tick interval registered for {symbol}")
            interval = definition.tick_interval_sec
            table = "ticks"
        else:
            interval = granularity
            table = self._candle_table(granularity)
            self._create_candles_table(granularity)

        first_expected = ceil(start_epoch / interval) * interval
        if first_expected > end_epoch:
            return []
        rows = self._connection.execute(
            f"""
            SELECT epoch FROM {table}
            WHERE symbol = ? AND epoch BETWEEN ? AND ?
            """,
            [symbol, first_expected, end_epoch],
        ).fetchall()
        observed = {int(row[0]) for row in rows}
        return [
            epoch
            for epoch in range(first_expected, end_epoch + 1, interval)
            if epoch not in observed
        ]

    async def backfill_gaps(
        self,
        client: DerivWebSocketClient,
        symbol: str,
        start_epoch: int,
        end_epoch: int,
        *,
        granularity: int | None = None,
    ) -> list[int]:
        """Fetch a missing history range, persist it, and return remaining gaps."""
        missing = self.find_gaps(symbol, start_epoch, end_epoch, granularity=granularity)
        if not missing:
            return []
        days = max((end_epoch - start_epoch + 1) / 86_400, 1 / 86_400)
        if granularity is None:
            records: list[Tick] = await client.backfill_ticks(
                symbol,
                days,
                end_epoch=end_epoch,
            )
            self.write_ticks(records)
        else:
            records_candles: list[Candle] = await client.backfill_candles(
                symbol,
                days,
                granularity=granularity,
                end_epoch=end_epoch,
            )
            self.write_candles(records_candles)
        return self.find_gaps(symbol, start_epoch, end_epoch, granularity=granularity)

    def quality_report(
        self,
        records: Sequence[MarketRecord] | None = None,
        *,
        symbol: str | None = None,
        granularity: int | None = None,
        start_epoch: int | None = None,
        end_epoch: int | None = None,
        sigma_threshold: float = 8.0,
        sigma_window: int = 100,
    ) -> DataQualityReport:
        """Build a report from raw observations or records stored in DuckDB."""
        self._ensure_open()
        if records is None:
            if granularity is None:
                records_to_check: Sequence[MarketRecord] = self.read_ticks(
                    symbol,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                )
            else:
                records_to_check = self.read_candles(
                    granularity,
                    symbol,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                )
        else:
            records_to_check = [
                record for record in records if symbol is None or record.symbol == symbol
            ]
        return build_quality_report(
            records_to_check,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            sigma_threshold=sigma_threshold,
            sigma_window=sigma_window,
        )

    def export_parquet(self, table: Literal["ticks"] | int, path: Path | str) -> Path:
        """Export ticks or one candle-granularity table as a Parquet file."""
        self._ensure_open()
        if table == "ticks":
            table_name = "ticks"
        elif isinstance(table, int) and not isinstance(table, bool):
            table_name = self._candle_table(table)
            self._create_candles_table(table)
        else:
            raise ValueError("table must be 'ticks' or a positive candle granularity")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        escaped_path = str(target.resolve()).replace("'", "''")
        self._connection.execute(
            f"COPY {table_name} TO '{escaped_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        return target

    def _create_ticks_table(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                symbol VARCHAR NOT NULL,
                epoch BIGINT NOT NULL,
                quote DECIMAL(38, 18) NOT NULL,
                tick_id VARCHAR,
                PRIMARY KEY (symbol, epoch)
            )
            """
        )

    def _create_candles_table(self, granularity: int) -> None:
        table = self._candle_table(granularity)
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                symbol VARCHAR NOT NULL,
                epoch BIGINT NOT NULL,
                open DECIMAL(38, 18) NOT NULL,
                high DECIMAL(38, 18) NOT NULL,
                low DECIMAL(38, 18) NOT NULL,
                close DECIMAL(38, 18) NOT NULL,
                granularity INTEGER NOT NULL CHECK (granularity = {granularity}),
                PRIMARY KEY (symbol, epoch)
            )
            """
        )

    @staticmethod
    def _candle_table(granularity: int) -> str:
        if granularity <= 0:
            raise ValueError("granularity must be greater than zero")
        return f"candles_{granularity}"

    @staticmethod
    def _range_conditions(
        symbol: str | None,
        start_epoch: int | None,
        end_epoch: int | None,
    ) -> tuple[str, list[str | int]]:
        conditions: list[str] = []
        parameters: list[str | int] = []
        if symbol is not None:
            conditions.append("symbol = ?")
            parameters.append(symbol)
        if start_epoch is not None:
            conditions.append("epoch >= ?")
            parameters.append(start_epoch)
        if end_epoch is not None:
            conditions.append("epoch <= ?")
            parameters.append(end_epoch)
        return (
            f" WHERE {' AND '.join(conditions)}" if conditions else "",
            parameters,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DuckDB storage is closed")
