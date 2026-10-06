"""Application entry point."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

import structlog

from deriv_vol_lab.api.ingestion import ingest_latest_snapshot
from deriv_vol_lab.data import Candle, DuckDBStorage
from deriv_vol_lab.features import screen_candles
from deriv_vol_lab.settings import Settings


def main(argv: Sequence[str] | None = None) -> None:
    """Initialize the application and dispatch its command."""
    settings = Settings()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level],
        ),
    )
    parser = argparse.ArgumentParser(prog="deriv-vol-lab")
    commands = parser.add_subparsers(dest="command")
    screen_parser = commands.add_parser("screen", help="screen latest stored candles")
    screen_parser.add_argument("--granularity", type=_positive_int, required=True)
    screen_parser.add_argument("--latest", type=_positive_int, default=500)
    screen_parser.add_argument("--permutations", type=_positive_int, default=499)
    screen_parser.add_argument("--seed", type=int, default=0)
    ingest_parser = commands.add_parser(
        "ingest-snapshot",
        help="fetch recent public candles and write the read-only API snapshot",
    )
    ingest_parser.add_argument("--days", type=_positive_float, default=2.0)
    ingest_parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)

    if arguments.command == "screen":
        with DuckDBStorage(settings.db_path) as storage:
            if not storage.has_candle_table(arguments.granularity):
                structlog.get_logger().error(
                    "screen_candle_table_missing",
                    granularity=arguments.granularity,
                    database=str(settings.db_path),
                )
                raise SystemExit(1)
            candles = storage.read_candles(arguments.granularity)
        candles_by_symbol: dict[str, list[Candle]] = {}
        for candle in candles:
            candles_by_symbol.setdefault(candle.symbol, []).append(candle)
        frame = screen_candles(
            candles_by_symbol,
            latest_n=arguments.latest,
            permutations=arguments.permutations,
            seed=arguments.seed,
        )
        sys.stdout.write(
            "Reliability edges use the calibration window only; the later holdout "
            "and current-display windows are excluded. Unadjusted p-values are "
            "exploratory; adjusted p-values use Benjamini-Hochberg across available "
            "flags and symbols.\n"
        )
        sys.stdout.write(f"{frame.to_string(index=False)}\n")
        return

    if arguments.command == "ingest-snapshot":
        candle_count = asyncio.run(ingest_latest_snapshot(days=arguments.days, seed=arguments.seed))
        sys.stdout.write(
            f"Wrote snapshot with {candle_count} candles to the configured snapshot path.\n"
        )
        return

    structlog.get_logger().info("deriv_vol_lab.ready")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed
