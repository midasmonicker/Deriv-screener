"""Typed data-quality report contracts and analysis helpers."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from math import ceil, isfinite, log
from statistics import pstdev
from typing import Literal

from pydantic import BaseModel, ConfigDict

from deriv_vol_lab.data.models import Candle, Tick
from deriv_vol_lab.data.symbols import SYMBOLS

MarketRecord = Tick | Candle
RecordKind = Literal["tick", "candle"]
SeriesKey = tuple[str, RecordKind, int]


class DuplicateEpoch(BaseModel):
    """An epoch observed more than once for a symbol."""

    symbol: str
    record_kind: RecordKind
    granularity: int
    epoch: int
    occurrences: int


class TimestampOrderIssue(BaseModel):
    """A record whose timestamp is earlier than the previous observation."""

    symbol: str
    record_kind: RecordKind
    granularity: int
    epoch: int
    previous_epoch: int


class InvalidPrice(BaseModel):
    """A tick or candle containing a non-positive price."""

    symbol: str
    record_kind: RecordKind
    granularity: int
    epoch: int
    field: str
    price: float


class AbnormalJump(BaseModel):
    """A local log-return exceeding the configured sigma threshold."""

    symbol: str
    record_kind: RecordKind
    granularity: int
    epoch: int
    log_return: float
    local_sigma: float
    sigma_multiple: float | None
    flag: Literal["sigma-threshold", "zero-variance-break"]


class DailyCoverage(BaseModel):
    """Observed data coverage for one UTC day and a single market-data series."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    record_kind: RecordKind
    granularity: int
    day: date
    expected_records: int
    observed_records: int
    coverage_pct: float


class DataQualityReport(BaseModel):
    """Summary of input data-quality defects and time coverage."""

    duplicate_epochs: list[DuplicateEpoch]
    non_monotonic_timestamps: list[TimestampOrderIssue]
    invalid_prices: list[InvalidPrice]
    abnormal_jumps: list[AbnormalJump]
    daily_coverage: list[DailyCoverage]


def build_quality_report(
    records: Sequence[MarketRecord],
    *,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    sigma_threshold: float = 8.0,
    sigma_window: int = 100,
) -> DataQualityReport:
    """Analyze raw observations while retaining their original order."""
    if sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be greater than zero")
    if sigma_window < 2:
        raise ValueError("sigma_window must be at least 2")
    if start_epoch is not None and end_epoch is not None and start_epoch > end_epoch:
        raise ValueError("start_epoch must be less than or equal to end_epoch")

    epochs_by_series: dict[SeriesKey, list[int]] = defaultdict(list)
    invalid_prices: list[InvalidPrice] = []
    prices_by_series: dict[SeriesKey, list[tuple[int, float]]] = defaultdict(list)
    duplicate_epochs: list[DuplicateEpoch] = []
    order_issues: list[TimestampOrderIssue] = []
    counts: Counter[tuple[SeriesKey, int]] = Counter()
    previous_epoch: dict[SeriesKey, int] = {}

    for record in records:
        if isinstance(record, Tick):
            definition = SYMBOLS.get(record.symbol)
            if definition is None:
                raise ValueError(f"No expected tick interval registered for {record.symbol}")
            record_kind: RecordKind = "tick"
            granularity = definition.tick_interval_sec
            price_fields: tuple[tuple[str, float], ...] = (("quote", float(record.quote)),)
        else:
            record_kind = "candle"
            granularity = record.granularity
            price_fields = (
                ("open", float(record.open)),
                ("high", float(record.high)),
                ("low", float(record.low)),
                ("close", float(record.close)),
            )
        series_key: SeriesKey = (record.symbol, record_kind, granularity)
        epochs_by_series[series_key].append(record.epoch)
        key = (series_key, record.epoch)
        counts[key] += 1
        prior = previous_epoch.get(series_key)
        if prior is not None and record.epoch < prior:
            order_issues.append(
                TimestampOrderIssue(
                    symbol=record.symbol,
                    record_kind=record_kind,
                    granularity=granularity,
                    epoch=record.epoch,
                    previous_epoch=prior,
                )
            )
        previous_epoch[series_key] = record.epoch

        for field, price in price_fields:
            if not isfinite(price) or price <= 0:
                invalid_prices.append(
                    InvalidPrice(
                        symbol=record.symbol,
                        record_kind=record_kind,
                        granularity=granularity,
                        epoch=record.epoch,
                        field=field,
                        price=price,
                    )
                )
        close = float(record.quote if isinstance(record, Tick) else record.close)
        if isfinite(close) and close > 0:
            prices_by_series[series_key].append((record.epoch, close))

    duplicate_epochs = [
        DuplicateEpoch(
            symbol=series_key[0],
            record_kind=series_key[1],
            granularity=series_key[2],
            epoch=epoch,
            occurrences=occurrences,
        )
        for (series_key, epoch), occurrences in sorted(counts.items())
        if occurrences > 1
    ]

    abnormal_jumps: list[AbnormalJump] = []
    for series_key, series_prices in prices_by_series.items():
        prior_returns: list[float] = []
        previous_positive_price: tuple[int, float] | None = None
        for epoch, price in series_prices:
            if previous_positive_price is not None:
                previous_epoch_value, previous_price = previous_positive_price
                if epoch > previous_epoch_value:
                    current_return = log(price / previous_price)
                    if len(prior_returns) >= 2:
                        local_sigma = pstdev(prior_returns[-sigma_window:])
                        zero_variance_break = local_sigma == 0 and current_return != 0
                        exceeds_threshold = (
                            local_sigma > 0 and abs(current_return) > sigma_threshold * local_sigma
                        )
                        if zero_variance_break or exceeds_threshold:
                            abnormal_jumps.append(
                                AbnormalJump(
                                    symbol=series_key[0],
                                    record_kind=series_key[1],
                                    granularity=series_key[2],
                                    epoch=epoch,
                                    log_return=current_return,
                                    local_sigma=local_sigma,
                                    sigma_multiple=(
                                        abs(current_return) / local_sigma
                                        if local_sigma > 0
                                        else None
                                    ),
                                    flag=(
                                        "zero-variance-break"
                                        if zero_variance_break
                                        else "sigma-threshold"
                                    ),
                                )
                            )
                    prior_returns.append(current_return)
            previous_positive_price = (epoch, price)

    daily_coverage = _daily_coverage(
        epochs_by_series,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )
    return DataQualityReport(
        duplicate_epochs=duplicate_epochs,
        non_monotonic_timestamps=order_issues,
        invalid_prices=invalid_prices,
        abnormal_jumps=abnormal_jumps,
        daily_coverage=daily_coverage,
    )


def _daily_coverage(
    epochs_by_series: dict[SeriesKey, list[int]],
    *,
    start_epoch: int | None,
    end_epoch: int | None,
) -> list[DailyCoverage]:
    result: list[DailyCoverage] = []
    for series_key, epochs in sorted(epochs_by_series.items()):
        symbol, record_kind, interval = series_key
        first = min(epochs) if start_epoch is None else start_epoch
        last = max(epochs) if end_epoch is None else end_epoch
        if first > last:
            continue
        first_day = datetime.fromtimestamp(first, UTC).date()
        last_day = datetime.fromtimestamp(last, UTC).date()
        current_day = first_day
        observed = set(epochs)

        while current_day <= last_day:
            day_start = int(
                datetime.combine(current_day, datetime.min.time(), tzinfo=UTC).timestamp()
            )
            day_end = day_start + 86_399
            lower = max(first, day_start)
            upper = min(last, day_end)
            first_expected = ceil(lower / interval) * interval
            last_expected = (upper // interval) * interval
            expected = (
                (last_expected - first_expected) // interval + 1
                if first_expected <= last_expected
                else 0
            )
            observed_count = sum(
                1 for epoch in observed if lower <= epoch <= upper and epoch % interval == 0
            )
            result.append(
                DailyCoverage(
                    symbol=symbol,
                    record_kind=record_kind,
                    granularity=interval,
                    day=current_day,
                    expected_records=expected,
                    observed_records=observed_count,
                    coverage_pct=100.0 * observed_count / expected if expected else 100.0,
                )
            )
            current_day += timedelta(days=1)
    return result
