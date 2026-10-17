"""Typed market-data contracts returned by Deriv."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator


class ActiveSymbol(BaseModel):
    """Symbol metadata returned by the public market-data endpoint."""

    model_config = ConfigDict(extra="ignore")

    underlying_symbol: str
    underlying_symbol_name: str | None = None
    underlying_symbol_type: str | None = None
    market: str | None = None


class TickHistory(BaseModel):
    """Parallel epoch and price arrays returned by ticks_history."""

    model_config = ConfigDict(extra="ignore")

    prices: list[Decimal]
    times: list[int]


class OhlcUpdate(BaseModel):
    """Live OHLC update; ``open_time`` is the candle's open epoch."""

    model_config = ConfigDict(extra="ignore")

    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    granularity: int | None = None
    pip_size: int | None = None


class Tick(BaseModel):
    """A single Deriv price tick."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    epoch: int
    quote: Decimal
    id: str | None = None
    ask: Decimal | None = None
    bid: Decimal | None = None
    pip_size: int | None = None


class Candle(BaseModel):
    """An OHLC candle returned by Deriv."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    epoch: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    granularity: int

    @field_validator("granularity")
    @classmethod
    def validate_granularity(cls, value: int) -> int:
        """Deriv OHLC candles are available only at intervals of 60 seconds or more."""
        if value < 60:
            raise ValueError("Candle granularity must be at least 60 seconds")
        return value
