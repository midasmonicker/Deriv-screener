"""Typed market-data contracts returned by Deriv."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator


class Tick(BaseModel):
    """A single Deriv price tick."""

    model_config = ConfigDict(extra="ignore")

    symbol: str
    epoch: int
    quote: Decimal
    id: str | None = None


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
