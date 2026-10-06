"""Static registry of Deriv Volatility Indices."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SymbolDefinition(BaseModel):
    """Metadata used to identify a supported volatility index."""

    model_config = ConfigDict(frozen=True)

    code: str
    family: Literal["R", "1HZ"]
    stated_vol_pct: float
    tick_interval_sec: int


_SYMBOL_ROWS: tuple[tuple[str, Literal["R", "1HZ"], float, int], ...] = (
    ("R_10", "R", 10, 2),
    ("R_25", "R", 25, 2),
    ("R_50", "R", 50, 2),
    ("R_75", "R", 75, 2),
    ("R_100", "R", 100, 2),
    ("1HZ10V", "1HZ", 10, 1),
    ("1HZ25V", "1HZ", 25, 1),
    ("1HZ50V", "1HZ", 50, 1),
    ("1HZ75V", "1HZ", 75, 1),
    ("1HZ100V", "1HZ", 100, 1),
)

SYMBOLS: dict[str, SymbolDefinition] = {
    code: SymbolDefinition(
        code=code,
        family=family,
        stated_vol_pct=volatility,
        tick_interval_sec=interval,
    )
    for code, family, volatility, interval in _SYMBOL_ROWS
}

VOLATILITY_SYMBOL_CODES = frozenset(SYMBOLS)
