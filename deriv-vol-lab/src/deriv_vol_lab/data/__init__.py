"""Market data access package."""

from deriv_vol_lab.data.client import (
    DEFAULT_ENDPOINT,
    MAX_HISTORY_COUNT,
    ClientClosedError,
    ConnectionLostError,
    DerivAPIError,
    DerivWebSocketClient,
    HandshakeConnectionError,
    SubscriptionClosed,
    SubscriptionHandle,
)
from deriv_vol_lab.data.models import ActiveSymbol, Candle, OhlcUpdate, Tick, TickHistory
from deriv_vol_lab.data.quality import (
    AbnormalJump,
    DailyCoverage,
    DataQualityReport,
    DuplicateEpoch,
    InvalidPrice,
    TimestampOrderIssue,
    build_quality_report,
)
from deriv_vol_lab.data.storage import DuckDBStorage
from deriv_vol_lab.data.symbols import SYMBOLS, SymbolDefinition

__all__ = [
    "DEFAULT_ENDPOINT",
    "MAX_HISTORY_COUNT",
    "SYMBOLS",
    "AbnormalJump",
    "ActiveSymbol",
    "Candle",
    "ClientClosedError",
    "ConnectionLostError",
    "DailyCoverage",
    "DataQualityReport",
    "DerivAPIError",
    "DerivWebSocketClient",
    "DuckDBStorage",
    "DuplicateEpoch",
    "HandshakeConnectionError",
    "InvalidPrice",
    "OhlcUpdate",
    "SubscriptionClosed",
    "SubscriptionHandle",
    "SymbolDefinition",
    "Tick",
    "TickHistory",
    "TimestampOrderIssue",
    "build_quality_report",
]
