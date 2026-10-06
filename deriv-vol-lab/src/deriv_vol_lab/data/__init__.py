"""Market data access package."""

from deriv_vol_lab.data.client import (
    DEFAULT_ENDPOINT,
    MAX_HISTORY_COUNT,
    ClientClosedError,
    ConnectionLostError,
    DerivAPIError,
    DerivWebSocketClient,
    SubscriptionClosed,
    SubscriptionHandle,
)
from deriv_vol_lab.data.models import Candle, Tick
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
    "Candle",
    "ClientClosedError",
    "ConnectionLostError",
    "DailyCoverage",
    "DataQualityReport",
    "DerivAPIError",
    "DerivWebSocketClient",
    "DuckDBStorage",
    "DuplicateEpoch",
    "InvalidPrice",
    "SubscriptionClosed",
    "SubscriptionHandle",
    "SymbolDefinition",
    "Tick",
    "TimestampOrderIssue",
    "build_quality_report",
]
