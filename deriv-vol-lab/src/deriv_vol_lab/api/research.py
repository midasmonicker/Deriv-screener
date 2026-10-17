"""Packaged, read-only summary of the offline flag-edge research.

The numbers come from ``scripts/flag_edge_report.py`` and ``scripts/payout_breakeven.py`` and
are frozen into ``research_summary.json`` by ``scripts/build_research_summary.py``. Nothing here
recomputes statistics or touches market data.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class ResearchUnavailableError(RuntimeError):
    """Raised when the research summary is missing or malformed."""


class NullControl(BaseModel):
    """Behaviour of the method on seeded driftless random-walk data."""

    model_config = ConfigDict(frozen=True)

    replications: int
    tests_per_replication: int
    raw_significant_fraction: float
    bh_surviving_fraction: float
    confirmed_edges: int


class DataWindow(BaseModel):
    """Stored candle window the research ran on."""

    model_config = ConfigDict(frozen=True)

    symbols: int
    candles_per_symbol: int
    first_epoch: int
    last_epoch: int


class PooledTest(BaseModel):
    """One pooled flag/horizon/direction hypothesis."""

    model_config = ConfigDict(frozen=True)

    flag: str
    horizon: int
    direction: Literal["long", "short"]
    discovery_net_bp: float
    confirmation_net_bp: float
    confirmation_q: float
    stouffer_z: float | None
    effective_n: int


class PayoutCheck(BaseModel):
    """Break-even comparison for one assumed fixed payout."""

    model_config = ConfigDict(frozen=True)

    payout: float
    break_even_win_rate: float
    rows_tested: int
    rows_above_break_even: int
    bh_significant_rows: int
    smallest_bh_q: float
    replicated_pooled_flags: list[str]


class ResearchSummary(BaseModel):
    """Everything the dashboard needs to show the research evidence."""

    model_config = ConfigDict(frozen=True)

    report_date: str
    data_window: DataWindow
    alpha: float
    hypothesis_rows: int
    confirmed_pooled_edges: int
    bonferroni_discoveries: int
    null_control: NullControl
    pooled_tests: list[PooledTest]
    payout_checks: list[PayoutCheck]
    conclusion: str
    caveats: list[str]


def default_summary_path() -> Path | None:
    """Return the override path from the environment, or None for the packaged file."""
    override = os.environ.get("RESEARCH_SUMMARY_PATH")
    return Path(override) if override else None


def load_research_summary(path: Path | None = None) -> ResearchSummary:
    """Load and validate the research summary."""
    try:
        if path is not None:
            text = path.read_text(encoding="utf-8")
        else:
            text = (
                files("deriv_vol_lab.api")
                .joinpath("research_summary.json")
                .read_text(encoding="utf-8")
            )
        return ResearchSummary.model_validate_json(text)
    except (OSError, ValidationError, ValueError) as exc:
        raise ResearchUnavailableError(f"Research summary unavailable: {exc}") from exc
