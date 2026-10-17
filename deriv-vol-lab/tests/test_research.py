"""Tests for the packaged research summary, its builder, and the payout break-even maths."""

from __future__ import annotations

import csv
from pathlib import Path

import httpx
import pytest
from scipy.stats import binomtest
from scripts import build_research_summary as brs
from scripts import payout_breakeven as pb

from deriv_vol_lab.api.app import app
from deriv_vol_lab.api.research import (
    ResearchSummary,
    ResearchUnavailableError,
    load_research_summary,
)

FIELDS = [
    "scope",
    "symbol",
    "flag",
    "horizon",
    "direction",
    "split",
    "n_signals",
    "n_effective",
    "n_non_tied",
    "gross_mean_bps",
    "net_mean_bps",
    "hit_rate",
    "binomial_p",
    "circular_shift_p",
    "stouffer_z",
    "adjusted_p",
    "confirmed",
]

MARKDOWN = """# Flag Edge Report

- UTC report date: 2026-01-02
- ProductSpec: multiplier=1.0.
- Alpha: 0.05; total hypothesis rows per p-value family: 8; Bonferroni threshold: 6e-03; \
Bonferroni discoveries: 0.

## Data Windows

| Symbol | Candles | First candle epoch | Last candle epoch |
|---|---:|---:|---:|
| R_10 | 1000 | 1000 | 60940 |
| R_25 | 990 | 1060 | 60900 |

## Null Control

- Seeded driftless GBM replications: 20.
- Tests per replication: 8.
- Raw circular-shift p < alpha: 5.000% (expected approximately alpha).
- BH survivors / all null tests: 0.000% (expected near zero).
- Confirmed null edges: 0.
"""


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "scope": "pooled",
        "symbol": "ALL",
        "flag": "stretched",
        "horizon": "5",
        "direction": "long",
        "split": "discovery",
        "n_signals": "500",
        "n_effective": "500",
        "n_non_tied": "500",
        "gross_mean_bps": "0.5",
        "net_mean_bps": "0.5",
        "hit_rate": "0.5",
        "binomial_p": "0.5",
        "circular_shift_p": "0.5",
        "stouffer_z": "0.1",
        "adjusted_p": "0.9",
        "confirmed": "False",
    }
    base.update(overrides)
    return base


def _write_report(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv_path = tmp_path / "flag_edge_20260102.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".md").write_text(MARKDOWN, encoding="utf-8")
    return csv_path


def _report_rows() -> list[dict[str, str]]:
    return [
        _row(split="discovery", net_mean_bps="-0.2", hit_rate="0.52", adjusted_p="0.9"),
        _row(split="confirmation", net_mean_bps="0.7", hit_rate="0.53", adjusted_p="0.4"),
        _row(flag="overbought", split="discovery", net_mean_bps="0.1", hit_rate="0.5"),
        _row(flag="overbought", split="confirmation", net_mean_bps="0.2", hit_rate="0.5"),
        _row(scope="symbol", symbol="R_10", split="confirmation", n_non_tied="0", hit_rate="0"),
    ]


def test_break_even_win_rate_matches_closed_form() -> None:
    assert pb.break_even_win_rate(0.95) == pytest.approx(1 / 1.95)
    assert pb.break_even_win_rate(1.0) == pytest.approx(0.5)


def test_benjamini_hochberg_known_answer() -> None:
    adjusted = pb.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert adjusted == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.20])


def test_evaluate_payout_matches_exact_binomial_and_expected_value() -> None:
    rows = [_row(n_non_tied="100", hit_rate="0.60")]
    (result,) = pb.evaluate_payout(rows, 1.0)

    assert result.n == 100
    assert result.win_rate == pytest.approx(0.60)
    assert result.expected_value == pytest.approx(0.60 * 1.0 - 0.40)
    assert result.p_value == pytest.approx(binomtest(60, 100, 0.5, alternative="greater").pvalue)
    assert result.q_value == pytest.approx(result.p_value)  # one test: BH changes nothing


def test_payout_reduces_expected_value_below_even_money() -> None:
    rows = [_row(n_non_tied="1000", hit_rate="0.51")]
    (even,) = pb.evaluate_payout(rows, 1.0)
    (house,) = pb.evaluate_payout(rows, 0.9)

    assert even.expected_value > 0 > house.expected_value
    assert house.p_value > even.p_value


def test_usable_rows_drops_rows_without_non_tied_trades() -> None:
    rows = [_row(n_non_tied="0"), _row(n_non_tied="3")]
    assert [r["n_non_tied"] for r in pb.usable_rows(rows)] == ["3"]


def test_replication_requires_enough_trades_in_both_splits() -> None:
    enough = [
        _row(split="discovery", n_non_tied="150", hit_rate="0.6"),
        _row(split="confirmation", n_non_tied="150", hit_rate="0.6"),
    ]
    thin = [
        _row(split="discovery", n_non_tied="150", hit_rate="0.6"),
        _row(split="confirmation", n_non_tied="20", hit_rate="0.9"),
    ]
    assert pb.replicated_pooled_flags(pb.evaluate_payout(enough, 1.0)) == [
        ("stretched", "5", "long")
    ]
    assert pb.replicated_pooled_flags(pb.evaluate_payout(thin, 1.0)) == []


def test_markdown_report_states_assumptions_and_sorts_deterministically() -> None:
    text = pb.render_markdown("x.csv", _report_rows(), [0.95])
    assert "Payouts are assumptions" in text
    assert "break-even win rate 51.28%" in text
    assert "4 of 5 rows have at least one non-tied trade" in text


def test_parse_markdown_extracts_header_window_and_null_control() -> None:
    date, window, alpha, rows, bonferroni, null = brs.parse_markdown(MARKDOWN)

    assert (date, alpha, rows, bonferroni) == ("2026-01-02", 0.05, 8, 0)
    assert window.symbols == 2
    assert window.candles_per_symbol == 990
    assert (window.first_epoch, window.last_epoch) == (1060, 60900)
    assert null.replications == 20
    assert null.raw_significant_fraction == pytest.approx(0.05)
    assert null.bh_surviving_fraction == 0.0


def test_parse_markdown_rejects_unexpected_format() -> None:
    with pytest.raises(ValueError, match="header"):
        brs.parse_markdown("# nothing useful")


def test_build_summary_pairs_splits_and_derives_conclusion(tmp_path: Path) -> None:
    csv_path = _write_report(tmp_path, _report_rows())

    summary = brs.build_summary(csv_path, csv_path.with_suffix(".md"), [0.95, 1.0])

    assert summary.confirmed_pooled_edges == 0
    assert [t.flag for t in summary.pooled_tests] == ["stretched", "overbought"]  # by q
    best = summary.pooled_tests[0]
    assert (best.discovery_net_bp, best.confirmation_net_bp) == (-0.2, 0.7)
    assert best.confirmation_q == pytest.approx(0.4)
    assert summary.conclusion.startswith("No screener flag showed")
    assert {c.payout for c in summary.payout_checks} == {0.95, 1.0}
    assert all(c.rows_tested == 4 for c in summary.payout_checks)  # zero-trade row excluded


def test_build_summary_does_not_claim_no_edge_when_one_is_confirmed(tmp_path: Path) -> None:
    rows = _report_rows()
    rows[1] = _row(split="confirmation", adjusted_p="0.01", confirmed="True")
    csv_path = _write_report(tmp_path, rows)

    summary = brs.build_summary(csv_path, csv_path.with_suffix(".md"), [0.95])

    assert summary.confirmed_pooled_edges == 1
    assert not summary.conclusion.startswith("No screener flag")
    assert "unverified" in summary.conclusion


def test_packaged_summary_loads_and_is_internally_consistent() -> None:
    summary = load_research_summary()

    assert summary.hypothesis_rows > 0
    assert summary.null_control.replications >= 20
    assert summary.null_control.bh_surviving_fraction <= 2 * summary.alpha
    assert summary.pooled_tests == sorted(summary.pooled_tests, key=lambda t: t.confirmation_q)
    assert (summary.confirmed_pooled_edges == 0) == summary.conclusion.startswith("No screener")
    assert any("no predictive edge" in c.lower() for c in summary.caveats)


def test_missing_summary_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ResearchUnavailableError):
        load_research_summary(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchUnavailableError):
        load_research_summary(bad)


async def test_research_endpoint_serves_packaged_summary() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/research/flag-edge")

    assert response.status_code == 200
    assert ResearchSummary.model_validate(response.json()) == load_research_summary()


async def test_research_endpoint_returns_503_when_summary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RESEARCH_SUMMARY_PATH", str(tmp_path / "absent.json"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/research/flag-edge")

    assert response.status_code == 503


async def test_dashboard_keeps_banner_and_shows_evidence_section() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert "No predictive edge is implied" in response.text
    assert "/research/flag-edge" in response.text
    assert "Limits of this evidence" in response.text
