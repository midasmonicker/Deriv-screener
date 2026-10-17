"""Freeze a flag-edge report into the packaged dashboard research summary (offline).

Usage:
    python scripts/build_research_summary.py reports/flag_edge_YYYYMMDD.csv \
        [--payouts 0.90 0.95 1.00] [--out src/deriv_vol_lab/api/research_summary.json]

Reads the report's CSV and Markdown (same stem) and writes one JSON file. The payouts are
assumptions, not broker quotes. Nothing is recomputed beyond the break-even comparison.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

import payout_breakeven as pb

from deriv_vol_lab.api.research import (
    DataWindow,
    NullControl,
    PayoutCheck,
    PooledTest,
    ResearchSummary,
)

TOP_POOLED = 15
CAVEATS = [
    "Descriptive analytics on random processes; no predictive edge is implied.",
    "Only one stretch of history (about 60 days) of one synthetic process was tested.",
    "Rare flags have too few independent trades to detect any realistic edge.",
    "Costs, spreads and slippage were set to zero, which favours the flags.",
    "Payouts in the break-even check are assumptions, not broker quotes.",
    "Not detecting an edge is not proof that none exists.",
]


def _percent(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)%", text)
    if match is None:
        raise ValueError(f"Missing '{label}' in report Markdown")
    return float(match.group(1)) / 100.0


def _integer(text: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}:\s*([0-9]+)", text)
    if match is None:
        raise ValueError(f"Missing '{label}' in report Markdown")
    return int(match.group(1))


def parse_markdown(text: str) -> tuple[str, DataWindow, float, int, int, NullControl]:
    """Extract date, data window, alpha, test counts and null control from the report."""
    date_match = re.search(r"UTC report date:\s*([0-9-]+)", text)
    alpha_match = re.search(r"Alpha:\s*([0-9.]+)", text)
    rows_match = re.search(r"rows per p-value family:\s*([0-9]+)", text)
    bonferroni_match = re.search(r"Bonferroni discoveries:\s*([0-9]+)", text)
    if not (date_match and alpha_match and rows_match and bonferroni_match):
        raise ValueError("Report Markdown header is not in the expected format")
    windows = re.findall(r"^\|\s*(\S+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", text, re.M)
    if not windows:
        raise ValueError("Report Markdown has no data-window table")
    window = DataWindow(
        symbols=len(windows),
        candles_per_symbol=min(int(w[1]) for w in windows),
        first_epoch=max(int(w[2]) for w in windows),
        last_epoch=min(int(w[3]) for w in windows),
    )
    null_control = NullControl(
        replications=_integer(text, "Seeded driftless GBM replications"),
        tests_per_replication=_integer(text, "Tests per replication"),
        raw_significant_fraction=_percent(text, "Raw circular-shift p < alpha"),
        bh_surviving_fraction=_percent(text, "BH survivors / all null tests"),
        confirmed_edges=_integer(text, "Confirmed null edges"),
    )
    return (
        date_match.group(1),
        window,
        float(alpha_match.group(1)),
        int(rows_match.group(1)),
        int(bonferroni_match.group(1)),
        null_control,
    )


def _optional_float(value: str) -> float | None:
    return float(value) if value not in ("", "None") else None


def pooled_tests(rows: Sequence[pb.Row]) -> list[PooledTest]:
    """Pair each pooled hypothesis' discovery and confirmation rows; rank by confirmation q."""
    pooled = [r for r in rows if r["scope"] == "pooled"]
    discovery = {
        (r["flag"], r["horizon"], r["direction"]): r for r in pooled if r["split"] == "discovery"
    }
    tests: list[PooledTest] = []
    for r in pooled:
        if r["split"] != "confirmation":
            continue
        key = (r["flag"], r["horizon"], r["direction"])
        d = discovery.get(key)
        if d is None:
            continue
        direction: Literal["long", "short"]
        if r["direction"] == "long":
            direction = "long"
        elif r["direction"] == "short":
            direction = "short"
        else:
            raise ValueError(f"Unexpected direction {r['direction']!r}")
        tests.append(
            PooledTest(
                flag=r["flag"],
                horizon=int(r["horizon"]),
                direction=direction,
                discovery_net_bp=float(d["net_mean_bps"]),
                confirmation_net_bp=float(r["net_mean_bps"]),
                confirmation_q=float(r["adjusted_p"]),
                stouffer_z=_optional_float(r["stouffer_z"]),
                effective_n=int(r["n_effective"]),
            )
        )
    tests.sort(key=lambda t: (t.confirmation_q, t.flag, t.horizon, t.direction))
    return tests[:TOP_POOLED]


def payout_checks(rows: Sequence[pb.Row], payouts: Sequence[float]) -> list[PayoutCheck]:
    """Summarise the break-even comparison for each assumed payout."""
    usable = pb.usable_rows(rows)
    checks: list[PayoutCheck] = []
    for payout in payouts:
        results = pb.evaluate_payout(usable, payout)
        breakeven = pb.break_even_win_rate(payout)
        checks.append(
            PayoutCheck(
                payout=payout,
                break_even_win_rate=breakeven,
                rows_tested=len(results),
                rows_above_break_even=sum(1 for item in results if item.win_rate > breakeven),
                bh_significant_rows=sum(item.q_value < pb.ALPHA for item in results),
                smallest_bh_q=min(item.q_value for item in results),
                replicated_pooled_flags=["/".join(k) for k in pb.replicated_pooled_flags(results)],
            )
        )
    return checks


def conclusion_text(
    confirmed: int, rows: int, null_control: NullControl, checks: Sequence[PayoutCheck]
) -> str:
    """State what the evidence does and does not show, derived from the numbers."""
    null_note = (
        f"The method was checked on {null_control.replications} simulated random datasets: "
        f"{null_control.raw_significant_fraction:.2%} of raw tests passed p < 0.05 and "
        f"{null_control.bh_surviving_fraction:.2%} survived multiple-testing correction."
    )
    if confirmed == 0:
        head = (
            f"No screener flag showed a replicated, statistically significant edge across "
            f"{rows} tests."
        )
    else:
        head = (
            f"{confirmed} pooled flag(s) met the confirmation rule across {rows} tests; treat "
            f"them as unverified until reproduced on a different period."
        )
    significant = sum(check.bh_significant_rows for check in checks)
    payout_note = (
        "None beat the break-even win rate of a fixed-payout contract after correction."
        if significant == 0
        else f"{significant} row(s) beat a fixed-payout break-even after correction; verify them."
    )
    return f"{head} {null_note} {payout_note}"


def build_summary(csv_path: Path, markdown_path: Path, payouts: Sequence[float]) -> ResearchSummary:
    """Combine the report CSV and Markdown into one validated summary."""
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    date, window, alpha, hypothesis_rows, bonferroni, null_control = parse_markdown(
        markdown_path.read_text(encoding="utf-8")
    )
    confirmed = sum(
        1
        for r in rows
        if r["scope"] == "pooled" and r["split"] == "confirmation" and r["confirmed"] == "True"
    )
    checks = payout_checks(rows, payouts)
    return ResearchSummary(
        report_date=date,
        data_window=window,
        alpha=alpha,
        hypothesis_rows=hypothesis_rows,
        confirmed_pooled_edges=confirmed,
        bonferroni_discoveries=bonferroni,
        null_control=null_control,
        pooled_tests=pooled_tests(rows),
        payout_checks=checks,
        conclusion=conclusion_text(confirmed, hypothesis_rows, null_control, checks),
        caveats=list(CAVEATS),
    )


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--payouts", nargs="+", type=float, default=[0.90, 0.95, 1.00])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "src"
        / "deriv_vol_lab"
        / "api"
        / "research_summary.json",
    )
    args = parser.parse_args()
    summary = build_summary(args.csv_path, args.csv_path.with_suffix(".md"), args.payouts)
    args.out.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
