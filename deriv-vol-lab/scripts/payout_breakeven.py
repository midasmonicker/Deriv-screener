"""Fixed-payout rise/fall break-even check on a stored flag-edge CSV (offline, read-only).

For a fixed-payout contract paying `payout` per unit stake on a win and losing the stake
otherwise, the break-even win rate is 1 / (1 + payout). For each hypothesis row of the
flag-edge report this tests, one-sided and exactly, whether the observed win rate exceeds
that break-even, then applies Benjamini-Hochberg across all rows for each payout.
Payouts are ASSUMPTIONS to be checked against the broker's live quotes; they are inputs.

Usage: python scripts/payout_breakeven.py reports/flag_edge_YYYYMMDD.csv [--payouts 0.90 0.95]
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import binomtest

ALPHA = 0.05
MIN_TABLE_N = 30
MIN_REPLICATION_N = 100

Row = Mapping[str, str]


@dataclass(frozen=True)
class RowResult:
    """Break-even test of one hypothesis row at one payout."""

    row: Row
    n: int
    win_rate: float
    expected_value: float
    p_value: float
    q_value: float


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-adjusted p-values in input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * n / rank)
        adjusted[i] = running
    return adjusted


def usable_rows(rows: Sequence[Row]) -> list[Row]:
    """Keep rows that have at least one non-tied trade."""
    return [r for r in rows if int(r["n_non_tied"]) > 0]


def break_even_win_rate(payout: float) -> float:
    """Win rate at which a fixed-payout contract has zero expected value."""
    return 1.0 / (1.0 + payout)


def evaluate_payout(rows: Sequence[Row], payout: float) -> list[RowResult]:
    """Test every usable row against the break-even win rate; BH across all of them."""
    breakeven = break_even_win_rate(payout)
    interim: list[tuple[Row, int, float, float, float]] = []
    for r in rows:
        n = int(r["n_non_tied"])
        wins = round(float(r["hit_rate"]) * n)
        p_value = float(binomtest(wins, n, breakeven, alternative="greater").pvalue)
        rate = wins / n
        interim.append((r, n, rate, rate * payout - (1.0 - rate), p_value))
    adjusted = benjamini_hochberg([item[4] for item in interim])
    return [
        RowResult(r, n, rate, ev, p, q)
        for (r, n, rate, ev, p), q in zip(interim, adjusted, strict=True)
    ]


def replicated_pooled_flags(results: Sequence[RowResult]) -> list[tuple[str, str, str]]:
    """Pooled flags with positive EV in both splits, each split having enough trades."""
    by_key: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for item in results:
        if item.row["scope"] != "pooled":
            continue
        key = (item.row["flag"], item.row["horizon"], item.row["direction"])
        by_key.setdefault(key, {})[item.row["split"]] = (
            item.expected_value if item.n >= MIN_REPLICATION_N else None
        )
    return sorted(
        key
        for key, splits in by_key.items()
        if len(splits) == 2 and all(v is not None and v > 0 for v in splits.values())
    )


def render_markdown(csv_name: str, rows: Sequence[Row], payouts: Sequence[float]) -> str:
    """Render the human-readable break-even report."""
    usable = usable_rows(rows)
    lines = [
        "# Fixed-payout break-even check",
        "",
        f"Source: `{csv_name}` ({len(usable)} of {len(rows)} rows have at least one "
        f"non-tied trade; the BH family is those {len(usable)} rows). Descriptive analytics "
        "on random processes; no predictive edge is implied. Payouts are assumptions, not "
        "broker quotes.",
        "A win is a trade with a positive return; ties lose. Costs beyond the payout are ignored.",
        "",
    ]
    for payout in payouts:
        results = evaluate_payout(usable, payout)
        breakeven = break_even_win_rate(payout)
        above = sum(1 for item in results if item.win_rate > breakeven)
        survivors = sum(item.q_value < ALPHA for item in results)
        small = sum(1 for item in results if item.n < MIN_TABLE_N)
        replicated = replicated_pooled_flags(results)
        ranked = sorted(results, key=lambda item: (item.p_value, -item.expected_value))
        lines += [
            f"## Payout {payout:.0%}: break-even win rate {breakeven:.2%}",
            "",
            f"- Rows with win rate above break-even: {above} of {len(results)} "
            f"({above / len(results):.1%}).",
            f"- Rows significant after BH (q < {ALPHA}): {survivors}.",
            f"- Rows with fewer than {MIN_TABLE_N} trades ({small}) are left out of the table "
            "below.",
            f"- Smallest BH q-value: {min(item.q_value for item in results):.3f}.",
            "",
            "| Scope | Symbol | Flag | Horizon | Dir | Split | n | Win rate | EV/stake | BH q |",
            "|---|---|---|---:|---|---|---:|---:|---:|---:|",
        ]
        for item in [t for t in ranked if t.n >= MIN_TABLE_N][:8]:
            r = item.row
            lines.append(
                f"| {r['scope']} | {r['symbol']} | {r['flag']} | {r['horizon']} | "
                f"{r['direction']} | {r['split']} | {item.n} | {item.win_rate:.2%} | "
                f"{item.expected_value:+.4f} | {item.q_value:.3f} |"
            )
        names = ", ".join("/".join(key) for key in replicated)
        lines += [
            "",
            f"Pooled flags (n >= {MIN_REPLICATION_N} in both splits) with positive EV in BOTH "
            f"discovery and confirmation: {len(replicated)}"
            + (f" ({names})" if names else ".")
            + "",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--payouts", nargs="+", type=float, default=[0.90, 0.95, 1.00])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if any(not 0.0 < p <= 1.5 for p in args.payouts):
        parser.error("payouts must be fractions such as 0.95")

    with args.csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = args.out or args.csv_path.with_name("payout_breakeven_" + args.csv_path.stem[-8:] + ".md")
    out.write_text(render_markdown(args.csv_path.name, rows, args.payouts), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
