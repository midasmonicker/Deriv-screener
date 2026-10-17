"""Offline tests for the flag-edge report pipeline."""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stderr
from decimal import Decimal
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scripts.flag_edge_report as edge_report
from scripts.flag_edge_report import (
    EdgeAnalysis,
    NullControlResult,
    analyze_flag_series,
    assert_truncation_invariance,
    binomial_p_value,
    build_forward_trades,
    circular_shift_p_value,
    hit_rate,
    load_stored_candles,
    main,
    make_gbm_candles,
    minimum_detectable_hit_deviation,
    run_null_control,
    stationary_bootstrap_ci,
    write_reports,
)

from deriv_vol_lab.backtest.engine import ProductSpec
from deriv_vol_lab.data import SYMBOLS, Candle, DuckDBStorage
from deriv_vol_lab.features.screener import FLAG_NAMES, causal_flag_series
from deriv_vol_lab.stats.analysis import benjamini_hochberg


def _candles_from_moves(symbol: str, moves: list[float]) -> list[Candle]:
    candles: list[Candle] = []
    open_price = 100.0
    for position, move in enumerate(moves):
        close_price = open_price * exp(move)
        candles.append(
            Candle(
                symbol=symbol,
                epoch=1_700_000_000 + position * 60,
                open=Decimal(str(open_price)),
                high=Decimal(str(max(open_price, close_price) * 1.00001)),
                low=Decimal(str(min(open_price, close_price) * 0.99999)),
                close=Decimal(str(close_price)),
                granularity=60,
            )
        )
        open_price = close_price
    return candles


def _planted_signal_fixture(count: int, seed: int) -> tuple[list[Candle], pd.Series[bool]]:
    generator = np.random.default_rng(seed)
    signals = generator.random(count) < 0.5
    moves = [0.0]
    for position in range(1, count):
        probability_up = 0.56 if signals[position - 1] else 0.44
        is_up = generator.random() < probability_up
        moves.append(0.0005 if is_up else -0.0005)
    candles = _candles_from_moves("R_75", moves)
    signal_series = pd.Series(
        signals,
        index=pd.Index([candle.epoch for candle in candles]),
        dtype=bool,
    )
    return candles, signal_series


def test_hit_rate_and_binomial_p_use_non_tied_outcomes() -> None:
    outcomes = [0.1, 0.0, -0.1, 0.05]

    assert hit_rate(outcomes) == pytest.approx(2 / 3)
    assert binomial_p_value(outcomes) == pytest.approx(0.5)


def test_circular_shift_p_value_matches_hand_computed_fixture() -> None:
    p_value = circular_shift_p_value(
        [True, False, False, False],
        [4.0, 1.0, 2.0, 3.0],
        shifts=3,
        seed=9,
    )

    assert p_value == pytest.approx(0.25)


def test_stationary_bootstrap_ci_is_exact_for_constant_returns() -> None:
    assert stationary_bootstrap_ci([2.5] * 12, seed=7, replications=50) == (2.5, 2.5)


def test_minimum_detectable_hit_deviation_matches_binomial_power_fixture() -> None:
    deviation = minimum_detectable_hit_deviation(100, alpha=0.05, power=0.8)

    assert deviation == pytest.approx(12.5614209725, abs=1e-8)


def test_benjamini_hochberg_matches_hand_computed_fixture() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_screener_flags_are_truncation_invariant_and_leak_is_detected() -> None:
    candles = _candles_from_moves("R_75", [0.001 if i % 3 else -0.001 for i in range(140)])
    flags = causal_flag_series(candles)

    assert set(flags) == set(FLAG_NAMES)
    assert_truncation_invariance(candles, causal_flag_series)

    def leaky_flags(prefix: list[Candle]) -> dict[str, pd.Series[bool]]:
        closes = pd.Series([float(candle.close) for candle in prefix])
        epochs = pd.Index([candle.epoch for candle in prefix])
        return {"leaky_next_close": closes.shift(-1).gt(closes).set_axis(epochs)}

    with pytest.raises(ValueError, match="causal"):
        assert_truncation_invariance(candles, leaky_flags)


def test_planted_edge_replicates_in_confirmation_window(tmp_path: Path) -> None:
    candles, planted = _planted_signal_fixture(7_000, 9281)
    analysis = analyze_flag_series(
        {"R_75": candles},
        {"R_75": {"planted": planted}},
        horizons=(1,),
        alpha=0.05,
        seed=12345,
        permutations=2_000,
        bootstrap_reps=100,
    )

    confirmed = [
        row
        for row in analysis.rows
        if row.scope == "pooled"
        and row.flag == "planted"
        and row.horizon == 1
        and row.direction == "long"
        and row.split == "confirmation"
        and row.confirmed
    ]
    assert len(confirmed) == 1
    assert confirmed[0].net_mean_bps > 0

    product = ProductSpec(
        contract_type="multiplier",
        stake=1.0,
        multiplier=1.0,
        stop_out_fraction=1.0,
    )
    null_result = NullControlResult(
        replications=20,
        tests_per_replication=analysis.test_count,
        raw_significant_fraction=0.05,
        bh_surviving_fraction=0.0,
        confirmed_edges=0,
    )
    csv_path, markdown_path = write_reports(
        analysis,
        null_result,
        {"R_75": candles},
        out_dir=tmp_path,
        product=product,
    )

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert csv_path.name.startswith("flag_edge_")
    assert len(csv_rows) == analysis.test_count
    assert "Descriptive analytics on random processes" in markdown
    assert "## Data Windows" in markdown
    assert "## Top 15 Pooled Tests" in markdown
    assert "## Confirmed Edges" in markdown


def test_null_control_failure_returns_exit_code_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candles = _candles_from_moves("R_75", [0.0001] * 2_000)
    observed_products: list[ProductSpec] = []
    observed_jobs: list[int] = []
    monkeypatch.setattr(edge_report, "load_stored_candles", lambda *_: {"R_75": candles})

    def capture_analysis(*_args: object, **kwargs: object) -> EdgeAnalysis:
        product = kwargs["product"]
        assert isinstance(product, ProductSpec)
        observed_products.append(product)
        return EdgeAnalysis(rows=[], test_count=0, bonferroni_count=0, alpha=0.05)

    monkeypatch.setattr(
        edge_report,
        "analyze_flag_series",
        capture_analysis,
    )

    def fake_null_control(*_args: object, **kwargs: object) -> NullControlResult:
        observed_jobs.append(kwargs["jobs"])
        return NullControlResult(
            replications=20,
            tests_per_replication=20,
            raw_significant_fraction=0.05,
            bh_surviving_fraction=0.11,
            confirmed_edges=0,
        )

    monkeypatch.setattr(edge_report, "run_null_control", fake_null_control)
    monkeypatch.setattr(
        edge_report,
        "write_reports",
        lambda *_args, **_kwargs: (tmp_path / "report.csv", tmp_path / "report.md"),
    )

    exit_code = main(
        [
            "--db",
            str(tmp_path / "unused.duckdb"),
            "--symbols",
            "R_75",
            "--horizons",
            "1",
            "--jobs",
            "1",
            "--stake",
            "2",
            "--multiplier",
            "3",
            "--commission-per-trade",
            "0.01",
            "--spread-fraction",
            "0.002",
            "--stop-out-fraction",
            "0.8",
            "--out-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 3
    assert observed_jobs == [1]
    assert observed_products[0].stake == 2
    assert observed_products[0].multiplier == 3
    assert observed_products[0].commission_per_trade == pytest.approx(0.01)
    assert observed_products[0].spread_fraction == pytest.approx(0.002)
    assert observed_products[0].stop_out_fraction == pytest.approx(0.8)
    assert "Null-control failure" in capsys.readouterr().out


def test_gbm_null_control_has_no_confirmed_edges_across_seeds() -> None:
    template = make_gbm_candles(
        "R_75",
        epochs=[1_700_000_000 + position * 60 for position in range(2_200)],
        stated_volatility=0.75,
        granularity=60,
        seed=10,
    )
    result = run_null_control(
        {"R_75": template},
        replications=20,
        horizons=(1,),
        alpha=0.05,
        seed=12345,
        permutations=2_000,
        bootstrap_reps=50,
    )

    assert result.replications == 20
    assert result.confirmed_edges == 0
    assert result.bh_surviving_fraction <= 0.10


def test_null_control_is_deterministic_across_job_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    template = make_gbm_candles(
        "R_75",
        epochs=[1_700_000_000 + position * 60 for position in range(2_000)],
        stated_volatility=0.75,
        granularity=60,
        seed=10,
    )
    arguments = {
        "replications": 20,
        "horizons": (1,),
        "alpha": 0.05,
        "seed": 12345,
        "permutations": 2_000,
        "bootstrap_reps": 1,
    }

    serial_result = run_null_control({"R_75": template}, jobs=1, **arguments)
    parallel_result = run_null_control({"R_75": template}, jobs=2, **arguments)

    assert serial_result == parallel_result
    progress = capsys.readouterr().err
    assert progress.count(" started\n") == 40
    assert progress.count(" finished in ") == 40
    assert "ETA " in progress


def test_progress_writer_flushes_to_stderr() -> None:
    class FlushTrackingStream(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    stream = FlushTrackingStream()

    with redirect_stderr(stream):
        edge_report._progress("stage started")

    assert stream.getvalue().endswith("stage started\n")
    assert stream.flush_count == 1


def test_jobs_default_is_cpu_count_minus_one() -> None:
    arguments = edge_report._argument_parser().parse_args([])

    assert arguments.jobs == edge_report._default_jobs()


def test_null_eta_accounts_for_parallel_work_already_underway() -> None:
    eta = edge_report._estimate_null_eta_seconds(
        replications=20,
        completed_count=7,
        worker_count=7,
        average_replication_seconds=240.0,
        active_replication_seconds=210.0,
    )

    assert eta == pytest.approx((240.0 * 13 - 210.0) / 7)


def test_quick_mode_skips_null_and_writes_invalid_evidence_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected_symbols: list[str] = []
    captured_horizons: list[tuple[int, ...]] = []
    first_symbol, second_symbol = tuple(SYMBOLS)[:2]
    fixtures = {
        symbol: _candles_from_moves(symbol, [0.0001] * 2_000)
        for symbol in (first_symbol, second_symbol)
    }

    def fake_load(_database_path: Path, symbols: list[str]) -> dict[str, list[Candle]]:
        selected_symbols.extend(symbols)
        return {symbol: fixtures[symbol] for symbol in symbols}

    def fake_analysis(candles_by_symbol: dict[str, list[Candle]], **kwargs: object) -> EdgeAnalysis:
        captured_horizons.extend(kwargs["horizons"])
        return EdgeAnalysis(rows=[], test_count=0, bonferroni_count=0, alpha=0.05)

    monkeypatch.setattr(edge_report, "load_stored_candles", fake_load)
    monkeypatch.setattr(edge_report, "analyze_flag_series", fake_analysis)
    monkeypatch.setattr(
        edge_report,
        "run_null_control",
        lambda *_args, **_kwargs: pytest.fail("quick mode must skip the null control"),
    )

    exit_code = main(
        ["--quick", "--db", str(tmp_path / "offline.duckdb"), "--out-dir", str(tmp_path)]
    )

    quick_reports = list(tmp_path.glob("quick_*.md"))
    assert exit_code == 0
    assert selected_symbols == [first_symbol, second_symbol]
    assert captured_horizons == [1, 5]
    assert len(quick_reports) == 1
    assert "NOT VALID EVIDENCE" in quick_reports[0].read_text(encoding="utf-8")
    progress = capsys.readouterr().err
    assert "load started" in progress and "load finished" in progress
    assert "real-data analysis started" in progress
    assert "real-data analysis finished" in progress
    assert "report writing started" in progress
    assert "report writing finished" in progress


@pytest.mark.parametrize(
    ("option", "value"),
    [("--permutations", "1999"), ("--null-reps", "19")],
)
def test_quick_mode_still_enforces_report_inference_minimums(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--quick", option, value, "--db", str(tmp_path / "offline.duckdb")])

    assert error.value.code == 2


def test_gap_suppresses_a_trade_whose_holding_period_crosses_it() -> None:
    candles = _candles_from_moves("R_75", [0.001] * 12)
    del candles[7]
    signals = pd.Series(
        [position == 6 for position in range(len(candles))],
        index=pd.Index([candle.epoch for candle in candles]),
        dtype=bool,
    )

    trades = build_forward_trades(
        candles,
        signals,
        horizon=3,
        direction="long",
        non_overlapping=True,
    )

    assert trades == []


def test_forward_trades_report_product_costs_in_net_pnl() -> None:
    candles = _candles_from_moves("R_75", [0.001, 0.001, -0.001])
    signals = pd.Series(
        [True, False, False],
        index=pd.Index([candle.epoch for candle in candles]),
        dtype=bool,
    )
    product = ProductSpec(
        contract_type="multiplier",
        stake=1.0,
        multiplier=1.0,
        stop_out_fraction=0.9,
        commission_per_trade=0.003,
        spread_fraction=0.002,
    )

    trades = build_forward_trades(
        candles,
        signals,
        horizon=1,
        direction="long",
        non_overlapping=False,
        product=product,
    )

    assert len(trades) == 1
    assert trades[0].net_pnl == pytest.approx(trades[0].gross_pnl - 0.005)


def test_storage_loader_is_read_only_for_candle_rows(tmp_path) -> None:
    database_path = tmp_path / "readonly.duckdb"
    candles = _candles_from_moves("R_75", [0.0001] * 2_000)
    with DuckDBStorage(database_path) as storage:
        storage.write_candles(candles)
        expected = storage.read_candles(60, "R_75")

    loaded = load_stored_candles(database_path, ["R_75"])

    assert loaded["R_75"] == expected
    with DuckDBStorage(database_path) as storage:
        assert storage.read_candles(60, "R_75") == expected


def test_missing_database_returns_exit_code_two_without_creating_it(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "missing.duckdb"

    exit_code = main(
        [
            "--db",
            str(database_path),
            "--symbols",
            "R_75",
            "--out-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 2
    assert "missing" in capsys.readouterr().out.lower()
    assert not database_path.exists()


def test_too_little_stored_history_returns_exit_code_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candles = _candles_from_moves("R_75", [0.0001] * 1_999)
    monkeypatch.setattr(edge_report, "load_stored_candles", lambda *_: {"R_75": candles})

    exit_code = main(
        [
            "--db",
            str(tmp_path / "unused.duckdb"),
            "--symbols",
            "R_75",
            "--out-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 2
    assert "at least 2000 candles" in capsys.readouterr().out
