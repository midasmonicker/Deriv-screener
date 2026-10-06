"""Tests for the descriptive cross-symbol screener and CLI."""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deriv_vol_lab.data import Candle, DuckDBStorage
from deriv_vol_lab.features import SCREEN_COLUMNS, screen_candles
from deriv_vol_lab.features.screener import (
    SECONDS_PER_YEAR,
    _realized_volatility_actual_time,
)
from deriv_vol_lab.main import main
from deriv_vol_lab.stats.analysis import benjamini_hochberg


def _candles(symbol: str, count: int, *, granularity: int = 60) -> list[Candle]:
    generator = np.random.default_rng(284)
    increments = generator.normal(0, 0.00015, count)
    log_prices = np.log(100.0) + np.cumsum(increments)
    prices = np.exp(log_prices)
    prices[1:80] = np.linspace(prices[0], prices[0] * 1.15, min(79, count - 1))
    if count > 80:
        prices[80:] = prices[79] * np.exp(np.cumsum(generator.normal(0, 0.0001, count - 80)))
    candles: list[Candle] = []
    for position, price in enumerate(prices):
        close = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                epoch=(position + 1) * granularity,
                open=close,
                high=close * Decimal("1.001"),
                low=close * Decimal("0.999"),
                close=close,
                granularity=granularity,
            )
        )
    return candles


def test_screener_returns_latest_rows_with_features_and_per_flag_reliability() -> None:
    """Output columns align with the contract and binary-flag p-values are measured."""
    candles = _candles("R_75", 180)
    result = screen_candles(
        {"R_75": candles},
        latest_n=180,
        permutations=199,
        seed=31,
    )

    assert tuple(result.columns) == SCREEN_COLUMNS
    assert len(result) == 1
    row = result.iloc[0]
    assert row["symbol"] == "R_75"
    assert row["epoch"] == candles[-1].epoch
    assert math.isfinite(float(row["price"]))
    assert math.isfinite(float(row["rsi"]))
    assert math.isfinite(float(row["bollinger_z"]))
    assert math.isfinite(float(row["ema20"]))
    assert math.isfinite(float(row["ema50"]))
    assert math.isfinite(float(row["realised_vol"]))
    assert row["stated_vol"] == pytest.approx(0.75)
    assert row["vol_ratio"] == pytest.approx(row["realised_vol"] / row["stated_vol"])
    assert isinstance(row["flags"], tuple)
    assert all("buy" not in flag.lower() and "sell" not in flag.lower() for flag in row["flags"])
    assert row["overbought_observations"] >= 5
    assert math.isfinite(float(row["overbought_edge_calibration"]))
    assert 0 <= row["overbought_p_value_unadjusted_exploratory"] <= 1
    assert 0 <= row["overbought_p_value_adjusted"] <= 1


def test_screener_latest_n_limits_history_and_keeps_short_history_descriptive() -> None:
    """Feature calculation respects the N-row cap and marks unavailable warmups."""
    candles = _candles("1HZ75V", 55)
    result = screen_candles(
        {"1HZ75V": candles},
        latest_n=40,
        permutations=9,
        seed=4,
    )

    assert result.iloc[0]["epoch"] == candles[-1].epoch
    assert result.iloc[0]["ema50"] != result.iloc[0]["ema50"]
    assert result.iloc[0]["ema20_50_trend"] == "insufficient_data"
    assert (
        result.iloc[0]["trending_regime_p_value_unadjusted_exploratory"]
        != result.iloc[0]["trending_regime_p_value_unadjusted_exploratory"]
    )


def test_screener_rejects_invalid_inputs() -> None:
    """Screening bounds and unsupported symbols are explicit errors."""
    with pytest.raises(ValueError, match="latest_n"):
        screen_candles({}, latest_n=0, seed=1)
    with pytest.raises(ValueError, match="permutations"):
        screen_candles({}, permutations=0, seed=1)
    with pytest.raises(ValueError, match="Unsupported"):
        screen_candles({"unknown": _candles("unknown", 2)}, seed=1)
    with pytest.raises(ValueError, match="different symbol"):
        screen_candles({"R_75": _candles("R_50", 2)}, seed=1)


def test_screen_cli_reads_duckdb_and_writes_dataframe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The documented screen CLI reports a row from stored candle data."""
    database_path = tmp_path / "screen.duckdb"
    monkeypatch.setenv("DB_PATH", str(database_path))
    with DuckDBStorage(database_path) as storage:
        storage.write_candles(_candles("R_75", 130))

    main(["screen", "--granularity", "60", "--latest", "120", "--permutations", "19"])

    output = capsys.readouterr().out
    assert "R_75" in output
    assert "overbought_p_value_unadjusted_exploratory" in output
    assert "exploratory" in output.lower()
    assert "benjamini-hochberg" in output.lower()
    assert "buy" not in output.lower()
    assert "sell" not in output.lower()


def test_screen_cli_fails_explicitly_without_candle_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing granularity table is reported as a non-zero command failure."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "empty.duckdb"))

    with pytest.raises(SystemExit) as error:
        main(["screen", "--granularity", "60"])

    assert error.value.code == 1


def test_screener_segments_gaps_before_rolling_features() -> None:
    """A short final segment cannot borrow warmup candles across a missing epoch."""
    candles = _candles("R_75", 100)
    candles[50:] = [
        candle.model_copy(update={"epoch": candle.epoch + candle.granularity})
        for candle in candles[50:]
    ]

    result = screen_candles(
        {"R_75": candles},
        latest_n=100,
        permutations=9,
        seed=3,
    )

    assert result.iloc[0]["ema50"] != result.iloc[0]["ema50"]


def test_screener_keeps_calibration_holdout_and_display_disjoint() -> None:
    """Changes confined to the holdout cannot affect calibration or display output."""
    candles = _candles("R_75", 200)
    changed = list(candles)
    for index in range(120, 140):
        candle = changed[index]
        changed[index] = candle.model_copy(
            update={
                "open": candle.open * 2,
                "high": candle.high * 2,
                "low": candle.low * 2,
                "close": candle.close * 2,
            }
        )

    original_result = screen_candles(
        {"R_75": candles},
        latest_n=200,
        permutations=19,
        seed=8,
    ).iloc[0]
    changed_result = screen_candles(
        {"R_75": changed},
        latest_n=200,
        permutations=19,
        seed=8,
    ).iloc[0]

    assert original_result["epoch"] == changed_result["epoch"]
    assert original_result["price"] == changed_result["price"]
    assert original_result["rsi"] == pytest.approx(changed_result["rsi"])
    assert original_result["overbought_edge_calibration"] == pytest.approx(
        changed_result["overbought_edge_calibration"]
    )
    assert original_result["overbought_p_value_unadjusted_exploratory"] == pytest.approx(
        changed_result["overbought_p_value_unadjusted_exploratory"]
    )


def test_screener_bh_adjustment_covers_available_flag_symbol_tests() -> None:
    """Adjusted p-values correct the full available cross-symbol flag family."""
    result = screen_candles(
        {"R_75": _candles("R_75", 200), "1HZ75V": _candles("1HZ75V", 200)},
        latest_n=200,
        permutations=19,
        seed=13,
    )
    p_columns = [
        f"{flag}_p_value_unadjusted_exploratory"
        for flag in (
            "overbought",
            "stretched",
            "vol_mismatch",
            "trending_regime",
            "mean_reverting_regime",
        )
    ]
    values = [
        float(value)
        for value in result[p_columns].to_numpy().ravel()
        if math.isfinite(float(value))
    ]
    expected = benjamini_hochberg(values)
    adjusted_values = [
        float(result.iloc[row_index][column.removesuffix("_unadjusted_exploratory") + "_adjusted"])
        for row_index in result.index
        for column in p_columns
        if math.isfinite(float(result.iloc[row_index][column]))
    ]

    assert adjusted_values == pytest.approx(expected)


def test_realized_volatility_annualizes_by_actual_elapsed_seconds() -> None:
    """Annualized variance uses observed elapsed time, not a fixed candle interval."""
    epochs = pd.Index([0, 60, 180], name="epoch")
    close = pd.Series([100.0, 101.0, 103.0], index=epochs)
    log_return = math.log(103.0 / 101.0)

    result = _realized_volatility_actual_time(close, epochs, window=1)

    assert result.loc[180] == pytest.approx(math.sqrt(log_return**2 * SECONDS_PER_YEAR / 120))
