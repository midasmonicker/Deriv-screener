"""Strict regression probes for the second quant-review round."""

from __future__ import annotations

from decimal import Decimal
from math import sqrt

import numpy as np
import pandas as pd
import pytest

from deriv_vol_lab.backtest import (
    ProductSpec,
    backtest,
    forward_trade_returns,
    walk_forward_backtest,
)
from deriv_vol_lab.api import snapshot as snapshot_module
from deriv_vol_lab.api.snapshot import build_snapshot
from deriv_vol_lab.data import Candle
from deriv_vol_lab.features import screen_candles, wilder_rsi
from deriv_vol_lab.stats.analysis import (
    SECONDS_PER_YEAR,
    _stationary_bootstrap_p_value,
    rolling_volatility_monitor,
)


def _candles(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    start_epoch: int = 1_700_000_000,
    granularity: int = 60,
) -> list[Candle]:
    selected_opens = closes if opens is None else opens
    return [
        Candle(
            symbol="R_75",
            epoch=start_epoch + index * granularity,
            open=Decimal(str(open_price)),
            high=Decimal(str(max(open_price, close) * 1.001)),
            low=Decimal(str(min(open_price, close) * 0.999)),
            close=Decimal(str(close)),
            granularity=granularity,
        )
        for index, (open_price, close) in enumerate(zip(selected_opens, closes, strict=True))
    ]


def test_volatility_monitor_excludes_gapped_returns_and_reports_insufficient_data() -> None:
    epochs = [0, 60, 120, 240, 300, 360]
    log_prices = [0.0, 0.001, 0.003, 1.0, 1.003, 1.007]
    closes = pd.Series(
        np.exp(log_prices),
        index=pd.Index(epochs, name="epoch"),
        dtype=float,
    )

    report = rolling_volatility_monitor(
        {"R_75": closes},
        window=4,
        sampling_intervals={"R_75": 60},
    )

    assert len(report.windows) == 1
    result = report.windows[0]
    assert result.status == "ok"
    assert result.observations == 4
    assert result.realized_volatility == pytest.approx(
        sqrt(float(np.var([0.001, 0.002, 0.003, 0.004], ddof=1))) * sqrt(SECONDS_PER_YEAR / 60)
    )

    insufficient = rolling_volatility_monitor(
        {"R_75": closes.iloc[[0, 1, 2, 3]]},
        window=4,
        sampling_intervals={"R_75": 60},
    )
    assert len(insufficient.windows) == 1
    assert insufficient.windows[0].status == "insufficient_data"
    assert insufficient.windows[0].alert is None
    assert insufficient.windows[0].observations == 2


def test_snapshot_uses_older_valid_returns_when_recent_history_has_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(snapshot_module, "VOL_MONITOR_WINDOW", 3)
    monkeypatch.setattr(snapshot_module, "SCREEN_PERMUTATIONS", 9)
    candles = _candles([100, 101, 102, 10_000, 10_001])
    candles[3:] = [
        candle.model_copy(update={"epoch": candle.epoch + candle.granularity})
        for candle in candles[3:]
    ]

    snapshot = build_snapshot(
        {"R_75": candles},
        generated_at_epoch=candles[-1].epoch,
    )

    result = next(
        window for window in snapshot.volatility_monitor.windows if window.symbol == "R_75"
    )
    assert result.status == "ok"
    assert result.observations == 3
    assert result.window_end == candles[-1].epoch


def test_screener_reliability_and_backtest_share_costed_open_to_close_returns() -> None:
    closes = [100.0 * (1.001**index) for index in range(120)]
    opens = [closes[0], *(close * 1.0002 for close in closes[:-1])]
    candles = _candles(closes, opens=opens)
    product = ProductSpec(
        contract_type="multiplier",
        stake=10,
        multiplier=1,
        stop_out_fraction=1,
        commission_per_trade=0.02,
        spread_fraction=0.0001,
    )

    row = screen_candles(
        {"R_75": candles},
        latest_n=len(candles),
        permutations=9,
        seed=17,
        product=product,
    ).iloc[0]
    calibration = candles[:72]
    calibration_close = pd.Series(
        [float(candle.close) for candle in calibration],
        index=pd.Index([candle.epoch for candle in calibration]),
    )
    flag = (wilder_rsi(calibration_close, period=14) >= 70).astype(float)
    backtest_result = backtest(calibration, flag, product)
    expected_edge = float(
        np.mean([trade.net_pnl / product.stake for trade in backtest_result.trades])
    )

    assert row["overbought_edge_calibration"] == pytest.approx(expected_edge)
    horizon_signal = pd.Series(
        [1.0, 0.0, 0.0],
        index=pd.Index([candle.epoch for candle in candles[:3]]),
    )
    two_candle_result = forward_trade_returns(
        candles[:3],
        horizon_signal,
        product,
        horizon=2,
    )
    assert len(two_candle_result) == 1
    assert two_candle_result[0].entry_epoch == candles[1].epoch
    assert two_candle_result[0].exit_epoch == candles[2].epoch + candles[2].granularity
    assert two_candle_result[0].commission == product.commission_per_trade


def test_walk_forward_reports_worst_fold_drawdown_without_stitching_folds() -> None:
    candles = _candles(
        [100, 100, 100, 200, 100, 100, 100, 50],
        opens=[100] * 8,
    )
    signal_values = [0.0] * len(candles)
    signal_values[2] = 1.0
    signal_values[6] = 1.0
    signal = pd.Series(signal_values, index=pd.Index([candle.epoch for candle in candles]))
    product = ProductSpec(
        contract_type="multiplier",
        stake=100,
        multiplier=1,
        stop_out_fraction=0.5,
    )

    report = walk_forward_backtest(
        candles,
        signal,
        product,
        train_size=2,
        test_size=2,
        step_size=4,
        initial_equity=100,
    )

    assert len(report.windows) == 2
    assert report.out_of_sample.max_drawdown == pytest.approx(0.5)
    assert report.worst_fold_drawdown == pytest.approx(0.5)
    # Evaluated candles are [2, 3, 6, 7], with returns [0, 1, 0, -0.5].
    expected_sharpe = (
        np.mean([0.0, 1.0, 0.0, -0.5])
        / np.std(
            [0.0, 1.0, 0.0, -0.5],
            ddof=1,
        )
        * sqrt(SECONDS_PER_YEAR / 60)
    )
    assert report.out_of_sample.sharpe == pytest.approx(expected_sharpe)


def test_stationary_bootstrap_known_answer_and_iid_limit() -> None:
    class FixedGenerator:
        def __init__(self, indices: list[int]) -> None:
            self._indices = iter(indices)

        def integers(self, high: int) -> int:
            return next(self._indices) % high

        def random(self) -> float:
            return 0.0

    values = np.asarray([1.0, 0.0, 0.0, 0.0])
    generator = FixedGenerator([0, 2, 1, 3, 2, 0, 3, 1])
    observed_p = _stationary_bootstrap_p_value(
        values,
        bootstrap_count=2,
        expected_block_length=2.0,
        generator=generator,  # type: ignore[arg-type]
    )
    # Both deterministic resamples have mean zero; the observed mean is 0.25.
    assert observed_p == pytest.approx(1 / 3)

    values = np.asarray([1.0, 0.0, 0.0])
    iid_generator = np.random.default_rng(21)
    iid_p = _stationary_bootstrap_p_value(
        values,
        bootstrap_count=5,
        expected_block_length=1.0,
        generator=iid_generator,
    )
    hand_rng = np.random.default_rng(21)
    centered = values - values.mean()
    observed = abs(float(values.mean()))
    sampled_statistics = []
    for _ in range(5):
        indices = [int(hand_rng.integers(len(values)))]
        for _ in range(1, len(values)):
            hand_rng.random()
            indices.append(int(hand_rng.integers(len(values))))
        sampled_statistics.append(abs(float(centered[indices].mean())))
    expected_iid_p = (1 + sum(stat >= observed for stat in sampled_statistics)) / 6
    assert iid_p == pytest.approx(expected_iid_p)


def test_screener_reports_floored_partition_candle_counts() -> None:
    candles = _candles([100.0 + index for index in range(17)])

    result = screen_candles(
        {"R_75": candles},
        latest_n=17,
        calibration_fraction=0.6,
        evaluation_fraction=0.1,
        permutations=9,
        seed=2,
    )
    row = result.iloc[0]

    assert row["calibration_candles"] == 10
    assert row["evaluation_candles"] == 1
    assert row["display_candles"] == 6
