"""Known-answer and causality tests for vectorized features."""

from __future__ import annotations

from math import log, sqrt

import numpy as np
import pandas as pd
import pytest

from deriv_vol_lab.features import (
    SECONDS_PER_YEAR,
    annualize_volatility,
    atr,
    bollinger_zscore,
    ema,
    hurst_rs,
    hurst_variance_ratio,
    parkinson_volatility,
    realized_volatility,
    return_kurtosis,
    return_skew,
    rolling_autocorrelation,
    rolling_variance_ratio,
    wilder_rsi,
)


def _series(values: list[float]) -> pd.Series[float]:
    return pd.Series(values, index=pd.Index(range(10, 10 + len(values)), name="epoch"))


def test_wilder_rsi_known_answers_and_warmup() -> None:
    """A monotonic rising series has RSI 100 after the requested warm-up."""
    close = _series([10, 11, 12, 13, 14])
    result = wilder_rsi(close, period=2)

    assert result.index.equals(close.index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == [100.0, 100.0, 100.0]
    assert wilder_rsi(_series([5, 5, 5, 5]), period=2).iloc[-1] == 50
    mixed = wilder_rsi(_series([10, 11, 10, 12, 11]), period=3)
    assert mixed.iloc[3] == pytest.approx(75.0)
    assert mixed.iloc[4] == pytest.approx(54.5454545455)


def test_ema_known_answers() -> None:
    """EMA uses span alpha and remains aligned to a non-default index."""
    values = _series([1, 2, 3, 4])
    result = ema(values, period=2)

    assert result.index.equals(values.index)
    assert result.iloc[0] != result.iloc[0]
    assert result.iloc[1] == pytest.approx(5 / 3)
    assert result.iloc[2] == pytest.approx(23 / 9)
    assert result.iloc[3] == pytest.approx(95 / 27)


def test_atr_known_answers() -> None:
    """Constant true range produces the same Wilder ATR after warm-up."""
    close = _series([1, 2, 3, 4])
    high = _series([2, 3, 4, 5])
    low = _series([0, 1, 2, 3])

    result = atr(high, low, close, period=2)

    assert result.index.equals(close.index)
    assert result.iloc[1:].tolist() == [2.0, 2.0, 2.0]


def test_bollinger_zscore_known_answer() -> None:
    """The final values in each [x, x+1, x+2] window have known z-scores."""
    values = _series([1, 2, 3, 4])
    result = bollinger_zscore(values, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(sqrt(3 / 2))
    assert result.iloc[3] == pytest.approx(sqrt(3 / 2))


def test_annualization_uses_365_day_seconds_and_preserves_index() -> None:
    """Annualization uses exactly the requested UTC seconds-per-year constant."""
    values = _series([0.1, 0.2])
    assert SECONDS_PER_YEAR == 365 * 24 * 3600
    assert annualize_volatility(values, 60).tolist() == pytest.approx(
        [0.1 * sqrt(SECONDS_PER_YEAR / 60), 0.2 * sqrt(SECONDS_PER_YEAR / 60)]
    )
    with pytest.raises(ValueError, match="granularity_seconds"):
        annualize_volatility(values, 0)


def test_close_to_close_realized_volatility_known_answer() -> None:
    """The final two log returns [1, 2] have sample standard deviation sqrt(0.5)."""
    close = _series([1, np.e, np.e**2, np.e**4])
    result = realized_volatility(close, window=2, granularity_seconds=SECONDS_PER_YEAR)

    assert result.index.equals(close.index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(0.0)
    assert result.iloc[3] == pytest.approx(sqrt(0.5))


def test_parkinson_volatility_known_answer() -> None:
    """Fixed high/low ratio gives a known annualized Parkinson volatility."""
    high = _series([2, 2, 2])
    low = _series([1, 1, 1])
    result = parkinson_volatility(
        high,
        low,
        window=2,
        granularity_seconds=SECONDS_PER_YEAR,
    )
    expected = log(2) / sqrt(4 * log(2))

    assert result.iloc[0] != result.iloc[0]
    assert result.iloc[1:].tolist() == pytest.approx([expected, expected])


@pytest.mark.parametrize("lag", range(1, 11))
def test_rolling_autocorrelation_linear_input_is_one(lag: int) -> None:
    """Each supported lag of a linear sequence is perfectly autocorrelated."""
    values = _series(list(np.arange(1, 31, dtype=float)))
    result = rolling_autocorrelation(values, window=15, lag=lag)

    assert result.index.equals(values.index)
    assert result.iloc[:14].isna().all()
    assert result.iloc[-1] == pytest.approx(1.0)


def test_hurst_rescaled_range_known_linear_sequence() -> None:
    """R/S for a deterministic linear ramp estimates a persistent Hurst value."""
    values = _series(list(np.arange(1, 129, dtype=float)))
    result = hurst_rs(values, window=64)

    assert result.index.equals(values.index)
    assert result.iloc[:63].isna().all()
    assert result.iloc[-1] == pytest.approx(1.0, abs=0.15)


def test_hurst_variance_ratio_known_linear_sequence() -> None:
    """Overlapping variance-ratio estimate matches its known finite-window value."""
    values = _series(list(np.arange(1, 21, dtype=float)))
    result = hurst_variance_ratio(values, window=10, horizon=2)

    assert result.index.equals(values.index)
    assert result.iloc[:9].isna().all()
    assert result.iloc[-1] == pytest.approx(0.8552466914025076)


def test_rolling_variance_ratio_known_alternating_sequence() -> None:
    """Overlapping two-step alternating returns have variance ratio zero."""
    values = _series([1, -1] * 5)
    result = rolling_variance_ratio(values, window=6, horizon=2)

    assert result.index.equals(values.index)
    assert result.iloc[:5].isna().all()
    assert result.iloc[5:].tolist() == pytest.approx([0.0] * 5)


def test_return_skew_and_excess_kurtosis_known_answers() -> None:
    """Symmetric four-point sample has zero skew and 1.5 Fisher excess kurtosis."""
    returns = _series([-1, 0, 0, 1, -1, 0, 0, 1])
    skew = return_skew(returns, window=4)
    kurtosis = return_kurtosis(returns, window=4)

    assert skew.index.equals(returns.index)
    assert kurtosis.index.equals(returns.index)
    assert skew.iloc[3] == pytest.approx(0.0)
    assert kurtosis.iloc[3] == pytest.approx(1.5)


def test_multi_input_indicators_reject_misaligned_indexes() -> None:
    """OHLC features require aligned timestamps rather than implicit joins."""
    close = _series([1, 2, 3])
    misaligned = _series([1, 2, 3]).rename(index={12: 99})

    with pytest.raises(ValueError, match="identical indexes"):
        atr(close, misaligned, close, period=2)
    with pytest.raises(ValueError, match="identical indexes"):
        parkinson_volatility(close, misaligned, window=2)


def test_features_have_no_lookahead_bias() -> None:
    """Changing observations after t cannot change any feature at or before t."""
    index = pd.Index(range(100, 160), name="epoch")
    close = pd.Series(np.exp(np.linspace(0, 0.5, len(index))), index=index)
    high = close * 1.02
    low = close * 0.98
    returns = np.log(close).diff().fillna(0)
    cutoff = 135

    def calculate(
        close_values: pd.Series[float],
        high_values: pd.Series[float],
        low_values: pd.Series[float],
        return_values: pd.Series[float],
    ) -> list[pd.Series[float]]:
        return [
            wilder_rsi(close_values, period=5),
            ema(close_values, period=5),
            atr(high_values, low_values, close_values, period=5),
            bollinger_zscore(close_values, window=5),
            realized_volatility(close_values, window=5, granularity_seconds=1),
            parkinson_volatility(
                high_values,
                low_values,
                window=5,
                granularity_seconds=1,
            ),
            rolling_autocorrelation(return_values, window=15, lag=3),
            hurst_rs(return_values, window=32),
            hurst_variance_ratio(return_values, window=20, horizon=2),
            rolling_variance_ratio(return_values, window=20, horizon=2),
            return_skew(return_values, window=5),
            return_kurtosis(return_values, window=5),
        ]

    baseline = calculate(close, high, low, returns)
    altered_close = close.copy()
    altered_close.loc[index > cutoff] *= 10
    altered_high = altered_close * 1.02
    altered_low = altered_close * 0.98
    altered_returns = np.log(altered_close).diff().fillna(0)
    altered = calculate(altered_close, altered_high, altered_low, altered_returns)

    for original_feature, changed_feature in zip(baseline, altered, strict=True):
        pd.testing.assert_series_equal(
            original_feature.loc[index <= cutoff],
            changed_feature.loc[index <= cutoff],
        )


def test_parameter_validation() -> None:
    """Invalid windows, lag, and non-positive price ranges fail explicitly."""
    values = _series([1, 2, 3, 4])
    with pytest.raises(ValueError, match="period"):
        ema(values, period=0)
    with pytest.raises(ValueError, match="window"):
        bollinger_zscore(values, window=0)
    with pytest.raises(ValueError, match="lag"):
        rolling_autocorrelation(values, window=3, lag=11)
    with pytest.raises(ValueError, match="window"):
        rolling_autocorrelation(values, window=1, lag=1)
    with pytest.raises(ValueError, match="window"):
        hurst_rs(values, window=8)
    with pytest.raises(ValueError, match="horizon"):
        hurst_variance_ratio(values, window=4, horizon=4)
    with pytest.raises(ValueError, match="horizon"):
        rolling_variance_ratio(values, window=4, horizon=4)
    with pytest.raises(ValueError, match="high and low"):
        parkinson_volatility(values, _series([1, 0, 1, 1]), window=2)
    with pytest.raises(ValueError, match="window"):
        return_skew(values, window=2)
    with pytest.raises(ValueError, match="window"):
        return_kurtosis(values, window=3)
