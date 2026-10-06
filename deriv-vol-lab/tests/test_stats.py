"""Known-answer tests for statistical monitors and null-hypothesis tools."""

from __future__ import annotations

from math import isclose, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, chi2, norm

from deriv_vol_lab.stats import (
    SECONDS_PER_YEAR,
    arch_lm_test,
    benjamini_hochberg,
    evaluate_signal,
    ljung_box_test,
    randomness_test_battery,
    rolling_volatility_monitor,
    runs_test,
    simulate_driftless_gbm,
    variance_ratio_test,
)


def _series(values: list[float], start: int = 0) -> pd.Series[float]:
    return pd.Series(values, index=pd.Index(range(start, start + len(values)), name="epoch"))


def test_volatility_monitor_known_chi_square_interval_and_alert() -> None:
    """Constant periodic return variance maps to its exact annualized chi-square CI."""
    periodic_return = 0.01
    returns = np.asarray([periodic_return, -periodic_return] * 5)
    close = pd.Series(np.exp(np.concatenate(([0.0], np.cumsum(returns)))))
    report = rolling_volatility_monitor(
        {"1HZ100V": close},
        window=10,
        confidence_level=0.99,
    )

    assert len(report.windows) == 1
    result = report.windows[0]
    sample_variance = float(np.var(returns, ddof=1))
    scale = sqrt(SECONDS_PER_YEAR)
    assert result.realized_volatility == pytest.approx(sqrt(sample_variance) * scale)
    assert result.lower_confidence_bound == pytest.approx(
        sqrt(9 * sample_variance / chi2.ppf(0.995, 9)) * scale
    )
    assert result.upper_confidence_bound == pytest.approx(
        sqrt(9 * sample_variance / chi2.ppf(0.005, 9)) * scale
    )
    assert result.alert is True


def test_volatility_monitor_validation() -> None:
    """Invalid inputs and unsupported symbols fail explicitly."""
    with pytest.raises(ValueError, match="window"):
        rolling_volatility_monitor({}, window=1)
    with pytest.raises(ValueError, match="confidence_level"):
        rolling_volatility_monitor({}, confidence_level=1.0)
    with pytest.raises(ValueError, match="Unsupported"):
        rolling_volatility_monitor({"unknown": _series([1, 2, 3])})
    with pytest.raises(ValueError, match="prices"):
        rolling_volatility_monitor({"R_75": _series([1, 0, 3])})
    with pytest.raises(ValueError, match="sampling_intervals"):
        rolling_volatility_monitor(
            {"R_75": _series([1, 2, 3])},
            window=2,
            sampling_intervals={"R_75": 0},
        )


def test_volatility_monitor_uses_actual_candle_sampling_interval() -> None:
    """Annualization follows 60-second candle sampling rather than native ticks."""
    periodic_returns = np.asarray([0.01, -0.01, 0.02, -0.02])
    close = pd.Series(
        np.exp(np.concatenate(([0.0], np.cumsum(periodic_returns)))),
        index=pd.Index(np.arange(5) * 60, name="epoch"),
    )
    report = rolling_volatility_monitor(
        {"R_75": close},
        window=4,
        sampling_intervals={"R_75": 60},
    )
    expected = sqrt(float(np.var(periodic_returns, ddof=1))) * sqrt(SECONDS_PER_YEAR / 60)

    assert len(report.windows) == 1
    assert report.windows[0].realized_volatility == pytest.approx(expected)


def test_ljung_box_known_answer() -> None:
    """The lag-one Ljung-Box result matches its explicit autocorrelation formula."""
    returns = _series([1, 2, 3, 4, 5, 6])
    statistic, p_value = ljung_box_test(returns, lags=1)
    centered = np.asarray(returns) - np.mean(returns)
    correlation = np.dot(centered[1:], centered[:-1]) / np.dot(centered, centered)
    expected = len(returns) * (len(returns) + 2) * correlation**2 / (len(returns) - 1)

    assert statistic == pytest.approx(expected)
    assert p_value == pytest.approx(chi2.sf(expected, 1))


def test_runs_test_known_alternating_signs() -> None:
    """Four alternating returns yield four runs and the known corrected z score."""
    statistic, p_value = runs_test(_series([1, -1, 1, -1]))

    expected_statistic = 0.5 / sqrt(2 / 3)
    assert statistic == pytest.approx(expected_statistic)
    assert p_value == pytest.approx(2 * norm.sf(expected_statistic))


def test_variance_ratio_known_alternating_returns() -> None:
    """Adjacent alternating returns cancel, yielding a variance ratio of zero."""
    statistic, p_value = variance_ratio_test(_series([1, -1] * 10), horizon=2)

    assert statistic == pytest.approx(-sqrt(20))
    assert p_value == pytest.approx(2 * norm.sf(sqrt(20)))


def test_variance_ratio_known_nonzero_overlapping_estimate() -> None:
    """Finite-sample Lo-MacKinlay normalization matches a hand-computed sample."""
    values = np.asarray([1.0, 2.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    horizon = 2
    sample_size = values.size
    demeaned = values - values.mean()
    variance_one = np.dot(demeaned, demeaned) / (sample_size - 1)
    overlapping = np.convolve(demeaned, np.ones(horizon), mode="valid")
    effective_observations = horizon * (sample_size - horizon + 1) * (1 - horizon / sample_size)
    variance_horizon = np.dot(overlapping, overlapping) / effective_observations
    ratio = variance_horizon / variance_one
    theta = 2 * (2 * horizon - 1) * (horizon - 1) / (3 * horizon * sample_size)
    expected_statistic = (ratio - 1) / sqrt(theta)

    statistic, p_value = variance_ratio_test(_series(values.tolist()), horizon=horizon)

    assert statistic == pytest.approx(expected_statistic)
    assert p_value == pytest.approx(2 * norm.sf(abs(expected_statistic)))


def test_arch_lm_known_constant_variance_series() -> None:
    """Constant squared returns produce no ARCH-LM evidence."""
    statistic, p_value = arch_lm_test(_series([1, -1] * 10), lags=2)

    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_benjamini_hochberg_known_answers() -> None:
    """Adjusted values are monotone in rank and restored to their original order."""
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.002]) == pytest.approx([0.02, 0.04, 0.04, 0.008])
    assert benjamini_hochberg([]) == []
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([1.1])


def test_randomness_battery_corrects_across_symbols_and_tests() -> None:
    """Battery returns all four tests per symbol with a shared BH family."""
    returns = _series([1, -1] * 30)
    results = randomness_test_battery(
        {"R_75": returns, "1HZ75V": returns},
        lags=3,
        variance_ratio_horizon=2,
        arch_lags=2,
    )

    assert len(results) == 8
    assert {item.test for item in results} == {
        "ljung_box",
        "runs",
        "variance_ratio",
        "arch_lm",
    }
    assert {item.symbol for item in results} == {"R_75", "1HZ75V"}
    assert all(0 <= item.adjusted_p_value <= 1 for item in results)
    assert all(item.adjusted_p_value >= item.p_value for item in results)


def test_randomness_test_sample_validation() -> None:
    """Test sample-size and parameter constraints are explicit."""
    with pytest.raises(ValueError, match="more observations than lags"):
        ljung_box_test(_series([1, 2, 3]), lags=3)
    with pytest.raises(ValueError, match="at least two"):
        runs_test(_series([1]))
    with pytest.raises(ValueError, match="more observations than horizon"):
        variance_ratio_test(_series([1, 2]), horizon=2)
    with pytest.raises(ValueError, match="lags \\+ 1"):
        arch_lm_test(_series([1, 2, 3]), lags=2)
    with pytest.raises(ValueError, match="fdr_alpha"):
        randomness_test_battery({}, fdr_alpha=0)


def test_seeded_null_simulator_is_reproducible_and_matches_tick_volatility() -> None:
    """GBM paths have stable seeded output, expected shape, and correct dt scale."""
    first = simulate_driftless_gbm(
        "1HZ75V",
        n_paths=2000,
        n_steps=1000,
        seed=2026,
    )
    second = simulate_driftless_gbm(
        "1HZ75V",
        n_paths=2000,
        n_steps=1000,
        seed=2026,
    )

    assert first.shape == (2000, 1001)
    np.testing.assert_array_equal(first, second)
    assert (first[:, 0] == 100).all()
    log_returns = np.log(first[:, 1:] / first[:, :-1]).ravel()
    measured_annualized = np.std(log_returns, ddof=1) * sqrt(SECONDS_PER_YEAR)
    assert measured_annualized == pytest.approx(0.75, rel=0.05)


def test_null_simulator_validation() -> None:
    """Unknown symbols, missing nonpositive dimensions, and bad start prices fail."""
    with pytest.raises(ValueError, match="Unsupported"):
        simulate_driftless_gbm("unknown", n_paths=1, n_steps=1, seed=1)
    with pytest.raises(ValueError, match="n_paths"):
        simulate_driftless_gbm("R_75", n_paths=0, n_steps=1, seed=1)
    with pytest.raises(ValueError, match="start_price"):
        simulate_driftless_gbm("R_75", n_paths=1, n_steps=1, seed=1, start_price=0)


def test_signal_evaluation_known_forward_statistics_and_reproducibility() -> None:
    """Directional hits, binomial test, and seeded shuffle test are measurable."""
    close = _series([100, 110, 100, 90, 90])
    signal = _series([1, 1, -1, -1, 0])
    first = evaluate_signal(signal, close, horizons=(1,), permutations=999, seed=7)
    second = evaluate_signal(signal, close, horizons=(1,), permutations=999, seed=7)

    result = first.horizons[0]
    assert result.observations == 4
    assert result.mean_forward_return == pytest.approx((0.1 - 10 / 110 - 0.1) / 4)
    assert result.median_forward_return == pytest.approx(-5 / 110)
    assert result.standard_deviation == pytest.approx(np.std([0.1, -10 / 110, -0.1, 0.0], ddof=1))
    assert result.tie_count == 1
    assert result.n_effective == 3
    assert result.hit_rate == pytest.approx(2 / 3)
    assert result.binomial_p_value == pytest.approx(1.0)
    assert result.binomial_is_inferential is True
    assert result.permutation_p_value == second.horizons[0].permutation_p_value
    assert isclose(result.mean_signed_forward_return, (0.1 - 10 / 110 + 0.1) / 4)


def test_signal_evaluation_validation_and_no_lookahead() -> None:
    """Evaluation requires aligned inputs and uses only future outcomes for labels."""
    close = _series([100, 101, 102, 103, 104, 105])
    signal = _series([1, -1, 1, -1, 1, 1])
    with pytest.raises(ValueError, match="identical indexes"):
        evaluate_signal(signal, close.rename(index={5: 99}), seed=3)
    with pytest.raises(ValueError, match="horizons"):
        evaluate_signal(signal, close, horizons=(), seed=3)
    with pytest.raises(ValueError, match="permutations"):
        evaluate_signal(signal, close, permutations=0, seed=3)

    baseline = evaluate_signal(signal, close, horizons=(1,), permutations=199, seed=9)
    altered_close = close.copy()
    altered_close.iloc[-1] *= 3
    changed = evaluate_signal(signal, altered_close, horizons=(1,), permutations=199, seed=9)
    assert baseline.horizons[0].observations == changed.horizons[0].observations
    assert baseline.horizons[0].mean_forward_return != changed.horizons[0].mean_forward_return


def test_overlapping_horizon_inference_uses_non_overlapping_decisions() -> None:
    """Longer horizons expose effective sample size and withhold binomial inference."""
    close = _series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95])
    signal = _series([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0])

    result = evaluate_signal(
        signal,
        close,
        horizons=(3,),
        permutations=199,
        stationary_bootstrap=True,
        bootstrap_count=199,
        seed=23,
    ).horizons[0]
    sampled_signed = signal.to_numpy()[::3] * (
        close.shift(-3).to_numpy()[::3] / close.to_numpy()[::3] - 1
    )
    sampled_signed = sampled_signed[np.isfinite(sampled_signed) & (sampled_signed != 0)]
    expected_binomial = binomtest(
        int(np.count_nonzero(sampled_signed > 0)),
        sampled_signed.size,
        p=0.5,
    ).pvalue

    assert result.n_effective == 3
    assert result.binomial_p_value == pytest.approx(expected_binomial)
    assert result.binomial_is_inferential is False
    assert result.stationary_bootstrap_p_value is not None
    assert 0 < result.stationary_bootstrap_p_value <= 1


def test_circular_shift_permutation_preserves_persistent_signal_order() -> None:
    """Circular-shift null differs from iid shuffling for a persistent signal."""
    close = _series([100, 101, 102, 103, 104, 103, 102, 101, 100])
    signal = _series([1, 1, 1, 1, -1, -1, -1, -1, 0])
    permutations = 499
    seed = 71

    result = evaluate_signal(
        signal,
        close,
        horizons=(1,),
        permutations=permutations,
        seed=seed,
    ).horizons[0]

    forward = (close.shift(-1) / close - 1).to_numpy()
    valid = np.isfinite(forward)
    directions = signal.to_numpy()[valid]
    returns = forward[valid]
    observed = float(np.mean(directions[directions != 0] * returns[directions != 0]))
    circular_rng = np.random.default_rng(seed)
    circular_stats = []
    for _ in range(permutations):
        offset = int(circular_rng.integers(1, len(directions)))
        shifted = np.roll(directions, offset)
        active = shifted != 0
        circular_stats.append(float(np.mean(shifted[active] * returns[active])))
    expected_circular_p = (1 + np.count_nonzero(np.abs(circular_stats) >= abs(observed))) / (
        permutations + 1
    )

    shuffle_rng = np.random.default_rng(seed)
    shuffled_stats = []
    for _ in range(permutations):
        shuffled = shuffle_rng.permutation(directions)
        active = shuffled != 0
        shuffled_stats.append(float(np.mean(shuffled[active] * returns[active])))
    independent_shuffle_p = (1 + np.count_nonzero(np.abs(shuffled_stats) >= abs(observed))) / (
        permutations + 1
    )

    assert result.permutation_p_value == pytest.approx(expected_circular_p)
    assert result.permutation_p_value != independent_shuffle_p


def test_evaluate_signal_stationary_bootstrap_is_seeded_and_validates_options() -> None:
    """Stationary block bootstrap can be reproduced and rejects invalid lengths."""
    close = _series([100, 101, 99, 102, 100, 104, 98, 105, 97, 103, 96])
    signal = _series([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0])
    first = evaluate_signal(
        signal,
        close,
        horizons=(2,),
        permutations=99,
        stationary_bootstrap=True,
        bootstrap_count=199,
        expected_block_length=3,
        seed=4,
    )
    second = evaluate_signal(
        signal,
        close,
        horizons=(2,),
        permutations=99,
        stationary_bootstrap=True,
        bootstrap_count=199,
        expected_block_length=3,
        seed=4,
    )

    assert first.horizons[0].stationary_bootstrap_p_value == (
        second.horizons[0].stationary_bootstrap_p_value
    )
    with pytest.raises(ValueError, match="expected_block_length"):
        evaluate_signal(
            signal,
            close,
            horizons=(2,),
            stationary_bootstrap=True,
            expected_block_length=0.5,
            seed=4,
        )
