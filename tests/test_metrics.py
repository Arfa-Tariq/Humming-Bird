import numpy as np
import pytest

from dsp.metrics import (
    cents_error,
    octave_error_mask,
    gross_error_mask,
    rmse_cents,
    gross_error_rate,
    octave_error_rate,
    voiced_accepted_rate,
    silence_falsely_accepted,
    summarize,
    latency_stats,
)


def test_cents_880_vs_440_is_1200():
    assert cents_error(880.0, 440.0)[0] == pytest.approx(1200.0)


def test_cents_equal_is_zero_and_semitone_is_100():
    assert cents_error(440.0, 440.0)[0] == pytest.approx(0.0)
    assert cents_error(440.0 * 2 ** (1 / 12), 440.0)[0] == pytest.approx(100.0)


def test_cents_missing_estimate_is_nan():
    e = cents_error([None, 440.0, np.nan, -5.0], 440.0)
    assert np.isnan(e[0]) and np.isnan(e[2]) and np.isnan(e[3])
    assert e[1] == pytest.approx(0.0)


def test_rmse_ignores_nan():
    assert rmse_cents([3.0, -4.0, np.nan]) == pytest.approx(np.sqrt(12.5))


def test_rmse_all_nan_returns_nan():
    assert np.isnan(rmse_cents([np.nan, np.nan]))


def test_gross_error_rate():
    # 1 of 4 accepted frames is beyond 50 cents; NaN is not counted
    assert gross_error_rate([10, -20, 80, 0, np.nan]) == pytest.approx(25.0)


def test_octave_errors_are_gross_errors():
    e = cents_error([880.0, 220.0, 440.0], 440.0)
    assert octave_error_mask(e).tolist() == [True, True, False]
    assert gross_error_mask(e).tolist() == [True, True, False]
    assert octave_error_rate(e) == pytest.approx(200 / 3)


def test_voicing_rates():
    accepted = np.array([True, True, False, True])
    voiced = np.array([True, True, True, False])
    assert voiced_accepted_rate(accepted, voiced) == pytest.approx(2 / 3)
    assert silence_falsely_accepted(accepted, voiced) == pytest.approx(1.0)


def test_summarize_reports_all_numbers():
    s = summarize([440.0, None, 880.0], 440.0)
    assert s["accepted_frac"] == pytest.approx(2 / 3)
    assert s["gross_error_pct"] == pytest.approx(50.0)
    assert s["octave_error_pct"] == pytest.approx(50.0)


def test_latency_stats():
    mean, std = latency_stats([100, 110, 90, 105, 95, 100, 100, 100, 100, 100])
    assert mean == pytest.approx(100.0)
    assert std > 0


def test_latency_needs_enough_trials():
    with pytest.raises(ValueError):
        latency_stats([100, 110, 90])