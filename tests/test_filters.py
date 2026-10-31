import numpy as np
import pytest

from dsp.filters import bandpass_filter, decimate_signal


FS = 16000
DURATION = 2


def test_bandpass_filter_length():
    t = np.arange(FS * DURATION) / FS
    signal = np.sin(2 * np.pi * 220 * t)

    filtered = bandpass_filter(signal, FS)

    assert len(filtered) == len(signal)


def test_bandpass_filter_removes_low_frequency():
    t = np.arange(FS * DURATION) / FS

    signal = np.sin(2 * np.pi * 20 * t)

    filtered = bandpass_filter(signal, FS)

    assert np.std(filtered) < np.std(signal)


def test_bandpass_filter_keeps_humming_frequency():
    t = np.arange(FS * DURATION) / FS

    signal = np.sin(2 * np.pi * 220 * t)

    filtered = bandpass_filter(signal, FS)

    assert np.std(filtered) > 0.5 * np.std(signal)


def test_decimator_reduces_samples():
    signal = np.sin(2 * np.pi * 220 * np.arange(16000) / FS)

    reduced = decimate_signal(signal, factor=2)

    assert len(reduced) == 8000


def test_decimator_factor_four():
    signal = np.sin(2 * np.pi * 220 * np.arange(16000) / FS)

    reduced = decimate_signal(signal, factor=4)

    assert len(reduced) == 4000


def test_decimator_factor_one():
    signal = np.array([1, 2, 3, 4, 5])

    reduced = decimate_signal(signal, factor=1)

    np.testing.assert_array_equal(reduced, signal)


def test_invalid_filter_frequency():
    signal = np.ones(1000)

    with pytest.raises(ValueError):
        bandpass_filter(signal, FS, lowcut=1000, highcut=500)


def test_invalid_decimation_factor():
    signal = np.ones(1000)

    with pytest.raises(ValueError):
        decimate_signal(signal, factor=0)