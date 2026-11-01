import numpy as np
import pytest

from dsp.windows import (
    rectangular_window,
    hann_window,
    hamming_window,
    blackman_window,
    get_window
)


N = 256


def test_rectangular_window():
    window = rectangular_window(N)

    assert len(window) == N
    assert np.all(window == 1)


def test_hann_window():
    window = hann_window(N)

    assert len(window) == N
    assert np.all(window >= 0)
    assert np.all(window <= 1)


def test_hamming_window():
    window = hamming_window(N)

    assert len(window) == N
    assert np.all(window >= 0)
    assert np.all(window <= 1)


def test_blackman_window():
    window = blackman_window(N)

    assert len(window) == N
    assert np.all(window >= -1e-10)
    assert np.all(window <= 1)


def test_get_window():
    window = get_window("hann", N)

    assert len(window) == N
    np.testing.assert_array_equal(window, hann_window(N))


def test_all_window_types():
    names = ["rectangular", "hann", "hamming", "blackman"]

    for name in names:
        window = get_window(name, N)
        assert len(window) == N


def test_invalid_window_name():
    with pytest.raises(ValueError):
        get_window("unknown", N)


def test_invalid_window_size():
    with pytest.raises(ValueError):
        hann_window(0)