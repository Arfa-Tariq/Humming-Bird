import numpy as np
import pytest

from dsp.noise_gate import NoiseGate


def test_noise_gate_starts_closed():
    gate = NoiseGate()

    assert gate.is_open is False


def test_noise_gate_opens_for_loud_signal():
    gate = NoiseGate(initial_noise_floor=0.01)

    signal = np.ones(1024) * 0.1

    gate.process(signal)

    assert gate.is_open is True


def test_noise_gate_closes_for_quiet_signal():
    gate = NoiseGate(initial_noise_floor=0.01)

    loud_signal = np.ones(1024) * 0.1
    quiet_signal = np.ones(1024) * 0.001

    gate.process(loud_signal)
    gate.process(quiet_signal)

    assert gate.is_open is False


def test_noise_gate_removes_background_noise():
    gate = NoiseGate(initial_noise_floor=0.01)

    quiet_signal = np.ones(1024) * 0.001

    output = gate.process(quiet_signal)

    assert np.all(output == 0)


def test_noise_gate_preserves_loud_signal():
    gate = NoiseGate(initial_noise_floor=0.01)

    loud_signal = np.ones(1024) * 0.1

    output = gate.process(loud_signal)

    np.testing.assert_array_equal(output, loud_signal)


def test_noise_gate_rejects_empty_signal():
    gate = NoiseGate()

    with pytest.raises(ValueError):
        gate.process(np.array([]))


def test_noise_gate_rejects_invalid_signal():
    gate = NoiseGate()

    signal = np.array([1, 2, np.nan])

    with pytest.raises(ValueError):
        gate.process(signal)