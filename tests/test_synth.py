import numpy as np

from dsp.synth import (
    pure_tone,
    harmonic_hum,
    weak_fundamental,
    chirp_signal,
    add_noise,
    silence_noise
)

FS = 16000
DURATION = 2


def test_pure_tone():
    signal = pure_tone(FS, DURATION, frequency=440)

    assert len(signal) == 32000
    assert np.max(np.abs(signal)) <= 1


def test_harmonic_hum():
    signal = harmonic_hum(FS, DURATION, f0=220)

    assert len(signal) == 32000


def test_weak_fundamental():
    signal = weak_fundamental(FS, DURATION, f0=220)

    assert len(signal) == 32000


def test_chirp():
    signal = chirp_signal(FS, DURATION)

    assert len(signal) == 32000


def test_noise():
    signal = pure_tone(FS, DURATION, frequency=440)

    noisy_signal = add_noise(signal, snr_db=10, seed=42)

    noise = noisy_signal - signal

    measured_snr = 10 * np.log10(
        np.mean(signal ** 2) / np.mean(noise ** 2)
    )

    assert abs(measured_snr - 10) < 0.01


def test_silence_noise():
    signal = silence_noise(FS, DURATION)

    assert len(signal) == 32000