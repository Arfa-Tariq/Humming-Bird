import numpy as np
import pytest

from dsp import synth
from dsp.estimators import estimate_pitch, METHODS
from dsp.metrics import cents_error

FS = 16000
N = 1024
FMIN, FMAX = 70, 1000          # margin below the lowest test note (80 Hz)

# tolerances (cents). AMDF and ACF are limited by the integer-lag grid plus a
# parabolic fit; FFT with zero-padding + interpolation is much tighter.
TOL = {"fft": 2.0, "amdf": 15.0, "acf": 15.0}


def middle_frame(signal, n=N, start=2000):
    return signal[start:start + n]


def semitone_grid(lo=82.0, hi=980.0):
    f = lo * 2.0 ** (np.arange(0, 60) / 12.0)
    return f[f <= hi]


# --------------------------------------------------------------------------
# easy cases: every method must work
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_440_tone_within_tolerance(method):
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=440))
    f0, info = estimate_pitch(x, FS, method, FMIN, FMAX)

    assert f0 is not None
    assert abs(cents_error(f0, 440.0)[0]) < TOL[method]
    assert 0.0 <= info["confidence"] <= 1.0


@pytest.mark.parametrize("method", METHODS)
def test_pure_tone_semitone_sweep(method):
    for f in semitone_grid():
        x = middle_frame(synth.pure_tone(FS, 0.5, frequency=f))
        f0, _ = estimate_pitch(x, FS, method, FMIN, FMAX)
        assert f0 is not None, f"{method} rejected {f:.1f} Hz"
        assert abs(cents_error(f0, f)[0]) < TOL[method], f"{method} at {f:.1f} Hz"


@pytest.mark.parametrize("method", METHODS)
def test_harmonic_hum_semitone_sweep(method):
    for f in semitone_grid():
        x = middle_frame(synth.harmonic_hum(FS, 0.5, f0=f))
        f0, _ = estimate_pitch(x, FS, method, FMIN, FMAX)
        assert f0 is not None, f"{method} rejected {f:.1f} Hz"
        assert abs(cents_error(f0, f)[0]) < TOL[method], f"{method} at {f:.1f} Hz"


# --------------------------------------------------------------------------
# silence and rejection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_silence_is_rejected(method):
    f0, info = estimate_pitch(np.zeros(N), FS, method, FMIN, FMAX)

    assert f0 is None
    assert info["confidence"] == 0.0


@pytest.mark.parametrize("method", ["amdf", "acf"])
def test_noise_only_is_mostly_rejected_by_time_domain_methods(method):
    noise = synth.silence_noise(FS, 2.0, seed=0, noise_rms=0.05)
    accepted = 0
    frames = 0
    for start in range(0, len(noise) - N, 256):
        f0, _ = estimate_pitch(noise[start:start + N], FS, method, FMIN, FMAX)
        accepted += f0 is not None
        frames += 1
    # a loose sanity bound, not a result: the proper numbers come from E2
    assert accepted / frames < 0.5


# --------------------------------------------------------------------------
# FFT specifics (E1: what zero-padding and interpolation fix)
# --------------------------------------------------------------------------
def test_interpolation_beats_raw_bins_on_off_bin_tone():
    f_true = 450.0                                  # 28.8 bins, not an integer
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=f_true))

    raw, _ = estimate_pitch(x, FS, "fft", FMIN, FMAX, pad_factor=1, interpolate=False)
    fine, _ = estimate_pitch(x, FS, "fft", FMIN, FMAX, pad_factor=4, interpolate=True)

    raw_err = abs(cents_error(raw, f_true)[0])
    fine_err = abs(cents_error(fine, f_true)[0])
    assert fine_err < raw_err
    assert fine_err < 2.0


@pytest.mark.parametrize("window", ["rect", "rectangular", "hann", "hamming", "blackman"])
def test_fft_accepts_all_windows(window):
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    f0, _ = estimate_pitch(x, FS, "fft", FMIN, FMAX, window=window)

    assert abs(cents_error(f0, 300.0)[0]) < 15.0


def test_fft_info_has_spectrum_and_axis():
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    _, info = estimate_pitch(x, FS, "fft", FMIN, FMAX)

    assert len(info["spectrum"]) == len(info["freqs"])
    assert info["freqs"][-1] == pytest.approx(FS / 2)
    assert info["d_curve"] is None and info["lags"] is None


def test_fft_confidence_is_higher_for_tone_than_noise():
    tone = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    noise = synth.silence_noise(FS, 0.5, seed=1, noise_rms=0.1)[:N]

    _, info_tone = estimate_pitch(tone, FS, "fft", FMIN, FMAX)
    _, info_noise = estimate_pitch(noise, FS, "fft", FMIN, FMAX)

    assert info_tone["confidence"] > 0.8
    assert info_noise["confidence"] < info_tone["confidence"]


# --------------------------------------------------------------------------
# time-domain specifics
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["amdf", "acf"])
def test_time_domain_info_has_curve_and_lags(method):
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    f0, info = estimate_pitch(x, FS, method, FMIN, FMAX)

    assert info["d_curve"] is not None
    assert len(info["d_curve"]) == len(info["lags"])
    assert info["spectrum"] is None and info["freqs"] is None
    # the chosen lag must sit next to the curve's valley/peak
    chosen_lag = FS / f0
    assert info["lags"][0] <= chosen_lag <= info["lags"][-1]


def test_amdf_confidence_is_high_for_clean_tone():
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    _, info = estimate_pitch(x, FS, "amdf", FMIN, FMAX)

    assert info["confidence"] > 0.8


def test_amdf_threshold_is_tunable():
    x = middle_frame(synth.harmonic_hum(FS, 0.5, f0=200))

    for threshold in (0.3, 0.4, 0.5, 0.6):
        f0, _ = estimate_pitch(x, FS, "amdf", FMIN, FMAX, threshold=threshold)
        assert f0 is None or f0 > 0


def test_amdf_impossible_threshold_rejects_frame():
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    f0, _ = estimate_pitch(x, FS, "amdf", FMIN, FMAX, threshold=0.0)

    assert f0 is None


def test_amdf_first_valley_rule_avoids_octave_down_on_hum():
    # the multiples of the period (2T, 3T, ...) are valleys too; the first-valley
    # rule must not return f0/2 on an ordinary harmonic hum
    for f in (110.0, 196.0, 330.0, 523.0):
        x = middle_frame(synth.harmonic_hum(FS, 0.5, f0=f))
        f0, _ = estimate_pitch(x, FS, "amdf", FMIN, FMAX)
        assert abs(cents_error(f0, f)[0]) < 50.0


# --------------------------------------------------------------------------
# robustness and plumbing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_tone_at_20db_snr_stays_close(method):
    clean = synth.pure_tone(FS, 0.5, frequency=330)
    noisy = synth.add_noise(clean, snr_db=20, seed=42)
    f0, _ = estimate_pitch(middle_frame(noisy), FS, method, FMIN, FMAX)

    assert f0 is not None
    assert abs(cents_error(f0, 330.0)[0]) < 50.0


@pytest.mark.parametrize("method", METHODS)
def test_result_is_deterministic(method):
    x = middle_frame(synth.harmonic_hum(FS, 0.5, f0=220))
    a, _ = estimate_pitch(x, FS, method, FMIN, FMAX)
    b, _ = estimate_pitch(x, FS, method, FMIN, FMAX)

    assert a == b


@pytest.mark.parametrize("method", METHODS)
def test_weak_fundamental_runs_and_stays_in_range(method):
    # Known hard case: no accuracy claim here. How often each method makes an
    # octave error on this signal is measured in E1 / E9, not asserted.
    x = middle_frame(synth.weak_fundamental(FS, 0.5, f0=200))
    f0, _ = estimate_pitch(x, FS, method, FMIN, FMAX)

    assert f0 is None or FMIN * 0.9 <= f0 <= FMAX * 1.1


def test_method_alias_and_case_are_accepted():
    x = middle_frame(synth.pure_tone(FS, 0.5, frequency=300))
    a, _ = estimate_pitch(x, FS, "FFT", FMIN, FMAX)
    b, _ = estimate_pitch(x, FS, "fft_peak", FMIN, FMAX)

    assert a == b


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        estimate_pitch(np.ones(N), FS, "yin", FMIN, FMAX)


def test_invalid_inputs_raise():
    good = np.sin(np.arange(N))

    with pytest.raises(ValueError):
        estimate_pitch(np.ones((4, 4)), FS, "fft", FMIN, FMAX)       # not 1-D
    with pytest.raises(ValueError):
        estimate_pitch(np.ones(8), FS, "fft", FMIN, FMAX)            # too short
    with pytest.raises(ValueError):
        estimate_pitch(np.append(good, np.nan), FS, "fft", FMIN, FMAX)
    with pytest.raises(ValueError):
        estimate_pitch(good, FS, "fft", 1000, 70)                    # fmin > fmax
    with pytest.raises(ValueError):
        estimate_pitch(good, FS, "fft", FMIN, FS)                    # fmax above Nyquist
    with pytest.raises(ValueError):
        estimate_pitch(good, 0, "fft", FMIN, FMAX)                   # bad fs


def test_works_at_both_candidate_sampling_rates():
    # E3 compares 16 kHz / N=1024 with 44.1 kHz / N=2048
    for fs, n in ((16000, 1024), (44100, 2048)):
        x = synth.pure_tone(fs, 0.5, frequency=220)[3000:3000 + n]
        for method in METHODS:
            f0, _ = estimate_pitch(x, fs, method, 70, 1000)
            assert abs(cents_error(f0, 220.0)[0]) < 15.0, f"{method} fs={fs}"