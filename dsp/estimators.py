"""Pitch estimators: FFT peak, AMDF and normalized autocorrelation (ACF).

Common interface (frozen contract):

    f0_hz_or_None, info = estimate_pitch(frame, fs, method, fmin, fmax, window='hann')

``info`` always has the same keys so the dashboard and experiments never need
to special-case a method:

    confidence : float in [0, 1] (meaning depends on the method, see below)
    spectrum   : magnitude spectrum (FFT method only, else None)
    freqs      : frequency axis in Hz for ``spectrum`` (FFT method only, else None)
    d_curve    : AMDF curve D[tau] or ACF curve r[tau] (time-domain methods, else None)
    lags       : lag axis in samples for ``d_curve`` (time-domain methods, else None)
    method     : canonical method name

A rejected frame (unvoiced, silent, no valid valley/peak) returns ``None`` for f0.
Only numpy is used, so this module is safe for both the live app and experiments.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dsp.windows import get_window

METHODS = ("fft", "amdf", "acf")
_ALIASES = {"fft_peak": "fft"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _check_inputs(frame, fs, fmin, fmax):
    x = np.asarray(frame, dtype=float)
    if x.ndim != 1:
        raise ValueError("Frame must be one-dimensional.")
    if x.size < 16:
        raise ValueError("Frame must contain at least 16 samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Frame must contain only finite values.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not (0 < fmin < fmax < fs / 2):
        raise ValueError("Need 0 < fmin < fmax < fs/2.")
    return x


def _empty_info(method, confidence=0.0):
    return {
        "confidence": float(confidence),
        "spectrum": None,
        "freqs": None,
        "d_curve": None,
        "lags": None,
        "method": method,
    }


def _parabolic_offset(y_m1, y0, y_p1):
    """Offset (in samples/bins, within [-1, 1]) of the vertex of the parabola
    through (-1, y_m1), (0, y0), (1, y_p1). Works for both peaks and valleys."""
    denom = y_m1 - 2.0 * y0 + y_p1
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(0.5 * (y_m1 - y_p1) / denom, -1.0, 1.0))


def _lag_range(fs, fmin, fmax, n):
    """Lag search range [tau_lo, tau_hi] in samples. The upper lag is capped at
    N/2 so at least two periods fit in the frame; this raises the effective fmin
    for short frames (document it if it matters in an experiment)."""
    tau_lo = max(2, int(np.floor(fs / fmax)))
    tau_hi = min(int(np.ceil(fs / fmin)), n // 2 - 1)
    return tau_lo, tau_hi


# --------------------------------------------------------------------------
# FFT peak
# --------------------------------------------------------------------------
def fft_peak(frame, fs, fmin, fmax, window="hann",
             pad_factor=4, interpolate=True, min_confidence=0.0):
    """Peak of the windowed, zero-padded spectrum inside fmin..fmax.

    pad_factor  : FFT length is pad_factor * N (4 by default as in the spec).
                  pad_factor=1 with interpolate=False gives the raw-bin result
                  that E1 compares against.
    interpolate : parabolic interpolation on the log magnitude of the 3 bins
                  around the peak.
    confidence  : fraction of the in-band power that sits within +-2 original
                  bins of the peak. Close to 1 for a clean tone, small for noise.
                  Frames below ``min_confidence`` are rejected (default 0 = keep).
    """
    x = _check_inputs(frame, fs, fmin, fmax)
    if int(pad_factor) != pad_factor or pad_factor < 1:
        raise ValueError("pad_factor must be a positive integer.")
    pad_factor = int(pad_factor)

    n = x.size
    x = (x - x.mean()) * get_window(window, n)
    nfft = pad_factor * n
    spec = np.abs(np.fft.rfft(x, nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    info = _empty_info("fft")
    info["spectrum"] = spec
    info["freqs"] = freqs

    band = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
    if band.size < 3:
        return None, info

    power = spec ** 2
    band_power = power[band].sum()
    if band_power <= 1e-24:
        return None, info

    k = band[0] + int(np.argmax(spec[band]))

    half = 2 * pad_factor
    lo, hi = max(0, k - half), min(len(power), k + half + 1)
    confidence = float(np.clip(power[lo:hi].sum() / band_power, 0.0, 1.0))
    info["confidence"] = confidence
    if confidence < min_confidence:
        return None, info

    if interpolate and 1 <= k < len(spec) - 1:
        eps = 1e-30
        delta = _parabolic_offset(
            np.log(spec[k - 1] + eps), np.log(spec[k] + eps), np.log(spec[k + 1] + eps)
        )
    else:
        delta = 0.0
    f0 = (k + delta) * fs / nfft
    return float(f0), info


# --------------------------------------------------------------------------
# AMDF
# --------------------------------------------------------------------------
def amdf(frame, fs, fmin, fmax, threshold=0.5):
    """Average magnitude difference function, first-valley rule.

    D[tau] = mean_n |x[n] - x[n + tau]|, computed over a fixed comparison length
    for every tau so the values are comparable. The pitch period is the SMALLEST
    tau that is a local minimum with D[tau] < threshold * mean(D); this avoids
    the multiples of the period (octave-down errors).

    confidence = 1 - D[tau] / mean(D). If no valley passes the threshold the frame
    is rejected (None).
    """
    x = _check_inputs(frame, fs, fmin, fmax)
    x = x - x.mean()
    n = x.size

    tau_lo, tau_hi = _lag_range(fs, fmin, fmax, n)
    if tau_hi - tau_lo < 3:
        return None, _empty_info("amdf")

    taus = np.arange(tau_lo - 1, tau_hi + 2)      # one extra lag on each side
    length = n - taus[-1]
    shifted = sliding_window_view(x, length)       # shifted[t] = x[t : t + length]
    d = np.mean(np.abs(x[:length][None, :] - shifted[taus]), axis=1)

    info = _empty_info("amdf")
    info["d_curve"] = d[1:-1]
    info["lags"] = taus[1:-1]

    mean_d = d[1:-1].mean()
    if mean_d <= 1e-12:
        return None, info

    j = np.arange(1, len(d) - 1)
    is_valley = (d[j] < d[j - 1]) & (d[j] <= d[j + 1])
    passes = d[j] < threshold * mean_d
    candidates = j[is_valley & passes]

    if candidates.size == 0:
        info["confidence"] = float(np.clip(1.0 - d[1:-1].min() / mean_d, 0.0, 1.0))
        return None, info

    j0 = int(candidates[0])
    delta = _parabolic_offset(d[j0 - 1], d[j0], d[j0 + 1])
    tau = taus[j0] + delta
    info["confidence"] = float(np.clip(1.0 - d[j0] / mean_d, 0.0, 1.0))
    return float(fs / tau), info


# --------------------------------------------------------------------------
# Normalized ACF
# --------------------------------------------------------------------------
def acf(frame, fs, fmin, fmax, rel_threshold=0.9, min_confidence=0.3):
    """Normalized autocorrelation computed through the FFT.

    r[tau] is the inverse FFT of the power spectrum, normalized by r[0] and
    corrected for the shrinking overlap (unbiased), so a perfectly periodic
    signal has peaks of height ~1 at every multiple of the period.
    The chosen peak is the FIRST local maximum whose height is at least
    ``rel_threshold`` times the tallest peak, which avoids octave-down errors.

    confidence = height of the chosen peak (0..1). Frames whose tallest peak is
    below ``min_confidence`` are rejected.
    """
    x = _check_inputs(frame, fs, fmin, fmax)
    x = x - x.mean()
    n = x.size

    tau_lo, tau_hi = _lag_range(fs, fmin, fmax, n)
    if tau_hi - tau_lo < 3:
        return None, _empty_info("acf")

    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    power = np.abs(np.fft.rfft(x, nfft)) ** 2
    r = np.fft.irfft(power, nfft)[:n]

    info = _empty_info("acf")
    if r[0] <= 1e-20:
        return None, info
    r = r / r[0]
    r = r * n / (n - np.arange(n))                # unbiased correction

    taus = np.arange(tau_lo - 1, tau_hi + 2)
    c = r[taus]
    info["d_curve"] = c[1:-1]
    info["lags"] = taus[1:-1]

    j = np.arange(1, len(c) - 1)
    is_peak = (c[j] > c[j - 1]) & (c[j] >= c[j + 1])
    peaks = j[is_peak]
    if peaks.size == 0:
        return None, info

    tallest = c[peaks].max()
    info["confidence"] = float(np.clip(tallest, 0.0, 1.0))
    if tallest < min_confidence:
        return None, info

    j0 = int(peaks[np.argmax(c[peaks] >= rel_threshold * tallest)])
    delta = _parabolic_offset(c[j0 - 1], c[j0], c[j0 + 1])
    tau = taus[j0] + delta
    info["confidence"] = float(np.clip(c[j0], 0.0, 1.0))
    return float(fs / tau), info


# --------------------------------------------------------------------------
# Dispatcher (the frozen interface)
# --------------------------------------------------------------------------
def estimate_pitch(frame, fs, method, fmin, fmax, window="hann", **params):
    """Estimate f0 of one frame. Returns (f0_hz or None, info_dict).

    method  : 'fft', 'amdf' or 'acf'
    window  : used by the FFT method only (AMDF and ACF work on the raw frame)
    params  : forwarded to the chosen estimator, e.g. threshold=0.4 for AMDF,
              pad_factor=1, interpolate=False for the raw-bin FFT in E1.
    """
    key = str(method).lower()
    key = _ALIASES.get(key, key)
    if key == "fft":
        return fft_peak(frame, fs, fmin, fmax, window=window, **params)
    if key == "amdf":
        return amdf(frame, fs, fmin, fmax, **params)
    if key == "acf":
        return acf(frame, fs, fmin, fmax, **params)
    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")