import warnings

import numpy as np
from scipy.signal import (
    butter,
    cheby2,
    firwin,
    freqz,
    group_delay,
    lfilter,
    resample_poly,
    sos2zpk,
    sosfilt,
    sosfiltfilt,
    sosfreqz,
)


# --------------------------------------------------------------------------
# OFFLINE helpers (zero-phase / whole-signal). Fine for plots and checks, but
# NOT causal, so they must never be used in the live audio path. Use
# StreamingBandpass and StreamingDecimator below for anything live or for
# experiments that must match the live app.
# --------------------------------------------------------------------------

# 1. Band-pass filter
def bandpass_filter(signal, fs, lowcut=70, highcut=1000, order=4):

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("Signal must contain at least 2 samples.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain only finite values.")

    nyquist = fs / 2

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if lowcut <= 0 or highcut <= lowcut:
        raise ValueError("Invalid cutoff frequencies.")

    if highcut >= nyquist:
        raise ValueError("High cutoff must be below Nyquist frequency.")

    if order < 1:
        raise ValueError("Filter order must be at least 1.")

    # Design Butterworth band-pass filter
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    # Apply filter
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


# 2. Decimator
def decimate_signal(signal, factor=2):

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain only finite values.")

    if not isinstance(factor, (int, np.integer)) or isinstance(factor, bool):
        raise ValueError("Decimation factor must be an integer.")

    if factor < 1:
        raise ValueError("Decimation factor must be at least 1.")

    if factor == 1:
        return signal.copy()

    # Reduce sampling rate with anti-aliasing filtering
    reduced_signal = resample_poly(signal, up=1, down=factor)

    return reduced_signal


# --------------------------------------------------------------------------
# STREAMING (causal, state carried between blocks)
# --------------------------------------------------------------------------
def _check_block(block):
    x = np.asarray(block, dtype=float)
    if x.ndim != 1:
        raise ValueError("Block must be one-dimensional.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Block must contain only finite values.")
    return x


class StreamingBandpass:
    """Causal band-pass filter that carries its state from block to block.

    kind
      'butter' : Butterworth, f_lo/f_hi are the -3 dB corners.
      'cheby2' : Chebyshev type II with ``stopband_db`` attenuation. NOTE: scipy
                 treats f_lo/f_hi as the STOPBAND edges for this type, so its
                 -3 dB passband is a little narrower than the Butterworth one for
                 the same numbers. Say so when comparing in E5.
      'fir'    : linear-phase FIR (firwin, ``numtaps`` taps, default 255), constant
                 group delay of (numtaps - 1) / 2 samples.

    order
      Passed straight to scipy. For a band-pass the prototype order is doubled, so
      order=2 gives a 4th-order (4-pole) band-pass. Report it this way in E5.
      Ignored for 'fir'.

    The IIR filters run as second-order sections with sosfilt and carried ``zi``;
    the FIR runs with lfilter and carried ``zi``. This removes the click at every
    block edge that a stateless filter produces.
    """

    KINDS = ("butter", "cheby2", "fir")

    def __init__(self, fs, f_lo, f_hi, order=2, kind="butter",
                 stopband_db=40.0, numtaps=255):
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive.")
        if not (0 < f_lo < f_hi < fs / 2):
            raise ValueError("Need 0 < f_lo < f_hi < fs/2.")
        if kind not in self.KINDS:
            raise ValueError(f"Unknown filter kind '{kind}'. Choose from {self.KINDS}.")
        if kind != "fir":
            if int(order) != order or order < 1:
                raise ValueError("Filter order must be a positive integer.")
            order = int(order)

        self.fs = float(fs)
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)
        self.order = order
        self.kind = kind
        self.sos = None
        self.taps = None

        if kind == "butter":
            self.sos = butter(order, [f_lo, f_hi], btype="bandpass",
                              fs=fs, output="sos")
        elif kind == "cheby2":
            self.sos = cheby2(order, stopband_db, [f_lo, f_hi], btype="bandpass",
                              fs=fs, output="sos")
        else:
            if int(numtaps) != numtaps or numtaps < 3:
                raise ValueError("numtaps must be an integer of at least 3.")
            self.taps = firwin(int(numtaps), [f_lo, f_hi], pass_zero=False, fs=fs)

        self.reset()

    # ---- streaming ----
    def reset(self):
        """Clear the filter memory (use when starting a new recording)."""
        if self.sos is not None:
            self._zi = np.zeros((self.sos.shape[0], 2))
        else:
            self._zi = np.zeros(len(self.taps) - 1)

    def process(self, block):
        """Filter one block and keep the state for the next one."""
        x = _check_block(block)
        if x.size == 0:
            return x.copy()
        if self.sos is not None:
            y, self._zi = sosfilt(self.sos, x, zi=self._zi)
        else:
            y, self._zi = lfilter(self.taps, [1.0], x, zi=self._zi)
        return y

    # ---- analysis ----
    def response(self, n_points=2048):
        """Return (freqs_hz, mag_db, group_delay_ms) for plotting.

        Group delay is in milliseconds (the unit E5 reports). It is computed per
        second-order section and summed, which stays accurate for higher orders.
        """
        if self.sos is not None:
            freqs, h = sosfreqz(self.sos, worN=n_points, fs=self.fs)
            gd_samples = np.zeros(len(freqs))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for section in self.sos:
                    _, gd = group_delay((section[:3], section[3:]), w=freqs, fs=self.fs)
                    gd_samples += gd
        else:
            freqs, h = freqz(self.taps, worN=n_points, fs=self.fs)
            gd_samples = np.full(len(freqs), (len(self.taps) - 1) / 2.0)

        mag_db = 20.0 * np.log10(np.abs(h) + 1e-12)
        gd_ms = 1000.0 * gd_samples / self.fs
        return freqs, mag_db, gd_ms

    def gain_db_at(self, freqs_hz):
        """Gain in dB at specific frequencies, e.g. gain_db_at([50, 3000])."""
        f = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
        if self.sos is not None:
            _, h = sosfreqz(self.sos, worN=f, fs=self.fs)
        else:
            _, h = freqz(self.taps, worN=f, fs=self.fs)
        return 20.0 * np.log10(np.abs(h) + 1e-12)

    def mean_passband_delay_ms(self, n_points=4096):
        """Mean group delay (ms) over f_lo..f_hi. The delay number used in E5."""
        freqs, _, gd_ms = self.response(n_points)
        inside = (freqs >= self.f_lo) & (freqs <= self.f_hi) & np.isfinite(gd_ms)
        return float(np.mean(gd_ms[inside]))

    def poles_zeros(self):
        """Return (zeros, poles, gain) for the pole-zero diagram."""
        if self.sos is not None:
            return sos2zpk(self.sos)
        zeros = np.roots(self.taps)
        poles = np.zeros(len(self.taps) - 1)
        return zeros, poles, 1.0


class StreamingDecimator:
    """Causal anti-alias low-pass + keep every M-th sample, state carried.

    Use an integer factor that lands on a convenient rate, for example
    48000 -> 16000 (factor 3) or 44100 -> 11025 (factor 4). If the sound card can
    open at the processing rate directly, skip this block.

    scipy.signal.decimate(zero_phase=True) is NOT used because it is not causal
    across blocks.
    """

    def __init__(self, fs_in, factor, numtaps=None):
        if fs_in <= 0:
            raise ValueError("Input sampling frequency must be positive.")
        if not isinstance(factor, (int, np.integer)) or isinstance(factor, bool) or factor < 1:
            raise ValueError("Decimation factor must be an integer of at least 1.")

        self.fs_in = float(fs_in)
        self.factor = int(factor)
        self.fs_out = self.fs_in / self.factor

        if numtaps is None:
            numtaps = 16 * self.factor + 1
        self.numtaps = int(numtaps)

        if self.factor == 1:
            self.taps = np.array([1.0])
        else:
            cutoff = 0.45 * self.fs_out                # just below the new Nyquist
            self.taps = firwin(self.numtaps, cutoff, fs=self.fs_in)
        self.reset()

    @property
    def delay_ms(self):
        """Group delay of the anti-alias filter in milliseconds."""
        return 1000.0 * (len(self.taps) - 1) / 2.0 / self.fs_in

    def reset(self):
        self._zi = np.zeros(len(self.taps) - 1)
        self._count = 0                                # input samples seen so far

    def process(self, block):
        x = _check_block(block)
        if x.size == 0:
            return x.copy()
        if self.factor == 1:
            self._count += x.size
            return x.copy()

        y, self._zi = lfilter(self.taps, [1.0], x, zi=self._zi)
        start = (-self._count) % self.factor            # keep global indices = 0 mod M
        self._count += x.size
        return y[start::self.factor]