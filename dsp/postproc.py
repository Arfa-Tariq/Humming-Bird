"""Post-processing of raw pitch estimates (the "make it playable" row).

Chain inside PitchTracker.update():
    f0 (Hz or None) -> semitones -> octave guard -> median -> one-pole smoother

All steps after the Hz-to-semitone conversion work in SEMITONES, because pitch
perception is logarithmic (equal ratios sound like equal steps).
Only numpy is used, so this module is safe for the live app and experiments.
"""
from collections import deque

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# --------------------------------------------------------------------------
# Hz <-> semitones <-> note names
# --------------------------------------------------------------------------
def hz_to_semitone(f_hz, f_ref=440.0):
    """s = 12 * log2(f / f_ref). Missing or non-positive frequencies give nan."""
    f = np.asarray(f_hz, dtype=float)               # None -> nan
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 12.0 * np.log2(f / f_ref)
    out = np.where(f > 0, s, np.nan)
    return float(out) if out.ndim == 0 else out


def semitone_to_hz(s, f_ref=440.0):
    """Inverse of hz_to_semitone."""
    out = f_ref * 2.0 ** (np.asarray(s, dtype=float) / 12.0)
    return float(out) if out.ndim == 0 else out


def hz_to_note(f_hz, a4=440.0):
    """Return (note_name, cents_offset), e.g. 445 Hz -> ('A4', +19.6).
    cents_offset is in [-50, +50] relative to the nearest note."""
    if f_hz is None or not np.isfinite(f_hz) or f_hz <= 0:
        return None, float("nan")
    midi = 69.0 + 12.0 * np.log2(f_hz / a4)         # 69 = MIDI number of A4
    nearest = int(round(midi))
    cents = 100.0 * (midi - nearest)
    name = NOTE_NAMES[nearest % 12] + str(nearest // 12 - 1)
    return name, float(cents)


# --------------------------------------------------------------------------
# Octave guard
# --------------------------------------------------------------------------
class OctaveGuard:
    """Reject sudden huge jumps (almost always octave errors).

    If the new value is more than max_jump semitones from the previous output,
    hold the previous output. Hold for at most max_hold hops in a row; after
    that accept the new value (the singer really did jump).
    """

    def __init__(self, max_jump=8.0, max_hold=3):
        if max_jump <= 0:
            raise ValueError("max_jump must be positive.")
        if int(max_hold) != max_hold or max_hold < 0:
            raise ValueError("max_hold must be a non-negative integer.")
        self.max_jump = float(max_jump)
        self.max_hold = int(max_hold)
        self.reset()

    def reset(self):
        self._prev = None
        self._held = 0

    def update(self, s):
        s = float(s)
        if self._prev is None:                      # first value: nothing to compare
            self._prev = s
            return s
        if abs(s - self._prev) > self.max_jump and self._held < self.max_hold:
            self._held += 1
            return self._prev                       # hold the old value
        self._prev = s                              # normal move, or jump accepted
        self._held = 0
        return s


# --------------------------------------------------------------------------
# Median filter (nonlinear)
# --------------------------------------------------------------------------
class MedianFilter:
    """Median of the last `length` values. Removes isolated spikes but keeps
    real steps sharp. Use an ODD length (1, 3, 5, 9); length 1 does nothing.
    Delay is (length - 1) / 2 hops. Until `length` values have arrived it
    takes the median of what it has."""

    def __init__(self, length=5):
        if int(length) != length or length < 1 or length % 2 == 0:
            raise ValueError("Median length must be an odd positive integer.")
        self.length = int(length)
        self._buf = deque(maxlen=self.length)

    @property
    def delay_hops(self):
        return (self.length - 1) // 2

    def reset(self):
        self._buf.clear()

    def update(self, x):
        self._buf.append(float(x))
        return float(np.median(self._buf))


# --------------------------------------------------------------------------
# One-pole smoother (first-order IIR low-pass)
# --------------------------------------------------------------------------
class OnePoleSmoother:
    """y[t] = y[t-1] + a * (x[t] - y[t-1])      (pole at 1 - a)

    The "LERP" of the original draft, but it IS a first-order IIR low-pass:
        H(z) = a / (1 - (1 - a) z^-1)
    Smaller a = smoother but laggier. `initial=None` starts at the first input
    (no sweep from nowhere); pass initial=0.0 to measure a step response from 0.
    """

    def __init__(self, a=0.18, initial=None):
        if not 0.0 < a <= 1.0:
            raise ValueError("Smoother coefficient a must be in (0, 1].")
        self.a = float(a)
        self._initial = initial
        self.reset()

    @property
    def pole(self):
        return 1.0 - self.a

    def reset(self):
        self._y = None if self._initial is None else float(self._initial)

    def update(self, target):
        target = float(target)
        if self._y is None:
            self._y = target
        else:
            self._y += self.a * (target - self._y)
        return self._y

    def time_constant_frames(self):
        """Updates needed to cover 63.2 % of a step: -1 / ln(1 - a)."""
        return 0.0 if self.a >= 1.0 else float(-1.0 / np.log(1.0 - self.a))

    def cutoff_hz(self, rate_hz):
        """-3 dB frequency when update() is called rate_hz times per second.
        Solves |H|^2 = 1/2:  cos(w) = (1 + p^2 - 2a^2) / (2p),  p = 1 - a.
        If there is no -3 dB point below Nyquist, returns Nyquist."""
        a, p = self.a, 1.0 - self.a
        if p == 0.0:
            return rate_hz / 2.0
        c = np.clip((1.0 + p * p - 2.0 * a * a) / (2.0 * p), -1.0, 1.0)
        return float(np.arccos(c) / (2.0 * np.pi) * rate_hz)

    def response(self, rate_hz, n_points=512):
        """Return (freqs_hz, mag_db) of the smoother, for the E6 plot."""
        w = np.linspace(0.0, np.pi, n_points)
        h = self.a / (1.0 - self.pole * np.exp(-1j * w))
        return w / (2.0 * np.pi) * rate_hz, 20.0 * np.log10(np.abs(h) + 1e-12)


# --------------------------------------------------------------------------
# PitchTracker (the frozen interface)
# --------------------------------------------------------------------------
class PitchTracker:
    """Turns a stream of raw estimates into one smooth semitone value.

    Call update() ONCE PER ANALYSIS HOP. So the smoother runs at the hop rate
    (62.5 Hz for hop 256 at 16 kHz), not at 60 FPS: its time constant is
    time_constant_frames() / hop_rate, about 4 % shorter than the 84 ms quoted
    for 60 FPS. State this in the report.

    update(f0_or_None, gate_open) returns the smoothed semitone value relative
    to f_ref (A4 = 440 Hz), or None until the first voiced estimate arrives.
    When the gate is closed, or the estimator rejected the frame, the last
    pitch is held (the bird does not snap down).
    The intermediate values are kept as attributes for the dashboard:
    raw_semitone, guarded_semitone, median_semitone, output, voiced.
    """

    def __init__(self, f_ref=440.0, median_len=5, smooth_a=0.18,
                 max_jump=8.0, max_hold=3):
        self.f_ref = float(f_ref)
        self.guard = OctaveGuard(max_jump, max_hold)
        self.median = MedianFilter(median_len)
        self.smoother = OnePoleSmoother(smooth_a)
        self.reset()

    def reset(self):
        self.guard.reset()
        self.median.reset()
        self.smoother.reset()
        self.raw_semitone = None
        self.guarded_semitone = None
        self.median_semitone = None
        self.output = None
        self.voiced = False

    def update(self, f0_or_None, gate_open=True):
        s = None
        if gate_open and f0_or_None is not None:
            f = float(f0_or_None)
            if np.isfinite(f) and f > 0:
                s = hz_to_semitone(f, self.f_ref)

        self.raw_semitone = s
        self.voiced = s is not None

        if s is not None:
            self.guarded_semitone = self.guard.update(s)
            self.median_semitone = self.median.update(self.guarded_semitone)

        if self.median_semitone is None:            # nothing heard yet
            return None
        self.output = self.smoother.update(self.median_semitone)
        return self.output