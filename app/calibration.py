"""Non-blocking microphone calibration and pitch-to-height mapping.

The UI calls ``start``, then ``update`` once per frame. After the noise
measurement, Enter starts each held-note recording. Only distinct, voiced
analysis frames are used, so a slow or silent microphone cannot produce a
false calibration.
"""
import math
from dataclasses import dataclass
from statistics import median
import time

from dsp.postproc import hz_to_semitone


@dataclass(frozen=True)
class PitchRange:
    """Comfortable range in semitones relative to A4 (440 Hz)."""

    low: float
    high: float

    def __post_init__(self):
        if not (math.isfinite(self.low) and math.isfinite(self.high)
                and self.high > self.low):
            raise ValueError("The high pitch must be above the low pitch.")

    def height(self, semitone, top, bottom):
        """Map low to bottom and high to top, clamping outside the range."""
        if not math.isfinite(semitone) or bottom < top:
            raise ValueError("Expected a finite pitch and valid height bounds.")
        fraction = max(0.0, min(1.0, (semitone - self.low) / (self.high - self.low)))
        return bottom - fraction * (bottom - top)


def default_range(mode="hum"):
    """Usable demo range until a microphone has been calibrated."""
    low, high = (600.0, 1800.0) if mode == "whistle" else (100.0, 400.0)
    return PitchRange(hz_to_semitone(low), hz_to_semitone(high))


class Calibration:
    """Guide noise-floor, low-note, and high-note calibration without blocking."""

    def __init__(self, engine, noise_seconds=1.5, note_seconds=1.5,
                 min_frames=5, min_span=3.0, clock=time.perf_counter):
        self.engine = engine
        self.noise_seconds = noise_seconds
        self.note_seconds = note_seconds
        self.min_frames = min_frames
        self.min_span = min_span
        self.clock = clock
        self.phase = "idle"
        self.pitch_range = None
        self.low = None
        self.message = ""
        self._seen_noise = False
        self._samples = []
        self._last_frame = None
        self._deadline = 0.0

    def start(self):
        self.pitch_range = None
        self.low = None
        self.message = "Stay quiet while the room noise is measured."
        self._seen_noise = False
        self._noise_start_frame = self.engine.get_state().get("frame_id", -1)
        self.phase = "noise"
        self.engine.calibrate_noise(self.noise_seconds)

    def record_note(self, state=None):
        """Begin recording the currently prompted note; returns whether it started."""
        if self.phase not in ("low_ready", "high_ready"):
            return False
        self.phase = "low_recording" if self.phase == "low_ready" else "high_recording"
        self._samples = []
        self._last_frame = (state or {}).get("frame_id")
        self._deadline = self.clock() + self.note_seconds
        self.message = "Keep holding the note..."
        return True

    def update(self, state):
        if self.phase == "noise":
            self._seen_noise |= bool(state.get("calibrating"))
            new_frame = state.get("frame_id", -1) > self._noise_start_frame
            if ((self._seen_noise or new_frame) and not state.get("calibrating")
                    and state.get("noise_calibrated")):
                self.phase = "low_ready"
                self.message = "Hum your lowest comfortable note and press ENTER."
        elif self.phase in ("low_recording", "high_recording"):
            frame_id = state.get("frame_id")
            f0 = state.get("f0")
            if (frame_id is not None and frame_id != self._last_frame
                    and state.get("gate_open") and f0 is not None
                    and math.isfinite(f0) and f0 > 0
                    and state.get("confidence", 0.0) >= 0.3):
                self._samples.append(hz_to_semitone(f0))
                self._last_frame = frame_id
            if self.clock() >= self._deadline:
                self._finish_note()
        return self.phase

    def _finish_note(self):
        is_low = self.phase == "low_recording"
        if len(self._samples) < self.min_frames:
            self.phase = "low_ready" if is_low else "high_ready"
            self.message = "No steady pitch detected. Hold the note and press ENTER to retry."
            return
        note = median(self._samples)
        if is_low:
            self.low = note
            self.phase = "high_ready"
            self.message = "Hum your highest comfortable note and press ENTER."
        elif note < self.low + self.min_span:
            self.phase = "high_ready"
            self.message = "High note too close to low. Hold a higher note and retry."
        else:
            self.pitch_range = PitchRange(self.low, note)
            self.phase = "done"
            self.message = "Calibration complete."