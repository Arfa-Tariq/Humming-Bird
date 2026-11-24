import math

import pytest

from app.calibration import Calibration, PitchRange, default_range
from dsp.postproc import hz_to_semitone


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.state = {"frame_id": 10, "noise_calibrated": True}

    def get_state(self):
        return self.state

    def calibrate_noise(self, seconds):
        self.calls.append(seconds)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


def voiced(frame_id, f0, gate=True, confidence=0.95):
    return {"frame_id": frame_id, "f0": f0, "gate_open": gate, "confidence": confidence}


def record(cal, clock, start_id, f0):
    cal.record_note({"frame_id": start_id})
    for i in range(1, 6):
        clock.now += 0.2
        cal.update(voiced(start_id + i, f0))
    clock.now += 0.6
    cal.update(voiced(start_id + 6, f0))


def test_noise_then_record_low_and_high():
    engine, clock = FakeEngine(), FakeClock()
    cal = Calibration(engine, clock=clock)
    cal.start()
    assert engine.calls == [1.5]
    assert cal.update(engine.state) == "noise"  # Stale success from an earlier calibration.
    assert cal.update({"frame_id": 11, "calibrating": True}) == "noise"
    assert cal.update({"frame_id": 12, "calibrating": False, "noise_calibrated": True}) == "low_ready"
    record(cal, clock, 12, 220.0)
    assert cal.phase == "high_ready"
    record(cal, clock, 20, 440.0)
    assert cal.phase == "done"
    assert cal.pitch_range.low == pytest.approx(-12)
    assert cal.pitch_range.high == pytest.approx(0)


def test_completed_noise_can_be_observed_after_the_active_frame():
    cal = Calibration(FakeEngine())
    cal.start()
    assert cal.update({"frame_id": 11, "noise_calibrated": True}) == "low_ready"


def test_retries_for_silence_duplicates_and_too_narrow_range():
    clock = FakeClock()
    cal = Calibration(FakeEngine(), clock=clock)
    cal.start()
    cal.update({"frame_id": 11, "noise_calibrated": True})
    cal.record_note({"frame_id": 12})
    for _ in range(10):
        clock.now += 0.2
        cal.update(voiced(12, 220))  # The frame predates recording.
    assert cal.phase == "low_ready"
    assert cal.low is None
    record(cal, clock, 20, 220)
    record(cal, clock, 30, 230)
    assert cal.phase == "high_ready"
    assert cal.pitch_range is None
    record(cal, clock, 40, 330)
    assert cal.phase == "done"


def test_unvoiced_or_low_confidence_frames_do_not_calibrate():
    clock = FakeClock()
    cal = Calibration(FakeEngine(), clock=clock)
    cal.start()
    cal.update({"frame_id": 11, "noise_calibrated": True})
    cal.record_note()
    for i in range(10):
        clock.now += 0.2
        cal.update(voiced(i, 220, gate=i % 2 == 0, confidence=0.1))
    assert cal.phase == "low_ready"


def test_pitch_range_maps_and_clamps_height():
    span = PitchRange(-12, 0)
    assert span.height(-20, 8, 156) == 156
    assert span.height(-12, 8, 156) == 156
    assert span.height(-6, 8, 156) == 82
    assert span.height(0, 8, 156) == 8
    assert span.height(5, 8, 156) == 8
    with pytest.raises(ValueError):
        PitchRange(0, 0)
    with pytest.raises(ValueError):
        span.height(math.nan, 8, 156)
    with pytest.raises(ValueError):
        span.height(0, 156, 8)
    assert default_range().low == pytest.approx(hz_to_semitone(100))
    assert default_range("whistle").low == pytest.approx(hz_to_semitone(600))