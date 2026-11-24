import numpy as np
import pytest

from dsp.postproc import (
    hz_to_semitone,
    semitone_to_hz,
    hz_to_note,
    OctaveGuard,
    MedianFilter,
    OnePoleSmoother,
    PitchTracker,
)


# --------------------------------------------------------------------------
# Hz <-> semitones <-> note names
# --------------------------------------------------------------------------
def test_hz_to_semitone_reference_and_octave():
    assert hz_to_semitone(440.0) == pytest.approx(0.0)
    assert hz_to_semitone(880.0) == pytest.approx(12.0)
    assert hz_to_semitone(220.0) == pytest.approx(-12.0)


def test_hz_to_semitone_invalid_is_nan():
    assert np.isnan(hz_to_semitone(None))
    assert np.isnan(hz_to_semitone(0.0))
    assert np.isnan(hz_to_semitone(-10.0))


def test_hz_to_semitone_accepts_arrays():
    s = hz_to_semitone([220.0, 440.0, 880.0])
    np.testing.assert_allclose(s, [-12.0, 0.0, 12.0], atol=1e-9)


def test_semitone_roundtrip():
    for f in (82.4, 110.0, 261.63, 440.0, 987.77):
        assert semitone_to_hz(hz_to_semitone(f)) == pytest.approx(f)


def test_hz_to_note_a4_and_offset():
    name, cents = hz_to_note(440.0)
    assert name == "A4"
    assert cents == pytest.approx(0.0, abs=1e-6)

    name, cents = hz_to_note(445.0)
    assert name == "A4"
    assert cents == pytest.approx(19.6, abs=0.1)


def test_hz_to_note_other_notes():
    assert hz_to_note(261.6256)[0] == "C4"
    assert hz_to_note(110.0)[0] == "A2"


def test_hz_to_note_invalid():
    name, cents = hz_to_note(None)
    assert name is None and np.isnan(cents)
    assert hz_to_note(-5.0)[0] is None


# --------------------------------------------------------------------------
# Octave guard
# --------------------------------------------------------------------------
def test_octave_guard_first_value_passes():
    g = OctaveGuard()
    assert g.update(7.0) == 7.0


def test_octave_guard_small_moves_pass():
    g = OctaveGuard(max_jump=8.0)
    g.update(0.0)
    assert g.update(3.0) == 3.0
    assert g.update(-4.0) == -4.0           # a 7-semitone move is still allowed


def test_octave_guard_holds_then_accepts_real_jump():
    g = OctaveGuard(max_jump=8.0, max_hold=3)
    g.update(0.0)
    outs = [g.update(12.0) for _ in range(4)]
    assert outs == [0.0, 0.0, 0.0, 12.0]    # held 3 hops, then accepted


def test_octave_guard_isolated_spike_is_removed():
    g = OctaveGuard(max_jump=8.0, max_hold=3)
    g.update(0.0)
    assert g.update(12.0) == 0.0            # spike held
    assert g.update(0.0) == 0.0             # back to normal, counter reset


def test_octave_guard_reset():
    g = OctaveGuard()
    g.update(0.0)
    g.reset()
    assert g.update(20.0) == 20.0           # first value again, nothing to compare


def test_octave_guard_invalid_parameters():
    with pytest.raises(ValueError):
        OctaveGuard(max_jump=0)
    with pytest.raises(ValueError):
        OctaveGuard(max_hold=-1)


# --------------------------------------------------------------------------
# Median filter
# --------------------------------------------------------------------------
def test_median_removes_isolated_spike():
    m = MedianFilter(3)
    out = [m.update(x) for x in [0, 0, 10, 0, 0]]
    assert out == [0.0] * 5


def test_median_keeps_step_with_delay():
    m = MedianFilter(3)
    out = [m.update(x) for x in [0, 0, 0, 5, 5, 5]]
    assert out == [0.0, 0.0, 0.0, 0.0, 5.0, 5.0]
    assert m.delay_hops == 1


def test_median_length_one_is_passthrough():
    m = MedianFilter(1)
    for x in (3.0, -2.0, 7.5):
        assert m.update(x) == x


def test_median_delay_hops():
    assert MedianFilter(5).delay_hops == 2
    assert MedianFilter(9).delay_hops == 4


def test_median_reset():
    m = MedianFilter(3)
    for x in (1, 1, 1):
        m.update(x)
    m.reset()
    assert m.update(9.0) == 9.0


@pytest.mark.parametrize("bad", [0, 2, 4, -3])
def test_median_rejects_even_or_nonpositive_length(bad):
    with pytest.raises(ValueError):
        MedianFilter(bad)


# --------------------------------------------------------------------------
# One-pole smoother
# --------------------------------------------------------------------------
def test_one_pole_step_matches_formula():
    # step input from 0 to 1 must follow y[n] = 1 - 0.82^n  (guide, Section 13)
    s = OnePoleSmoother(a=0.18, initial=0.0)
    for n in range(1, 40):
        y = s.update(1.0)
        assert y == pytest.approx(1.0 - 0.82 ** n)


def test_one_pole_pole_location():
    assert OnePoleSmoother(a=0.18).pole == pytest.approx(0.82)
    assert OnePoleSmoother(a=0.4).pole == pytest.approx(0.6)


def test_one_pole_starts_at_first_input_by_default():
    s = OnePoleSmoother(a=0.18)
    assert s.update(5.0) == 5.0


def test_one_pole_a_equal_one_is_passthrough():
    s = OnePoleSmoother(a=1.0, initial=0.0)
    assert s.update(3.0) == pytest.approx(3.0)
    assert s.update(-1.0) == pytest.approx(-1.0)


def test_one_pole_time_constant():
    # about 5 updates for a = 0.18; y after that many updates is ~63.2 %
    s = OnePoleSmoother(a=0.18)
    tau = s.time_constant_frames()
    assert tau == pytest.approx(5.04, abs=0.05)
    assert 1.0 - 0.82 ** tau == pytest.approx(1.0 - np.exp(-1.0), abs=1e-9)


def test_one_pole_cutoff_is_minus_3db_and_about_1p9hz_at_60hz():
    s = OnePoleSmoother(a=0.18)
    fc = s.cutoff_hz(60.0)
    assert fc == pytest.approx(1.9, abs=0.1)

    # check directly that |H| at fc is 1/sqrt(2)
    w = 2 * np.pi * fc / 60.0
    h = s.a / (1.0 - s.pole * np.exp(-1j * w))
    assert 20 * np.log10(abs(h)) == pytest.approx(-3.0103, abs=0.01)


def test_one_pole_frequency_response_shape():
    freqs, mag_db = OnePoleSmoother(a=0.18).response(60.0)
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(30.0)
    assert mag_db[0] == pytest.approx(0.0, abs=1e-6)     # unity gain at DC
    assert np.all(np.diff(mag_db) <= 1e-9)               # low-pass: never rises


def test_one_pole_smaller_a_is_slower():
    slow = OnePoleSmoother(a=0.05, initial=0.0)
    fast = OnePoleSmoother(a=0.4, initial=0.0)
    for _ in range(5):
        ys, yf = slow.update(1.0), fast.update(1.0)
    assert ys < yf


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_one_pole_rejects_bad_a(bad):
    with pytest.raises(ValueError):
        OnePoleSmoother(a=bad)


# --------------------------------------------------------------------------
# PitchTracker
# --------------------------------------------------------------------------
def test_tracker_returns_none_until_first_voiced():
    t = PitchTracker()
    assert t.update(None, True) is None
    assert t.update(440.0, False) is None           # gate closed: ignored
    assert t.voiced is False
    assert t.update(440.0, True) == pytest.approx(0.0)
    assert t.voiced is True


def test_tracker_steady_pitch_gives_steady_output():
    t = PitchTracker()
    for _ in range(20):
        out = t.update(440.0, True)
    assert out == pytest.approx(0.0, abs=1e-9)


def test_tracker_converges_to_new_pitch():
    t = PitchTracker()
    for _ in range(10):
        t.update(440.0, True)
    target = 12 * np.log2(523.25 / 440.0)          # about 3 semitones up
    prev = 0.0
    for _ in range(80):
        out = t.update(523.25, True)
        assert out >= prev - 1e-12                  # moves one way, no overshoot
        prev = out
    assert out == pytest.approx(target, abs=0.05)


def test_tracker_holds_when_gate_closed_or_rejected():
    t = PitchTracker()
    for _ in range(30):
        t.update(440.0, True)
    held = t.output
    assert t.update(880.0, False) == pytest.approx(held, abs=1e-6)   # gate closed
    assert t.update(None, True) == pytest.approx(held, abs=1e-6)     # rejected frame
    assert t.voiced is False


def test_tracker_suppresses_single_octave_spike():
    t = PitchTracker()
    for _ in range(20):
        t.update(440.0, True)
    out = t.update(880.0, True)                    # one-hop octave error
    assert out == pytest.approx(0.0, abs=1e-6)
    for _ in range(5):
        out = t.update(440.0, True)
    assert out == pytest.approx(0.0, abs=1e-6)


def test_tracker_exposes_intermediate_values():
    t = PitchTracker()
    t.update(440.0, True)
    assert t.raw_semitone == pytest.approx(0.0)
    assert t.guarded_semitone == pytest.approx(0.0)
    assert t.median_semitone == pytest.approx(0.0)
    assert t.output == pytest.approx(0.0)


def test_tracker_reset():
    t = PitchTracker()
    t.update(440.0, True)
    t.reset()
    assert t.output is None
    assert t.update(None, True) is None