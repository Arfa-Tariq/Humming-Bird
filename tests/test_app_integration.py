import time

import numpy as np
import pygame
import pytest

from app import game
from app.audio_io import PitchEngine
from app.calibration import PitchRange, default_range
from app.dashboard import Dashboard, handle_engine_key


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    pytest.fail("Timed out waiting for the audio engine")


def test_synthetic_engine_and_dashboard_during_live_reconfiguration(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    engine = PitchEngine(source="synthetic")
    try:
        engine.start()
        wait_for(lambda: engine.get_state().get("f0"))
        state = engine.get_state()
        assert state["gate_open"]
        assert np.isfinite(state["semitone"])
        assert len(engine.get_history()["t"]) > 0
        assert engine.get_spectrogram().shape[1] > 0
        dashboard = Dashboard()
        surface = pygame.Surface((960, 540))
        dashboard.draw(surface, engine, state, default_range(), True)
        for key in (pygame.K_1, pygame.K_w, pygame.K_b, pygame.K_RIGHTBRACKET,
                    pygame.K_n, pygame.K_m, pygame.K_2, pygame.K_3):
            handle_engine_key(engine, key)
            dashboard.draw(surface, engine, engine.get_state(), default_range(engine.mode), True)
            time.sleep(0.03)
        assert engine.method == "acf"
        assert engine.mode == "whistle"
        assert engine.window == "hamming"
        assert engine.bypass
        assert engine.N == 2048
        assert engine.noise_snr_db == 20
        wait_for(lambda: engine.get_state().get("mode") == "whistle"
                 and engine.get_state().get("N") == 2048)
        dashboard.draw(surface, engine, engine.get_state(), default_range("whistle"))
        assert engine.last_error is None
    finally:
        engine.stop()
        pygame.quit()


def test_pitch_game_movement_and_manual_fallback():
    game.reset_game()
    game.game_state = "game"
    game.flowers = [game.create_flower(1000, 120)]
    game.bees = [{"x": 1000, "y": 60, "start_y": 60, "direction": -1, "wing": 0}]
    pitch_range = PitchRange(-12, 0)
    game.update_game({"semitone": 0}, pitch_range, True)
    assert game.bird_y == 8
    game.update_game({"semitone": -12}, pitch_range, True)
    assert game.bird_y == 156
    game.update_game({"semitone": None}, pitch_range, True)
    assert game.bird_y == 156
    game.bird_velocity = game.FLAP_STRENGTH
    game.update_game({}, pitch_range, False)
    assert game.bird_y < 156


def test_pitch_landing_detects_crossing_a_flower():
    game.reset_game()
    game.game_state = "game"
    game.flowers = [game.create_flower(90, 120), game.create_flower(1000, 120)]
    game.bees = [{"x": 1000, "y": 60, "start_y": 60, "direction": -1, "wing": 0}]
    game.bird_y = 40
    game.update_game({"semitone": -12}, PitchRange(-12, 0), True)
    assert game.landing
    assert game.score == 1
    assert game.bird_y == 111


def test_failed_microphone_starts_synthetic_and_cleans_up(monkeypatch):
    instances = []

    class FakeEngine:
        def __init__(self, source, device=None):
            self.source, self.device, self.stopped = source, device, False
            instances.append(self)

        def start(self):
            if self.source == "mic":
                raise RuntimeError("no input device")

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(game, "PitchEngine", FakeEngine)
    engine, reason = game.start_audio("mic", 3)
    assert instances[0].stopped and instances[0].device == 3
    assert engine.source == "synthetic"
    assert "no input device" in reason


def test_wav_validation_and_unsigned_pcm_conversion(monkeypatch):
    from scipy.io import wavfile

    monkeypatch.setattr(wavfile, "read", lambda path: (16000, np.array([0, 128, 255], dtype=np.uint8)))
    engine = PitchEngine(source="sample.wav")
    block = next(engine._file_blocks("sample.wav"))
    np.testing.assert_allclose(block[:3], [-1, 0, 127 / 128])
    monkeypatch.setattr(wavfile, "read", lambda path: (16000, np.array([], dtype=np.int16)))
    with pytest.raises(ValueError, match="no samples"):
        engine.start()
    assert not engine._threads


def test_game_main_handles_dashboard_and_input_headlessly(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(game, "MUSIC_ENABLED", False)
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    frames = iter([
        [key(pygame.K_RETURN)],
        [key(pygame.K_d)],
        [key(pygame.K_1), key(pygame.K_w), key(pygame.K_RIGHTBRACKET)],
        [key(pygame.K_d)],
        [key(pygame.K_SPACE)],
        [key(pygame.K_p)],
        [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(frames, [pygame.event.Event(pygame.QUIT)]))
    game.main(["--source", "synthetic"])
    assert not pygame.get_init()


def test_rebuild_keeps_changes_requested_during_pipeline_setup(monkeypatch):
    from app import audio_io

    engine = PitchEngine(source="synthetic")
    original = audio_io.StreamingBandpass
    calls = 0

    def changing_filter(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            engine.set_mode("whistle")
            engine.set_frame_size(2048)
        return original(*args, **kwargs)

    monkeypatch.setattr(audio_io, "StreamingBandpass", changing_filter)
    engine._setup_pipeline()
    assert engine._rebuild
    engine._setup_pipeline()
    assert not engine._rebuild
    assert engine._pipeline_mode == "whistle"
    assert engine._N == 2048


def test_repeated_noise_calibration_resets_published_state():
    engine = PitchEngine(source="synthetic")
    engine._setup_pipeline()
    engine.calibrate_noise(0)
    engine._analyse(np.zeros(1024), np.zeros(1024), 1024, 256)
    assert engine.get_state()["noise_calibrated"]
    engine.calibrate_noise(1)
    state = engine.get_state()
    assert state["calibrating"] and not state["noise_calibrated"]
    assert not state["gate_open"]


def test_stalled_microphone_switches_to_synthetic(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(game, "MUSIC_ENABLED", False)
    instances = []

    class FakeEngine:
        def __init__(self, source, **kwargs):
            self.actual_source = source
            self.method = kwargs.get("method", "amdf")
            self.window = kwargs.get("window", "hann")
            self.mode = kwargs.get("mode", "hum")
            self.N = kwargs.get("N", 1024)
            self.bypass = False
            self.noise_snr_db = None
            self.last_error = None
            self.started = self.stopped = False
            instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def get_state(self):
            return {"running": True, "frame_id": 1, "source": self.actual_source}

        def set_bypass(self, value):
            self.bypass = value

        def set_noise_snr(self, value):
            self.noise_snr_db = value

    mic = FakeEngine("mic")
    monkeypatch.setattr(game, "PitchEngine", FakeEngine)
    monkeypatch.setattr(game, "start_audio", lambda source, device: (mic, None))
    ticks = iter(range(100))
    monkeypatch.setattr(game, "time", SimpleNamespace(perf_counter=lambda: float(next(ticks))))
    frames = iter([[] for _ in range(8)] + [[pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(frames, [pygame.event.Event(pygame.QUIT)]))
    game.main([])
    assert mic.stopped
    assert len(instances) == 2
    assert instances[1].actual_source == "synthetic"
    assert instances[1].started and instances[1].stopped


def test_game_runs_guided_microphone_calibration(monkeypatch):
    from app.calibration import Calibration
    from dsp.postproc import hz_to_semitone

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(game, "MUSIC_ENABLED", False)

    class FakeEngine:
        actual_source = "mic"
        mode = "hum"
        last_error = None
        frame = 0
        pitch = 220.0
        noise_frames = -1
        stopped = False

        def get_state(self):
            self.frame += 1
            active = self.noise_frames > 0
            if active:
                self.noise_frames -= 1
            return {"running": True, "frame_id": self.frame, "calibrating": active,
                    "noise_calibrated": self.noise_frames == 0, "gate_open": not active,
                    "f0": self.pitch, "confidence": 0.95,
                    "semitone": hz_to_semitone(self.pitch), "source": "mic"}

        def calibrate_noise(self, seconds):
            self.noise_frames = 2

        def stop(self):
            self.stopped = True

    class FastCalibration(Calibration):
        def __init__(self, engine):
            super().__init__(engine, note_seconds=0.005, min_frames=1)

    engine = FakeEngine()
    monkeypatch.setattr(game, "start_audio", lambda source, device: (engine, None))
    monkeypatch.setattr(game, "Calibration", FastCalibration)
    captured = []
    monkeypatch.setattr(game, "update_game", lambda state, pitch_range, control:
                        captured.append((pitch_range, control)))
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    frames = iter([
        [key(pygame.K_RETURN)], [], [], [],
        [key(pygame.K_RETURN)], [],
        [key(pygame.K_RETURN)], [], [],
        [pygame.event.Event(pygame.QUIT)],
    ])
    calls = 0

    def events():
        nonlocal calls
        calls += 1
        if calls >= 7:
            engine.pitch = 440.0
        return next(frames, [pygame.event.Event(pygame.QUIT)])

    monkeypatch.setattr(pygame.event, "get", events)
    game.main([])
    assert engine.stopped
    assert captured
    pitch_range, pitch_control = captured[-1]
    assert pitch_control
    assert pitch_range.low == pytest.approx(-12)
    assert pitch_range.high == pytest.approx(0)