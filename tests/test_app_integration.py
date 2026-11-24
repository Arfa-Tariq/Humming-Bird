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


class DashboardEngineStub:
    method = "amdf"
    window = "hann"
    mode = "hum"
    N = 1024
    fs = 16000
    bypass = False
    noise_snr_db = None

    def get_history(self):
        return {}

    def get_spectrogram(self):
        return np.zeros((1, 1))


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


def empty_game():
    game.reset_game()
    game.game_state = "game"
    game.flowers = []
    game.walls = []


def advance(state, pitch_range, pitch_control, seconds, dt=1 / 60):
    for _ in range(round(seconds / dt)):
        game.update_game(state, pitch_range, pitch_control, dt)


def test_pitch_movement_is_smooth_and_manual_flap_uses_gravity():
    empty_game()
    pitch_range = PitchRange(-12, 0)
    game.update_game({"semitone": 0}, pitch_range, True)
    assert 98 < game.bird_y < 100  # A pitch change must not teleport the bird.
    advance({"semitone": 0}, pitch_range, True, 2)
    assert game.bird_y == pytest.approx(14, abs=0.2)
    game.update_game({"semitone": -12}, pitch_range, True)
    assert 14 < game.bird_y < 16
    advance({"semitone": -12}, pitch_range, True, 2)
    assert game.bird_y == pytest.approx(game.ground_y - game.bird_height - 12, abs=0.2)
    before = game.bird_y
    game.flap()
    game.update_game({}, pitch_range, False)
    assert game.bird_y < before
    assert game.bird_velocity > game.FLAP_STRENGTH


def test_pitch_dropout_grace_then_gravity_and_reacquisition():
    empty_game()
    pitch_range = PitchRange(-12, 0)
    game.bird_y = pitch_range.height(-6, 14, game.ground_y - game.bird_height - 12)
    advance({"semitone": -6, "gate_open": True}, pitch_range, True, 0.2)
    original = game.bird_y
    advance({"semitone": -6, "gate_open": False}, pitch_range, True, 0.2)
    assert game.bird_y == pytest.approx(original, abs=0.01)
    advance({"semitone": -6, "gate_open": False}, pitch_range, True, 0.3)
    assert game.bird_y > original + 5
    falling = game.bird_y
    advance({"semitone": 0, "gate_open": True}, pitch_range, True, 0.6)
    assert game.bird_y < falling


def test_landing_stays_safe_until_voice_takeoff_and_scores_once():
    empty_game()
    game.flowers = [game.create_flower(90, 120)]
    game.bird_y = 103
    game.bird_velocity = 80
    pitch_range = PitchRange(-12, 0)
    advance({"semitone": -12}, pitch_range, True, 0.1)
    assert game.landing
    assert game.score == 1
    assert game.bird_y == 120 - 5 - game.bird_height
    stopped = game.bird_y, game.camera_x
    advance({"semitone": -12}, pitch_range, True, 2)
    advance({"semitone": None, "gate_open": False}, pitch_range, True, 2)
    assert game.landing and (game.bird_y, game.camera_x) == stopped
    advance({"semitone": 0, "gate_open": True}, pitch_range, True, 0.1)
    assert not game.landing and game.bird_y < stopped[0]
    # Landing on the same flower again must not farm points.
    game.bird_y = stopped[0]
    game.bird_velocity = 60
    game.update_game({}, pitch_range, False)
    assert game.landing and game.score == 1
    game.flap()
    game.update_game({}, pitch_range, False)
    assert not game.landing and game.bird_velocity < 0


def test_game_physics_follow_elapsed_time():
    heights = []
    for dt in (1 / 30, 1 / 60, 1 / 120):
        empty_game()
        advance({"semitone": 0}, PitchRange(-12, 0), True, 0.5, dt)
        assert game.camera_x == pytest.approx(22.5, abs=0.01)
        heights.append(game.bird_y)
    assert max(heights) - min(heights) < 1.0
    assert 40 < heights[0] < 60


@pytest.mark.parametrize("y, collision", [(70, False), (30, True), (120, True)])
def test_wall_gap_collision_matches_flight_path(y, collision):
    empty_game()
    game.bird_y = y
    game.walls = [{"x": game.bird_x + 5, "gap_top": 50, "passed": False}]
    game.update_game({}, None, False)
    assert (game.game_state == "game_over") is collision


def test_wall_is_scored_once_after_clearing_gap():
    empty_game()
    game.bird_y = 70
    wall = {"x": game.bird_x - game.WALL_WIDTH - 1, "gap_top": 50, "passed": False}
    game.walls = [wall]
    advance({}, None, False, 0.1)
    assert wall["passed"] and game.gaps_passed == 1
    advance({}, None, False, 0.1)
    assert game.gaps_passed == 1


def test_game_clears_a_wall_while_holding_a_pitch():
    empty_game()
    pitch_range = PitchRange(-12, 0)
    game.bird_y = 72
    wall = {"x": game.bird_x + 35, "gap_top": 50, "passed": False}
    game.walls = [wall]
    fraction = (game.ground_y - game.bird_height - 12 - game.bird_y) / (
        game.ground_y - game.bird_height - 12 - 14)
    semitone = pitch_range.low + fraction * (pitch_range.high - pitch_range.low)
    advance({"semitone": semitone, "gate_open": True}, pitch_range, True, 1.5)
    assert game.game_state == "game" and wall["passed"] and game.gaps_passed == 1


def test_generated_flowers_are_between_walls():
    game.reset_game()
    game.game_state = "game"
    game.update_game({"semitone": -6}, PitchRange(-12, 0), True)
    assert [wall["x"] for wall in game.walls[:2]] == [400, 590]
    assert any(flower["x"] == 495 for flower in game.flowers)
    assert any(flower["x"] == 685 for flower in game.flowers)
    assert all(0 < wall["gap_top"] < wall["gap_top"] + game.WALL_GAP < game.ground_y
               for wall in game.walls)


def test_synthetic_audio_drives_smooth_flight():
    empty_game()
    engine = PitchEngine(source="synthetic")
    try:
        engine.start()
        wait_for(lambda: engine.get_state().get("f0"))
        positions = [game.bird_y]
        for _ in range(60):
            game.update_game(engine.get_state(), default_range(), True)
            positions.append(game.bird_y)
            time.sleep(1 / 60)
        assert positions[-1] < positions[0] - 5
        assert max(abs(b - a) for a, b in zip(positions, positions[1:])) <= (
            game.MAX_FALL_SPEED / 60 + 1e-6)
        assert game.game_state == "game" and engine.last_error is None
    finally:
        engine.stop()


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


@pytest.mark.parametrize("size", [(960, 600), (1280, 720), (1440, 800)])
def test_side_by_side_layout_and_mouse_mapping(size, monkeypatch):
    game_area, dashboard_area = game.layout_for_window(size)
    assert not game_area.colliderect(dashboard_area)
    assert pygame.Rect((0, 0), size).contains(game_area)
    assert pygame.Rect((0, 0), size).contains(dashboard_area)
    monkeypatch.setattr(game, "game_rect", game_area)
    center = game.start_button.center
    mouse_pos = (game_area.left + (center[0] * game_area.width // game.GAME_WIDTH),
                 game_area.top + (center[1] * game_area.height // game.GAME_HEIGHT))
    assert game.start_button.collidepoint(game.screen_to_game(mouse_pos))
    assert game.screen_to_game(dashboard_area.center) is None


def test_dashboard_explanations_fit_minimum_window(monkeypatch):
    pygame.font.init()
    engine = DashboardEngineStub()
    engine.actual_source = "synthetic"
    dashboard = Dashboard()
    captions = []
    original = dashboard._text

    def capture(surface, text, pos, color=None, font=None, max_width=None):
        if text in ("Amplitude versus time", "Peaks: tones / harmonics; unnormalized dB",
                    "Bands: harmonics  |  -80 to +40 dB",
                    "Orange holds when no pitch is detected"):
            captions.append(text)
            assert (font or dashboard.font).size(text)[0] <= max_width
        return original(surface, text, pos, color, font, max_width)

    monkeypatch.setattr(dashboard, "_text", capture)
    _, dashboard_area = game.layout_for_window((960, 600))
    dashboard.draw(pygame.Surface(dashboard_area.size), engine, {}, default_range())
    assert len(captions) == 4


def test_spectrogram_frequency_orientation_and_color_strength():
    pygame.font.init()
    dashboard = Dashboard()
    surface = pygame.Surface((320, 180))
    engine = DashboardEngineStub()
    engine.get_spectrogram = lambda: np.array([[40, -80], [-80, -80], [-80, 40]])
    dashboard._spectrogram(surface, surface.get_rect(),
                           {"spec_freqs": [0, 500, 1000], "fs": 16000, "hop": 256}, engine)
    # The older low-frequency bin and newer high-frequency bin are bright.
    assert sum(surface.get_at((70, 130))[:3]) > 500
    assert sum(surface.get_at((280, 55))[:3]) > 500
    assert sum(surface.get_at((70, 55))[:3]) < 20
    assert sum(surface.get_at((280, 130))[:3]) < 20


def test_game_and_dashboard_update_together_while_playing(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr(game, "MUSIC_ENABLED", False)
    updates, draws = [], []
    original_update, original_draw = game.update_game, Dashboard.draw

    def track_update(state, pitch_range, pitch_control, dt):
        updates.append((game.game_state, pitch_control))
        original_update(state, pitch_range, pitch_control, dt)

    def track_draw(self, surface, engine, state, *args):
        draws.append((game.game_state, surface.get_size(), state.get("frame_id")))
        original_draw(self, surface, engine, state, *args)

    monkeypatch.setattr(game, "update_game", track_update)
    monkeypatch.setattr(Dashboard, "draw", track_draw)
    calls = 0

    def events():
        nonlocal calls
        calls += 1
        if calls == 1:
            r = game.game_rect
            x, y = game.start_button.center
            pos = (r.left + x * r.width // game.GAME_WIDTH,
                   r.top + y * r.height // game.GAME_HEIGHT)
            return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)]
        if calls == 2:
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)]
        if calls == 3:
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)]
        if calls == 4:
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)]
        if calls == 6:
            return [pygame.event.Event(pygame.VIDEORESIZE, w=1280, h=720, size=(1280, 720))]
        if calls == 20:
            return [pygame.event.Event(pygame.QUIT)]
        return []

    monkeypatch.setattr(pygame.event, "get", events)
    game.main(["--source", "synthetic"])
    assert len(updates) >= 19 and all(phase == "game" for phase, _ in updates)
    assert any(not pitch for _, pitch in updates) and updates[-1][1]
    assert len(draws) >= 2 and all(phase == "game" for phase, _, _ in draws)
    assert len({size for _, size, _ in draws}) == 2
    assert len({frame for _, _, frame in draws if frame is not None}) >= 2


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

    class FakeEngine(DashboardEngineStub):
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

    class FakeEngine(DashboardEngineStub):
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
    monkeypatch.setattr(game, "update_game", lambda state, pitch_range, control, dt:
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