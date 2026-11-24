"""Pygame dashboard for the live pitch engine, displayed with D in the game."""
import math
import os

import numpy as np
import pygame

from app.audio_io import MODES
from dsp.postproc import hz_to_note, semitone_to_hz


METHODS = ("fft", "amdf", "acf")
WINDOWS = ("hann", "hamming", "blackman", "rectangular")
FRAME_SIZES = (512, 1024, 2048)
NOISE_LEVELS = (None, 20.0, 10.0, 0.0, -5.0)


def _next(values, current, step=1):
    try:
        return values[(values.index(current) + step) % len(values)]
    except ValueError:
        return values[0]


def handle_engine_key(engine, key):
    """Apply a dashboard control and return 'mode' if calibration is invalidated."""
    if key in (pygame.K_1, pygame.K_2, pygame.K_3):
        engine.set_method(METHODS[key - pygame.K_1])
    elif key == pygame.K_w:
        engine.set_window(_next(WINDOWS, engine.window))
    elif key == pygame.K_b:
        engine.set_bypass(not engine.bypass)
    elif key == pygame.K_m:
        engine.set_mode("whistle" if engine.mode == "hum" else "hum")
        return "mode"
    elif key == pygame.K_LEFTBRACKET:
        engine.set_frame_size(_next(FRAME_SIZES, engine.N, -1))
    elif key == pygame.K_RIGHTBRACKET:
        engine.set_frame_size(_next(FRAME_SIZES, engine.N))
    elif key == pygame.K_n:
        engine.set_noise_snr(_next(NOISE_LEVELS, engine.noise_snr_db))
    return None


class Dashboard:
    BACKGROUND = (14, 22, 34)
    PANEL = (25, 38, 53)
    GRID = (52, 69, 82)
    TEXT = (225, 235, 239)
    MUTED = (156, 177, 187)
    CYAN = (77, 218, 225)
    ORANGE = (255, 179, 85)
    GREEN = (135, 229, 159)

    def __init__(self):
        self.font = pygame.font.Font(None, 19)
        self.small = pygame.font.Font(None, 16)
        self.title = pygame.font.Font(None, 24)

    def _text(self, surface, text, pos, color=None, font=None):
        image = (font or self.font).render(str(text), True, color or self.TEXT)
        surface.blit(image, pos)

    def _panel(self, surface, rect, title):
        pygame.draw.rect(surface, self.PANEL, rect, border_radius=5)
        pygame.draw.rect(surface, self.GRID, rect, 1, border_radius=5)
        self._text(surface, title, (rect.x + 8, rect.y + 5), self.TEXT, self.small)
        plot = pygame.Rect(rect.x + 41, rect.y + 25, rect.w - 52, rect.h - 45)
        for fraction in (0.0, 0.5, 1.0):
            y = round(plot.top + fraction * (plot.h - 1))
            pygame.draw.line(surface, self.GRID, (plot.left, y), (plot.right, y))
        return plot

    @staticmethod
    def _curve(surface, rect, x, y, xmin, xmax, ymin, ymax, color):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or x.size != y.size or xmax <= xmin or ymax <= ymin:
            return
        valid = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
        px = rect.left + np.clip((x - xmin) / (xmax - xmin), 0, 1) * (rect.w - 1)
        py = rect.bottom - 1 - np.clip((y - ymin) / (ymax - ymin), 0, 1) * (rect.h - 1)
        points = []
        for xx, yy, ok in zip(px, py, valid):
            if ok:
                points.append((int(xx), int(yy)))
            else:
                if len(points) > 1:
                    pygame.draw.lines(surface, color, False, points, 1)
                points = []
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 1)

    def _waveform(self, surface, rect, state):
        plot = self._panel(surface, rect, "WAVEFORM   raw (cyan) / filtered (orange)")
        raw = np.asarray(state.get("raw_wave", []), dtype=float)
        filt = np.asarray(state.get("filt_wave", []), dtype=float)
        peak = max(0.02, float(np.max(np.abs(raw))) if raw.size else 0,
                   float(np.max(np.abs(filt))) if filt.size else 0)
        self._text(surface, f"{peak:.2f}", (rect.x + 4, plot.top), self.MUTED, self.small)
        self._text(surface, f"-{peak:.2f}", (rect.x + 2, plot.bottom - 12), self.MUTED, self.small)
        fs = state.get("fs", 16000)
        n = max(raw.size, filt.size)
        duration = max((n - 1) / fs * 1000, 0.01)
        for wave, color in ((raw, self.CYAN), (filt, self.ORANGE)):
            if wave.size:
                self._curve(surface, plot, np.arange(wave.size) / fs * 1000, wave,
                            0, duration, -peak, peak, color)
        self._text(surface, f"0                         {duration:.0f} ms",
                   (plot.left, plot.bottom + 3), self.MUTED, self.small)

    def _spectrum(self, surface, rect, state):
        name = "raw" if state.get("bypass") else "filtered"
        plot = self._panel(surface, rect, f"SPECTRUM  ({name}, dB)")
        freqs = np.asarray(state.get("spec_freqs", []), dtype=float)
        db = np.asarray(state.get("spec_db", []), dtype=float)
        max_freq = float(freqs[-1]) if freqs.size else MODES[state.get("mode", "hum")]["disp"]
        self._curve(surface, plot, freqs, db, 0, max_freq, -80, 40, self.CYAN)
        self._text(surface, "40", (rect.x + 10, plot.top), self.MUTED, self.small)
        self._text(surface, "-80", (rect.x + 7, plot.bottom - 12), self.MUTED, self.small)
        self._text(surface, f"0                                      {max_freq:.0f} Hz",
                   (plot.left, plot.bottom + 3), self.MUTED, self.small)

    def _spectrogram(self, surface, rect, state, engine):
        plot = self._panel(surface, rect, "SPECTROGRAM  (one column / hop, dB)")
        spec = np.asarray(engine.get_spectrogram(), dtype=float)
        if spec.ndim == 2 and spec.size:
            level = np.clip((np.nan_to_num(spec, nan=-80) + 80) / 120, 0, 1)
            # pygame's surfarray is indexed (x, y); flip frequency so low is at the bottom.
            level = np.flipud(level).T
            rgb = np.empty((*level.shape, 3), dtype=np.uint8)
            rgb[..., 0] = np.clip((level - 0.35) * 2.5, 0, 1) * 255
            rgb[..., 1] = np.clip(level * 2, 0, 1) * 190
            rgb[..., 2] = np.clip(level * 3, 0, 1) * 200
            image = pygame.surfarray.make_surface(rgb)
            surface.blit(pygame.transform.scale(image, plot.size), plot)
        freqs = state.get("spec_freqs", [])
        max_freq = float(freqs[-1]) if len(freqs) else MODES[state.get("mode", "hum")]["disp"]
        self._text(surface, f"{max_freq:.0f}", (rect.x + 2, plot.top), self.MUTED, self.small)
        self._text(surface, "0 Hz", (rect.x + 3, plot.bottom - 12), self.MUTED, self.small)
        self._text(surface, "older                                  now",
                   (plot.left, plot.bottom + 3), self.MUTED, self.small)

    def _pitch_track(self, surface, rect, state, engine):
        plot = self._panel(surface, rect, "PITCH TRACK   raw (cyan) / smoothed (orange)")
        history = engine.get_history()
        t = np.asarray(history.get("t", []), dtype=float)
        mode = MODES[state.get("mode", "hum")]
        lo, hi = mode["fmin"], mode["fmax"]
        for key, color in (("f0", self.CYAN), ("smooth_hz", self.ORANGE)):
            hz = np.asarray(history.get(key, []), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(hz > 0, np.log2(hz), np.nan)
            self._curve(surface, plot, t, values, -8, 0, math.log2(lo), math.log2(hi), color)
        self._text(surface, f"{hi:.0f}", (rect.x + 2, plot.top), self.MUTED, self.small)
        self._text(surface, f"{lo:.0f} Hz", (rect.x + 2, plot.bottom - 12), self.MUTED, self.small)
        self._text(surface, "-8 s                                      now",
                   (plot.left, plot.bottom + 3), self.MUTED, self.small)

    def draw(self, surface, engine, state, pitch_range, pitch_control=True,
             fallback_reason=None, game_state="game"):
        surface.fill(self.BACKGROUND)
        self._text(surface, "HUMMING BIRD  /  LIVE DSP", (12, 8), self.CYAN, self.title)
        self._text(surface, f"{game_state.upper()} PAUSED  -  D or ESC to return",
                   (580, 11), self.MUTED, self.small)
        self._waveform(surface, pygame.Rect(12, 38, 460, 148), state)
        self._spectrum(surface, pygame.Rect(488, 38, 460, 148), state)
        self._spectrogram(surface, pygame.Rect(12, 194, 460, 148), state, engine)
        self._pitch_track(surface, pygame.Rect(488, 194, 460, 148), state, engine)

        def number(value, decimals=1):
            return f"{value:.{decimals}f}" if value is not None and math.isfinite(value) else "--"

        f0 = number(state.get("f0"))
        cents = number(state.get("cents"), 0)
        low = hz_to_note(semitone_to_hz(pitch_range.low))[0]
        high = hz_to_note(semitone_to_hz(pitch_range.high))[0]
        source = str(state.get("source", engine.actual_source))
        if source not in ("mic", "synthetic"):
            source = os.path.basename(source)
        if fallback_reason:
            source += " (mic unavailable)"
        snr = state.get("noise_snr_db", engine.noise_snr_db)
        lines = [
            f"Source: {source}   |   {'PITCH' if pitch_control else 'MANUAL'}   |   Range: {low} - {high}",
            f"f0: {f0} Hz   Note: {state.get('note') or '--'}  {cents} cents   Confidence: {number(state.get('confidence'), 2)}   Gate: {'OPEN' if state.get('gate_open') else 'CLOSED'}",
            f"Method: {engine.method.upper()}   Window: {engine.window}   Mode: {engine.mode}   Filter: {'bypass' if engine.bypass else 'on'}   Noise: {'off' if snr is None else str(snr) + ' dB'}",
            f"N: {state.get('N', engine.N)}   Hop: {state.get('hop', engine.N // 4)}   fs: {number(state.get('fs', engine.fs), 0)} Hz   RMS: {number(state.get('rms'), 4)}   Floor: {number(state.get('noise_floor'), 4)}",
            f"Smoothed: {number(state.get('semitone'), 2)} st / {number(state.get('smooth_hz'))} Hz   Pipeline: {number(state.get('pipeline_ms'))} ms   Estimated latency: {number(state.get('est_latency_ms'))} ms",
        ]
        for i, line in enumerate(lines):
            self._text(surface, line, (14, 351 + 17 * i), self.TEXT, self.small)
        self._text(surface, "KEY MAP", (14, 443), self.CYAN, self.small)
        self._text(surface, "1/2/3: FFT / AMDF / ACF     W: window     B: filter bypass     M: hum / whistle",
                   (14, 461), self.TEXT, self.small)
        self._text(surface, "[ / ]: frame size     N: injected noise SNR     C: calibrate mic     D: dashboard",
                   (14, 478), self.TEXT, self.small)
        self._text(surface, "SPACE: manual flap     P: pitch control     ENTER: start / record     ESC: back",
                   (14, 495), self.TEXT, self.small)
        if state.get("error"):
            self._text(surface, f"Audio error: {state['error']}"[:120], (14, 515), self.ORANGE, self.small)