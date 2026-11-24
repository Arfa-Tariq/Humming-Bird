"""Live DSP dashboard displayed beside the game."""
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
    MUTED = (180, 196, 210)
    CYAN = (77, 218, 225)
    ORANGE = (255, 179, 85)
    GREEN = (135, 229, 159)

    def __init__(self):
        self.font = pygame.font.Font(None, 18)
        self.small = pygame.font.Font(None, 16)
        self.title = pygame.font.Font(None, 24)

    def _text(self, surface, text, pos, color=None, font=None, max_width=None):
        font = font or self.font
        text = str(text)
        if max_width is not None and font.size(text)[0] > max_width:
            while text and font.size(text + "...")[0] > max_width:
                text = text[:-1]
            text += "..."
        image = font.render(text, True, color or self.TEXT)
        surface.blit(image, pos)
        return image.get_size()

    def _right_text(self, surface, text, right, y, color=None):
        width = self.small.size(str(text))[0]
        self._text(surface, text, (right - width, y), color, self.small)

    def _panel(self, surface, rect, title, subtitle, caption):
        pygame.draw.rect(surface, self.PANEL, rect, border_radius=5)
        pygame.draw.rect(surface, self.GRID, rect, 1, border_radius=5)
        self._text(surface, title, (rect.x + 8, rect.y + 5), self.TEXT, self.font,
                   rect.w - 16)
        self._text(surface, subtitle, (rect.x + 8, rect.y + 23), self.MUTED, self.small,
                   rect.w - 16)
        plot = pygame.Rect(rect.x + 48, rect.y + 42, rect.w - 58, rect.h - 76)
        for fraction in (0.0, 0.5, 1.0):
            y = round(plot.top + fraction * (plot.h - 1))
            pygame.draw.line(surface, self.GRID, (plot.left, y), (plot.right - 1, y))
        self._text(surface, caption, (rect.x + 8, rect.bottom - 16), self.MUTED,
                   self.small, rect.w - 16)
        return plot

    def _axes(self, surface, rect, plot, high, low, left, right):
        self._text(surface, high, (rect.x + 4, plot.top), self.MUTED, self.small)
        self._text(surface, low, (rect.x + 4, plot.bottom - 13), self.MUTED, self.small)
        self._text(surface, left, (plot.left, plot.bottom + 2), self.MUTED, self.small)
        self._right_text(surface, right, plot.right, plot.bottom + 2, self.MUTED)

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

    @staticmethod
    def _max_freq(state):
        freqs = state.get("spec_freqs", [])
        if len(freqs):
            return float(freqs[-1])
        return MODES.get(state.get("mode"), MODES["hum"])["disp"]

    def _waveform(self, surface, rect, state):
        plot = self._panel(surface, rect, "1. WAVEFORM", "Cyan: input   |   Orange: filtered",
                           "Amplitude versus time")
        raw = np.asarray(state.get("raw_wave", []), dtype=float)
        filt = np.asarray(state.get("filt_wave", []), dtype=float)
        peak = max(0.02, float(np.max(np.abs(raw))) if raw.size else 0,
                   float(np.max(np.abs(filt))) if filt.size else 0)
        fs = state.get("fs") or 16000
        duration = max((max(raw.size, filt.size) - 1) / fs * 1000, 0.01)
        for wave, color in ((raw, self.CYAN), (filt, self.ORANGE)):
            if wave.size:
                self._curve(surface, plot, np.arange(wave.size) / fs * 1000, wave,
                            0, duration, -peak, peak, color)
        self._axes(surface, rect, plot, f"{peak:.2f}", f"-{peak:.2f}", "0 ms",
                   f"{duration:.0f} ms")

    def _spectrum(self, surface, rect, state):
        name = "input" if state.get("bypass") else "filtered"
        plot = self._panel(surface, rect, "2. SPECTRUM", f"{name.title()} FFT  |  White: estimated f0",
                           "Peaks: tones / harmonics; unnormalized dB")
        freqs = np.asarray(state.get("spec_freqs", []), dtype=float)
        db = np.asarray(state.get("spec_db", []), dtype=float)
        max_freq = self._max_freq(state)
        self._curve(surface, plot, freqs, db, 0, max_freq, -80, 40, self.CYAN)
        f0 = state.get("f0")
        if f0 is not None and math.isfinite(f0) and 0 <= f0 <= max_freq:
            x = plot.left + round((f0 / max_freq) * (plot.w - 1))
            for y in range(plot.top, plot.bottom, 6):
                pygame.draw.line(surface, self.TEXT, (x, y), (x, min(y + 2, plot.bottom - 1)))
        self._axes(surface, rect, plot, "+40 dB", "-80 dB", "0 Hz", f"{max_freq:.0f} Hz")

    def _spectrogram(self, surface, rect, state, engine):
        plot = self._panel(surface, rect, "3. SPECTROGRAM", "Frequency over time; brighter = stronger",
                           "Bands: harmonics  |  -80 to +40 dB")
        spec = np.asarray(engine.get_spectrogram(), dtype=float)
        freqs = state.get("spec_freqs", [])
        if spec.ndim == 2 and spec.size and len(freqs) == spec.shape[0]:
            level = np.clip((np.nan_to_num(spec, nan=-80) + 80) / 120, 0, 1)
            level = np.flipud(level).T  # pygame arrays use (x, y); low frequencies at bottom.
            rgb = np.empty((*level.shape, 3), dtype=np.uint8)
            rgb[..., 0] = np.clip((level - 0.35) * 2.5, 0, 1) * 255
            rgb[..., 1] = np.clip(level * 2, 0, 1) * 190
            rgb[..., 2] = np.clip(level * 3, 0, 1) * 200
            surface.blit(pygame.transform.scale(pygame.surfarray.make_surface(rgb), plot.size), plot)
            hop = state.get("hop") or 256
            fs = state.get("fs") or 16000
            seconds = max(0, spec.shape[1] - 1) * hop / fs
            left = f"-{seconds:.1f} s"
        else:
            left = "older"
        self._axes(surface, rect, plot, f"{self._max_freq(state):.0f}", "0 Hz", left, "now")

    def _pitch_track(self, surface, rect, state, engine):
        plot = self._panel(surface, rect, "4. PITCH TRACK", "Cyan: f0   |   Orange: smoothed",
                           "Orange holds when no pitch is detected")
        history = engine.get_history()
        t = np.asarray(history.get("t", []), dtype=float)
        mode = MODES.get(state.get("mode"), MODES["hum"])
        lo, hi = mode["fmin"], mode["fmax"]
        for key, color in (("f0", self.CYAN), ("smooth_hz", self.ORANGE)):
            hz = np.asarray(history.get(key, []), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(hz > 0, np.log2(hz), np.nan)
            self._curve(surface, plot, t, values, -8, 0, math.log2(lo), math.log2(hi), color)
        self._axes(surface, rect, plot, f"{hi:.0f}", f"{lo:.0f} Hz", "-8 s", "now")

    def draw(self, surface, engine, state, pitch_range, pitch_control=True,
             fallback_reason=None, game_state="game"):
        """Draw all plots and controls at the size of the supplied surface."""
        surface.fill(self.BACKGROUND)
        width, height = surface.get_size()
        self._text(surface, "LIVE PITCH LAB", (12, 9), self.CYAN, self.title)
        self._right_text(surface, "PLAYING" if game_state == "game" else game_state.upper(),
                         width - 12, 13, self.GREEN)

        def number(value, decimals=1):
            return f"{value:.{decimals}f}" if value is not None and math.isfinite(value) else "--"

        f0 = number(state.get("f0"))
        note = state.get("note") or "--"
        gate = "OPEN" if state.get("gate_open") else "CLOSED"
        self._text(surface, f"Pitch: {f0} Hz  |  Note: {note}  |  Gate: {gate}",
                   (12, 37), self.TEXT, self.font, width - 24)

        gap, top, footer = 8, 62, 190
        panel_width = (width - 3 * gap) // 2
        panel_height = (height - top - footer - 3 * gap) // 2
        rects = [pygame.Rect(gap + col * (panel_width + gap),
                             top + row * (panel_height + gap), panel_width, panel_height)
                 for row in range(2) for col in range(2)]
        self._waveform(surface, rects[0], state)
        self._spectrum(surface, rects[1], state)
        self._spectrogram(surface, rects[2], state, engine)
        self._pitch_track(surface, rects[3], state, engine)

        low = hz_to_note(semitone_to_hz(pitch_range.low))[0]
        high = hz_to_note(semitone_to_hz(pitch_range.high))[0]
        source = str(state.get("source", engine.actual_source))
        if source not in ("mic", "synthetic"):
            source = os.path.basename(source)
        if fallback_reason:
            source += " (mic unavailable)"
        snr = state.get("noise_snr_db", engine.noise_snr_db)
        fs = state.get("fs", engine.fs)
        n = state.get("N", engine.N)
        lines = [
            f"Input: {source}  |  Control: {'pitch' if pitch_control else 'manual'}  |  Range: {low} - {high}",
            f"Smoothed: {number(state.get('smooth_hz'))} Hz  |  Confidence: {number(state.get('confidence'), 2)} (method-specific)",
            f"Method: {engine.method.upper()}  |  Mode: {engine.mode}  |  Window: {engine.window}",
            f"Filter: {'bypass' if engine.bypass else 'on'}  |  Added noise: {'off' if snr is None else str(snr) + ' dB'}",
            f"Frame: {n}  |  Hop: {state.get('hop', n // 4)}  |  fs: {number(fs, 0)} Hz  |  FFT bins: {number(fs / n, 1)} Hz",
            f"Latency budget: {number(state.get('est_latency_ms'))} ms (estimated)  |  Block age: {number(state.get('pipeline_ms'))} ms",
            "Gate open means signal above threshold, not necessarily a valid pitch.",
            "FFT peaks can be harmonics; dB values depend on frame and window.",
            "1/2/3: FFT/AMDF/ACF   W: window   B: bypass   M: hum/whistle",
            "[ / ]: frame size   N: added noise   C: calibrate   SPACE/P: control",
        ]
        y = rects[-1].bottom + 10
        for index, line in enumerate(lines):
            color = self.CYAN if index >= 8 else (self.MUTED if index >= 6 else self.TEXT)
            self._text(surface, line, (12, y), color, self.small, width - 24)
            y += 16
        if state.get("error"):
            self._text(surface, f"Audio error: {state['error']}", (12, y), self.ORANGE,
                       self.small, width - 24)