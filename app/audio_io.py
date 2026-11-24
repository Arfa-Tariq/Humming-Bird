"""Live audio engine for Humming Bird.

    mic / file / synthetic  ->  ring buffer  ->  worker thread:
    (decimate) -> streaming band-pass -> frame (N, hop) -> noise gate
    -> estimator -> PitchTracker (octave guard, median, one-pole) -> state

The sounddevice callback ONLY copies the block into a ring buffer (a bounded
deque). All DSP runs in the worker thread. The game and dashboard just call
get_state() / get_history() / get_spectrogram(), which are lock-protected.

Quick check without pygame:
    python -m app.audio_io --source synthetic --seconds 8
    python -m app.audio_io --source mic --method amdf
    python -m app.audio_io --source data/hum.wav
    python -m app.audio_io --list-devices
"""
import argparse
import threading
import time
from collections import deque
from itertools import chain
from math import gcd

import numpy as np

from dsp.estimators import estimate_pitch
from dsp.filters import StreamingBandpass, StreamingDecimator
from dsp.noise_gate import NoiseGate
from dsp.postproc import PitchTracker, hz_to_note, semitone_to_hz
from dsp.windows import get_window

TARGET_FS = 16000
BLOCK = 256                     # samples per audio block at TARGET_FS

# band-pass corners (f_lo, f_hi), estimator search range (fmin, fmax),
# and the top of the spectrum panel, per mode
MODES = {
    "hum":     dict(f_lo=80.0,  f_hi=1000.0, fmin=70.0,  fmax=1000.0, disp=1500.0),
    "whistle": dict(f_lo=600.0, f_hi=3000.0, fmin=500.0, fmax=3000.0, disp=3000.0),
}


def list_input_devices():
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], int(d["default_samplerate"])))
    return out


class PitchEngine:
    def __init__(self, source="mic", device=None, method="amdf", window="hann",
                 mode="hum", N=1024, bp_order=2, bp_kind="butter",
                 median_len=5, smooth_a=0.18):
        self.source = source                # 'mic', 'synthetic' or a WAV path
        self.device = device
        self.method = method
        self.window = window
        self.mode = mode
        self.N = int(N)
        self.bypass = False
        self.noise_snr_db = None            # None = no injected noise
        self.bp_order = bp_order
        self.bp_kind = bp_kind

        self.fs = float(TARGET_FS)          # processing rate (set in start)
        self.gate = NoiseGate(initial_noise_floor=0.01)
        self.tracker = PitchTracker(median_len=median_len, smooth_a=smooth_a)

        self._queue = deque(maxlen=512)     # ring buffer of (timestamp, block)
        self._evt = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads = []
        self._stream = None
        self._dec = None
        self._rebuild = True
        self._rng = np.random.default_rng(0)

        self._state = {"running": False}
        self._hist = deque(maxlen=600)      # (t, f0|nan, smoothed_hz|nan, gate)
        self._spec_cols = deque(maxlen=240)  # one column per analysis hop
        self._frame_id = 0
        self._t_last = time.perf_counter()
        self._block_ms = 1000.0 * BLOCK / TARGET_FS
        self.last_error = None
        self.actual_source = source

        self._cal_active = False
        self._cal_until = 0.0
        self._cal_rms = []
        self.noise_calibrated = False

    # ------------------------------------------------------------------
    # settings (safe to call from the UI thread)
    # ------------------------------------------------------------------
    def set_method(self, method):
        self.method = method

    def set_window(self, window):
        get_window(window, 16)              # raises on a bad name
        self.window = window

    def set_bypass(self, flag):
        self.bypass = bool(flag)

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        with self._lock:
            self.mode = mode
            self._rebuild = True

    def set_frame_size(self, N):
        with self._lock:
            self.N = int(N)
            self._rebuild = True

    def set_noise_snr(self, snr_db):
        """None turns injection off. Noise is scaled to each block's own RMS, so
        the SNR is right while you hum and nothing is added to silence."""
        self.noise_snr_db = snr_db

    def calibrate_noise(self, seconds=1.5):
        """Stay quiet: measures the room and sets T_off = mean(RMS) + 3 std(RMS)."""
        with self._lock:
            self.noise_calibrated = False
            self._cal_rms = []
            self._cal_until = time.perf_counter() + seconds
            self._cal_active = True
            self._state = dict(self._state, calibrating=True, noise_calibrated=False,
                               gate_open=False)

    def estimated_latency_ms(self):
        """Latency budget formula from the guide (Section 12)."""
        bp = getattr(self, "_bp", None)
        bp_ms = bp.mean_passband_delay_ms() if bp is not None else 0.0
        hop = self.N // 4
        hop_s = hop / self.fs
        return (self._block_ms
                + 1000.0 * self.N / self.fs
                + 1000.0 * hop_s
                + bp_ms
                + 1000.0 * ((self.tracker.median.length - 1) / 2.0) * hop_s
                + 1000.0 * self.tracker.smoother.time_constant_frames() * hop_s
                + 16.0)

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------
    def start(self):
        if self._threads:
            return
        self._stop.clear()
        self._queue.clear()
        self.last_error = None

        feeder = None
        if self.source == "mic":
            fs_in, factor = self._open_mic()
        elif self.source == "synthetic":
            fs_in, factor, feeder = TARGET_FS, 1, self._synthetic_blocks()
        else:
            blocks = self._file_blocks(self.source)
            first_block = next(blocks)  # Open and validate the WAV before starting threads.
            fs_in, factor, feeder = TARGET_FS, 1, chain((first_block,), blocks)
        self.actual_source = self.source

        self._dec = StreamingDecimator(fs_in, factor) if factor > 1 else None
        self.fs = fs_in / factor
        self._block_ms = 1000.0 * BLOCK * factor / fs_in
        self._setup_pipeline()

        worker = threading.Thread(target=self._worker, daemon=True)
        self._threads.append(worker)
        worker.start()

        if feeder is not None:
            t = threading.Thread(target=self._feed, args=(feeder,), daemon=True)
            self._threads.append(t)
            t.start()
        else:
            self._stream.start()

    def stop(self):
        self._stop.set()
        self._evt.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        with self._lock:
            self._state = dict(self._state, running=False)

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        # NO DSP here: copy and return.
        self._queue.append((time.perf_counter(), indata[:, 0].copy()))
        self._evt.set()

    def _open_mic(self):
        import sounddevice as sd
        try:
            self._stream = sd.InputStream(
                device=self.device, channels=1, samplerate=TARGET_FS,
                blocksize=BLOCK, dtype="float32", callback=self._callback)
            return float(TARGET_FS), 1
        except Exception:
            pass
        try:
            info = sd.query_devices(self.device, "input")
            rate = int(info["default_samplerate"])
            factor = max(1, int(round(rate / TARGET_FS)))
            self._stream = sd.InputStream(
                device=self.device, channels=1, samplerate=rate,
                blocksize=BLOCK * factor, dtype="float32", callback=self._callback)
            return float(rate), factor
        except Exception as exc:
            raise RuntimeError(f"Could not open the microphone: {exc}")

    def _feed(self, blocks):
        """Push blocks into the ring buffer at real-time speed (file / synthetic)."""
        next_t = time.perf_counter()
        for block in blocks:
            if self._stop.is_set():
                return
            self._queue.append((time.perf_counter(), block))
            self._evt.set()
            next_t += len(block) / TARGET_FS
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def _synthetic_blocks(self):
        """Slow glide between about 100 and 400 Hz (harmonics 1/k), a short pause
        every cycle so the gate is visible. Works with no microphone at all."""
        phase, n = 0.0, 0
        period = 10.0
        while True:
            t = (n + np.arange(BLOCK)) / TARGET_FS
            center = 1200.0 if self.mode == "whistle" else 200.0
            f = center * 2.0 ** np.sin(2 * np.pi * t / period)
            ph = phase + np.cumsum(2 * np.pi * f / TARGET_FS)
            phase = float(ph[-1] % (2 * np.pi))
            x = sum(np.sin(k * ph) / k for k in range(1, 5)) * 0.15
            silent = (t % period) > (period - 1.0)
            x = np.where(silent, 0.0, x) + self._rng.normal(0, 0.002, BLOCK)
            n += BLOCK
            yield x.astype(np.float32)

    def _file_blocks(self, path):
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        rate, data = wavfile.read(path)
        x = data.astype(float)
        if np.issubdtype(data.dtype, np.unsignedinteger):
            midpoint = (float(np.iinfo(data.dtype).max) + 1.0) / 2.0
            x = (x - midpoint) / midpoint
        elif np.issubdtype(data.dtype, np.integer):
            x /= float(np.iinfo(data.dtype).max)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size == 0:
            raise ValueError("WAV file contains no samples.")
        if rate != TARGET_FS:
            g = gcd(int(rate), TARGET_FS)
            x = resample_poly(x, TARGET_FS // g, int(rate) // g)
        pad = (-len(x)) % BLOCK
        x = np.concatenate([x, np.zeros(pad)])
        while True:                           # loop the clip
            for i in range(0, len(x), BLOCK):
                yield x[i:i + BLOCK].astype(np.float32)

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------
    def _setup_pipeline(self):
        with self._lock:
            mode, N = self.mode, self.N
        cfg = MODES[mode]
        self._bp = StreamingBandpass(self.fs, cfg["f_lo"], cfg["f_hi"],
                                     order=self.bp_order, kind=self.bp_kind)
        self._fmin, self._fmax = cfg["fmin"], cfg["fmax"]
        self._N = int(N)
        self._pipeline_mode = mode
        self._hop = self._N // 4
        self._raw = np.zeros(0)
        self._filt = np.zeros(0)
        freqs = np.fft.rfftfreq(self._N, 1.0 / self.fs)
        self._disp_mask = freqs <= cfg["disp"]
        self._disp_freqs = freqs[self._disp_mask]
        self.tracker.reset()
        with self._lock:
            self._spec_cols.clear()
            # Keep a later UI change pending if it arrived during the rebuild.
            self._rebuild = self.mode != mode or self.N != N

    def _worker(self):
        while not self._stop.is_set():
            if self._rebuild:
                self._setup_pipeline()
            if not self._queue:
                self._evt.wait(0.05)
                self._evt.clear()
                continue
            t_cap, block = self._queue.popleft()
            try:
                self._process_block(t_cap, block)
            except Exception as exc:          # never let the thread die
                self.last_error = repr(exc)

    def _process_block(self, t_cap, block):
        x = np.asarray(block, dtype=float)
        if self._dec is not None:
            x = self._dec.process(x)
        if x.size == 0:
            return

        snr = self.noise_snr_db
        if snr is not None:
            rms = np.sqrt(np.mean(x ** 2))
            if rms > 1e-4:
                x = x + self._rng.normal(0.0, rms / 10 ** (snr / 20.0), x.size)

        y = self._bp.process(x)              # always run, keeps state warm
        self._raw = np.concatenate([self._raw, x])
        self._filt = np.concatenate([self._filt, y])
        self._t_last = t_cap

        N, hop = self._N, self._hop
        while len(self._filt) >= N:
            self._analyse(self._raw[:N], self._filt[:N], N, hop)
            self._raw = self._raw[hop:]
            self._filt = self._filt[hop:]
            if self._rebuild:
                return

    def _analyse(self, raw, filt, N, hop):
        frame = raw if self.bypass else filt
        rms = float(np.sqrt(np.mean(frame ** 2)))

        with self._lock:
            if self._cal_active:
                self._cal_rms.append(rms)
                if time.perf_counter() >= self._cal_until:
                    r = np.array(self._cal_rms)
                    self.gate.noise_floor = float(r.mean() + 3.0 * r.std())
                    self.gate.is_open = False
                    self._cal_active = False
                    self.noise_calibrated = True
            calibrating = self._cal_active

        self.gate.process(frame)
        gate_open = self.gate.is_open and not calibrating

        f0, info = None, None
        if gate_open:
            f0, info = estimate_pitch(frame, self.fs, self.method,
                                      self._fmin, self._fmax, window=self.window)
        s = self.tracker.update(f0, gate_open)
        smooth_hz = semitone_to_hz(s) if s is not None else float("nan")
        note, cents = hz_to_note(f0)

        # one spectrogram column per analysis hop (not per drawn frame)
        spec = np.abs(np.fft.rfft((frame - frame.mean()) * get_window(self.window, N)))
        spec_db = 20.0 * np.log10(spec[self._disp_mask] + 1e-9)

        now = time.perf_counter()
        self._frame_id += 1

        state = {
            "running": True,
            "frame_id": self._frame_id,
            "f0": f0,
            "confidence": float(info["confidence"]) if info else 0.0,
            "gate_open": gate_open,
            "calibrating": self._cal_active,
            "noise_calibrated": self.noise_calibrated,
            "semitone": s,                    # smoothed, relative to A4
            "smooth_hz": smooth_hz,
            "note": note,
            "cents": cents,
            "rms": rms,
            "noise_floor": self.gate.noise_floor,
            "method": self.method,
            "window": self.window,
            "mode": self._pipeline_mode,
            "bypass": self.bypass,
            "noise_snr_db": self.noise_snr_db,
            "N": N, "hop": hop, "fs": self.fs,
            "raw_wave": raw.copy(),
            "filt_wave": filt.copy(),
            "spec_freqs": self._disp_freqs,
            "spec_db": spec_db,
            "pipeline_ms": (now - self._t_last) * 1000.0,
            "est_latency_ms": self.estimated_latency_ms(),
            "source": self.actual_source,
            "error": self.last_error,
        }
        with self._lock:
            # Calibration may have been requested while this frame was analysed.
            state["calibrating"] = self._cal_active
            state["noise_calibrated"] = self.noise_calibrated
            if self._cal_active:
                state.update(gate_open=False, f0=None, confidence=0.0,
                             note=None, cents=float("nan"))
            self._spec_cols.append(spec_db.astype(np.float32))
            self._hist.append((now, np.nan if state["f0"] is None else state["f0"],
                               smooth_hz, state["gate_open"]))
            self._state = state

    # ------------------------------------------------------------------
    # read side (called by the game / dashboard)
    # ------------------------------------------------------------------
    def get_state(self):
        with self._lock:
            return dict(self._state)

    def get_history(self):
        """Arrays for the pitch-track panel: t (s, relative to now), f0 Hz (nan
        when rejected), smoothed Hz, gate state."""
        with self._lock:
            h = list(self._hist)
        if not h:
            e = np.zeros(0)
            return {"t": e, "f0": e, "smooth_hz": e, "gate": e.astype(bool)}
        a = np.array(h, dtype=float)
        return {"t": a[:, 0] - time.perf_counter(), "f0": a[:, 1],
                "smooth_hz": a[:, 2], "gate": a[:, 3].astype(bool)}

    def get_spectrogram(self):
        """2-D array (freq bins x time columns), dB. Oldest column first."""
        with self._lock:
            cols = list(self._spec_cols)
        if not cols:
            return np.zeros((1, 1))
        return np.stack(cols, axis=1)


# --------------------------------------------------------------------------
# command-line check
# --------------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    help="mic, synthetic, or path to a WAV file")
    ap.add_argument("--method", default="amdf", choices=["fft", "amdf", "acf"])
    ap.add_argument("--mode", default="hum", choices=list(MODES))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        for i, name, rate in list_input_devices():
            print(f"{i}: {name} ({rate} Hz)")
        return

    eng = PitchEngine(source=args.source, method=args.method, mode=args.mode)
    eng.start()
    print(f"source={args.source} fs_proc={eng.fs:.0f} Hz N={eng.N}")
    if args.source == "mic":
        print("Stay quiet for 1.5 s (measuring the room), then hum...")
        time.sleep(0.5)
        eng.calibrate_noise(1.5)
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        time.sleep(0.25)
        s = eng.get_state()
        if not s.get("running"):
            continue
        f0 = "  --  " if s["f0"] is None else f"{s['f0']:6.1f}"
        print(f"gate={'OPEN ' if s['gate_open'] else 'closed'} f0={f0} Hz "
              f"note={s['note'] or '--':>4} conf={s['confidence']:.2f} "
              f"rms={s['rms']:.4f} floor={s['noise_floor']:.4f} "
              f"pipe={s['pipeline_ms']:.1f} ms")
    eng.stop()
    print(f"estimated latency budget: {eng.estimated_latency_ms():.0f} ms")


if __name__ == "__main__":
    _main()