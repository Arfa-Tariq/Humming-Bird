import numpy as np


# 1. Generate a simple sine wave
def pure_tone(fs, duration, seed=0, frequency=440):
    t = np.arange(int(fs * duration)) / fs
    signal = 0.8 * np.sin(2 * np.pi * frequency * t)
    return signal


# 2. Generate a hum containing multiple harmonics
def harmonic_hum(fs, duration, seed=0, f0=220):
    t = np.arange(int(fs * duration)) / fs
    signal = np.zeros(len(t))

    for k in range(1, 6):
        amplitude = 1 / k
        signal += amplitude * np.sin(2 * np.pi * k * f0 * t)

    signal = signal / np.max(np.abs(signal))
    return signal


# 3. Generate a hum with a weak fundamental frequency
def weak_fundamental(fs, duration, seed=0, f0=220):
    t = np.arange(int(fs * duration)) / fs

    fundamental = 0.2 * np.sin(2 * np.pi * f0 * t)
    second_harmonic = 0.8 * np.sin(2 * np.pi * 2 * f0 * t)

    signal = fundamental + second_harmonic
    signal = signal / np.max(np.abs(signal))

    return signal


# 4. Generate a chirp (frequency continuously increases, linear in Hz)
def chirp_signal(fs, duration, seed=0,
                 start_freq=100, end_freq=1000):

    t = np.arange(int(fs * duration)) / fs

    k = (end_freq - start_freq) / duration

    phase = 2 * np.pi * (
        start_freq * t + 0.5 * k * t**2
    )

    signal = np.sin(phase)

    return signal


# 5. Add noise to an existing signal
def add_noise(signal, snr_db, seed=0):

    rng = np.random.default_rng(seed)

    signal_power = np.mean(signal ** 2)

    noise = rng.normal(0, 1, len(signal))
    noise = noise - np.mean(noise)

    noise_power = np.mean(noise ** 2)

    target_noise_power = signal_power / (10 ** (snr_db / 10))

    noise = noise * np.sqrt(
        target_noise_power / noise_power
    )

    return signal + noise


# 6. Generate background noise during silence
def silence_noise(fs, duration, seed=0, noise_rms=0.01):

    rng = np.random.default_rng(seed)

    samples = int(fs * duration)

    noise = rng.normal(0, noise_rms, samples)

    return noise


# 7. Exponential (constant octaves-per-second) chirp for E7
def exp_chirp(fs, octaves_per_sec, seed=0, f_start=100.0, f_end=1000.0, amplitude=0.8):
    """Sweep from f_start to f_end at a constant rate in octaves per second.

    Pitch is logarithmic, so a sweep rate in octaves/s is the natural unit for
    "how fast is the glide". The duration follows from the rate:
    duration = log2(f_end / f_start) / octaves_per_sec.

    Returns (signal, f_true) where f_true[i] is the exact instantaneous
    frequency in Hz at sample i (the ground truth for tracking error).
    """
    if octaves_per_sec <= 0:
        raise ValueError("octaves_per_sec must be positive.")
    if not (0 < f_start < f_end):
        raise ValueError("Need 0 < f_start < f_end.")

    duration = np.log2(f_end / f_start) / octaves_per_sec
    t = np.arange(int(round(fs * duration))) / fs

    r = octaves_per_sec * np.log(2.0)             # growth rate in nepers/s
    f_true = f_start * np.exp(r * t)
    phase = 2 * np.pi * f_start * (np.exp(r * t) - 1.0) / r

    return amplitude * np.sin(phase), f_true


# 8. Stepped pitch track (in semitones) for the smoother lag measurement in E6
def step_track(rate_hz, duration, step_time, s_from=0.0, s_to=5.0):
    """Pitch track sampled at rate_hz (one value per analysis hop), in semitones.
    Holds s_from until step_time seconds, then jumps to s_to."""
    n = int(round(rate_hz * duration))
    t = np.arange(n) / rate_hz
    return np.where(t < step_time, s_from, s_to).astype(float)


# 9. Noisy pitch track (in semitones) for the jitter measurement in E6
def noisy_track(rate_hz, duration, seed=0, s0=0.0, jitter_cents=30.0,
                outlier_prob=0.0, outlier_semitones=12.0):
    """Steady pitch s0 plus Gaussian jitter (std in cents).

    outlier_prob > 0 adds isolated +-outlier_semitones spikes (like octave
    errors) so the median filter has something to remove."""
    rng = np.random.default_rng(seed)
    n = int(round(rate_hz * duration))
    track = s0 + rng.normal(0.0, jitter_cents / 100.0, n)

    if outlier_prob > 0:
        spikes = rng.random(n) < outlier_prob
        signs = rng.choice([-1.0, 1.0], size=n)
        track = track + spikes * signs * outlier_semitones

    return track