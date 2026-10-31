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


# 4. Generate a chirp (frequency continuously increases)
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