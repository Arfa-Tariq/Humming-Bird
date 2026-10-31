import numpy as np


class NoiseGate:
    def __init__(
        self,
        initial_noise_floor=0.01,
        threshold_multiplier=2.5,
        hysteresis_ratio=0.7,
        adaptation_rate=0.05
    ):
        self.noise_floor = initial_noise_floor
        self.threshold_multiplier = threshold_multiplier
        self.hysteresis_ratio = hysteresis_ratio
        self.adaptation_rate = adaptation_rate

        self.is_open = False

    def process(self, signal):

        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 1:
            raise ValueError("Signal must be one-dimensional.")

        if len(signal) == 0:
            raise ValueError("Signal cannot be empty.")

        if not np.all(np.isfinite(signal)):
            raise ValueError("Signal must contain only finite values.")

        # Calculate the loudness of the audio frame
        rms = np.sqrt(np.mean(signal ** 2))

        # Calculate opening and closing thresholds
        open_threshold = self.noise_floor * self.threshold_multiplier
        close_threshold = open_threshold * self.hysteresis_ratio

        # Hysteresis: decide whether the gate should open or close
        if self.is_open:
            if rms < close_threshold:
                self.is_open = False
        else:
            if rms > open_threshold:
                self.is_open = True

        # Update noise floor only when the gate is closed
        if not self.is_open:
            self.noise_floor = (
                (1 - self.adaptation_rate) * self.noise_floor
                + self.adaptation_rate * rms
            )

        # Return audio if gate is open; otherwise return silence
        if self.is_open:
            return signal.copy()

        return np.zeros_like(signal)