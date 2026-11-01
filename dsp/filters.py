import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


# 1. Band-pass filter
def bandpass_filter(signal, fs, lowcut=70, highcut=1000, order=4):

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("Signal must contain at least 2 samples.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain only finite values.")

    nyquist = fs / 2

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if lowcut <= 0 or highcut <= lowcut:
        raise ValueError("Invalid cutoff frequencies.")

    if highcut >= nyquist:
        raise ValueError("High cutoff must be below Nyquist frequency.")

    if order < 1:
        raise ValueError("Filter order must be at least 1.")

    # Design Butterworth band-pass filter
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    # Apply filter
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


# 2. Decimator
def decimate_signal(signal, factor=2):

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain only finite values.")

    if not isinstance(factor, (int, np.integer)) or isinstance(factor, bool):
        raise ValueError("Decimation factor must be an integer.")

    if factor < 1:
        raise ValueError("Decimation factor must be at least 1.")

    if factor == 1:
        return signal.copy()

    # Reduce sampling rate with anti-aliasing filtering
    reduced_signal = resample_poly(signal, up=1, down=factor)

    return reduced_signal