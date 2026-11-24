import numpy as np


# Check that window size is valid
def validate_size(N):
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool):
        raise ValueError("Window size must be an integer.")

    if N <= 0:
        raise ValueError("Window size must be positive.")


# 1. Rectangular Window
def rectangular_window(N):
    validate_size(N)
    return np.ones(N)


# 2. Hann Window
def hann_window(N):
    validate_size(N)
    return np.hanning(N)


# 3. Hamming Window
def hamming_window(N):
    validate_size(N)
    return np.hamming(N)


# 4. Blackman Window
def blackman_window(N):
    validate_size(N)
    return np.blackman(N)


# 5. Select a window by name
def get_window(name, N):

    windows = {
        "rectangular": rectangular_window,
        "rect": rectangular_window,   # short alias used in the key map / guide
        "hann": hann_window,
        "hamming": hamming_window,
        "blackman": blackman_window
    }

    if name not in windows:
        raise ValueError("Unknown window type.")

    return windows[name](N)


# 6. Window quality numbers for E4 (main-lobe width and side-lobe level)
def window_metrics(name, N=1024, pad_factor=64):
    """Measure a window's spectrum shape.

    The window is zero-padded to N * pad_factor so the lobes are finely sampled.
    Widths are in units of ORIGINAL FFT bins (1 bin = fs / N), so they do not
    depend on fs.

    Returns a dict with
      main_lobe_width_bins : null-to-null width of the main lobe
      width_3db_bins       : full width where the magnitude is within 3 dB of peak
      peak_sidelobe_db     : highest side lobe relative to the main-lobe peak (dB, negative)
    """
    window = get_window(name, N)
    spec = np.abs(np.fft.rfft(window, N * pad_factor))
    spec_db = 20.0 * np.log10(spec / spec.max() + 1e-300)

    # first null = first local minimum going away from DC
    slope = np.diff(spec)
    minima = np.flatnonzero((slope[:-1] < 0) & (slope[1:] >= 0))
    if minima.size == 0:
        raise ValueError("Could not find the first null of the window spectrum.")
    first_null = int(minima[0]) + 1

    below_3db = np.flatnonzero(spec_db < -3.0103)
    half_3db = float(below_3db[0]) if below_3db.size else float("nan")

    return {
        "main_lobe_width_bins": 2.0 * first_null / pad_factor,
        "width_3db_bins": 2.0 * half_3db / pad_factor,
        "peak_sidelobe_db": float(spec_db[first_null:].max()),
    }