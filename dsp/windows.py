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
        "hann": hann_window,
        "hamming": hamming_window,
        "blackman": blackman_window
    }

    if name not in windows:
        raise ValueError("Unknown window type.")

    return windows[name](N)