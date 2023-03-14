import matplotlib
import numpy as np

import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import savgol_filter

matplotlib.use("MacOSX")


def mean_freq(y, fs):
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/fs)
    plt.show()
    amp = spec / spec.sum()
    return (freq * amp).sum()


def to_dominant_freq(timestamps, pressure, chunk_size=128):
    fs = 1/(timestamps[1] - timestamps[0])
    dominant_y = []
    for i in range(0, len(pressure), chunk_size):
        chunk = pressure[i:i+chunk_size]
        dominant_y.append(mean_freq(chunk, fs))

    chunked_timestamps = np.arange(0, pressure.shape[0]/fs, chunk_size/fs)
    return chunked_timestamps, dominant_y


def debug_plot(x, y):
    plt.plot(x, y)
    plt.show()


def wav_to_sample_xy(filename):
    sample_rate, pressure = wavfile.read(filename)
    timestamps = np.arange(0, pressure.shape[0]/sample_rate, 1/sample_rate)
    return timestamps, pressure

def ema(array):
    N = len(array)
    alpha = 2/N
    numerator = sum([array[k] * (1-alpha)**k for k in range(N)])
    denominator = sum([(1-alpha)**k for k in range(N)])
    return numerator/denominator


def signal_clean(signal):
    return savgol_filter(signal, 51, 3)