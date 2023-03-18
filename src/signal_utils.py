import matplotlib
import numpy as np

import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import savgol_filter

matplotlib.use("MacOSX")


def dominant_freq(y, fs):
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/fs)
    return freq[np.argmax(spec)]


def mean_freq(y, fs):
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/fs)
    amp = spec / spec.sum()
    return (freq * amp).sum()


def to_dominant_freq(timestamps, pressure, chunk_size=352):
    fs = 1/(timestamps[1] - timestamps[0])
    dominant_y = []
    for i in range(0, len(pressure), chunk_size):
        chunk = pressure[i:i+chunk_size]
        dominant_y.append(dominant_freq(chunk, fs))

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
    return savgol_filter(signal, 100, 4)


def debug_plot_marked(t, f, labels):
    x_in = []
    y_in = []
    x_out = []
    y_out = []

    for i in range(len(t)):
        if labels[i] == 'in':
            x_in.append(t[i])
            y_in.append(f[i])
        else:
            x_out.append(t[i])
            y_out.append(f[i])

    plt.scatter(x_in, y_in, s=5)
    plt.scatter(x_out, y_out, s=5)
    plt.legend(['in', 'out'])
    plt.show()