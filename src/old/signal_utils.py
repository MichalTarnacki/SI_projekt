import matplotlib
import numpy as np

import matplotlib.pyplot as plt

import src.data_engineering.data_utils as du

from scipy.signal import savgol_filter

try:
    matplotlib.use("MacOSX")
except:
    pass


def file_with_labels(file_csv, file_wav):
    x, y = du.wav_to_sample_xy(file_wav)
    t, f = to_dominant_freq(1/(x[1]-x[0]), y)

    return buffer_get_labels(file_csv, t), t, signal_clean(f)


def dominant_freq(y, fs):
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/fs)
    return freq[np.argmax(spec)]


def mean_freq(y, fs):
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/fs)
    amp = spec / spec.sum()
    return (freq * amp).sum()


def to_spectro_frames(timestamps, pressure):
    # pressure = pressure[101430:190512]
    # pressure = pressure[:101430]
    spec = np.abs(np.fft.rfft(pressure))
    freq = np.fft.rfftfreq(len(pressure), d=timestamps[1] - timestamps[0])
    mf = mean_freq(pressure, 1/(timestamps[1] - timestamps[0]))

    plt.plot(freq, spec)
    plt.scatter([mf], [spec[int(mf)]])
    plt.show()


# example chunk sizes: 352, 22050
def to_dominant_freq(fs, pressure, chunk_size=352):
    dominant_y = []
    for i in range(0, len(pressure), chunk_size):
        chunk = pressure[i:i+chunk_size]
        dominant_y.append(mean_freq(chunk, fs))

    chunked_timestamps = np.arange(0, pressure.shape[0]/fs, chunk_size/fs)
    return chunked_timestamps, dominant_y


def debug_plot(x, y):
    plt.plot(x, y)
    plt.show()


def buffer_get_labels(filename, timestamps):
    all_data = pd.read_csv(filename, sep=',').values
    current = 0
    labels = []

    for timestamp in timestamps:
        if timestamp > all_data[current][1]:
            current += 1
        if current >= len(all_data): break
        labels.append(all_data[current][0])

    while len(labels) != len(timestamps):
        labels.append('out' if all_data[current-1][0] == 'in' else 'in')

    return labels


def ema(array):
    N = len(array)
    alpha = 2/N
    numerator = sum([array[k] * (1-alpha)**k for k in range(N)])
    denominator = sum([(1-alpha)**k for k in range(N)])
    return numerator/denominator


def signal_clean(signal):
    return savgol_filter(signal, 200, 5)


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
