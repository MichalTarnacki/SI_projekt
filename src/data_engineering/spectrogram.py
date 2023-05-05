import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import noisereduce
from scipy.signal import savgol_filter

import macros

CHUNK_SIZE = 1024
SAMPLE_FREQ = 44100


# change wav from sound pressure
# to spectrograms, each spectrogram
# is constructed for chunk_size samples
# so size of original input is decreased
# i.e. size = original_input_size/chunk_size
# returns frame and timestamp for each frame
def to_spectro(pressure, fs, chunk_size=CHUNK_SIZE):
    pressure = noisereduce.reduce_noise(pressure, sr=fs)
    frames = []
    chunked_timestamps = []
    chunk_index = 0

    for i in range(0, len(pressure), chunk_size):
        if i + chunk_size <= len(pressure):  # if there's enough elements in pressure to form a full chunk
            chunk = pressure[i:i + chunk_size]
            freq = abs(np.fft.rfft(chunk))
            freq = signal_clean(freq)

            frames.append(freq)
            chunked_timestamps.append(chunk_size / fs * chunk_index)
            chunk_index += 1

    return chunked_timestamps, frames, chunk_size


def to_spectro_loudonly(pressure, fs, chunk_size=CHUNK_SIZE):
    pressure = noisereduce.reduce_noise(pressure, sr=fs)
    frames = []
    chunked_timestamps = []
    chunk_index = 0

    for i in range(0, len(pressure), chunk_size):
        if i + chunk_size <= len(pressure):  # if there's enough elements in pressure to form a full chunk
            chunk = pressure[i:i + chunk_size]
            freq = abs(np.fft.rfft(chunk))
            freq = signal_clean(freq)

            if sum(freq) > 0.05:
                frames.append(freq)
                chunked_timestamps.append(chunk_size / fs * chunk_index)
            chunk_index += 1

    return chunked_timestamps, frames, chunk_size


# Saves first 30 spectrograms in specified spectro directory, prints most intense frequency for every frame,
# prints intensity sums for every frame
def show_spectrograms(pressure, sample_rate, filename):
    pressure = noisereduce.reduce_noise(pressure, sr=sample_rate)
    chunk_index = 0

    for i in range(0, len(pressure), CHUNK_SIZE):
        chunk = pressure[i:i + CHUNK_SIZE]
        freq = abs(np.fft.rfft(chunk))
        freq = signal_clean(freq)

        most_intense_freq_intensity = np.max(freq)
        most_intense_freq = np.where(freq == most_intense_freq_intensity)[0][0]
        intensity_sum = sum(freq)

        chunk_start_time = round(chunk_index * CHUNK_SIZE / sample_rate, 2)
        chunk_end_time = round((chunk_index + 1) * CHUNK_SIZE / sample_rate, 2)

        print(f"Chunk {chunk_start_time}-{chunk_end_time} s:\n"
              f"\tmost intense frequency: {most_intense_freq} FU_DFT"
              f" = {round(43.0789 * most_intense_freq - 2.8174, 2)} Hz\n"
              f"\tintensity sum: {round(intensity_sum, 2)} FU_DFT")

        if chunk_index < 30:
            plt.plot([i for i in range(len(freq))], freq)
            plt.scatter(most_intense_freq, most_intense_freq_intensity, c='r', s=10)
            plt.title(f'Spectrogram for chunk {chunk_start_time}-{chunk_end_time} s')
            plt.savefig(f'{macros.spectros_path}spectro_{filename}_{chunk_index}.png')
            plt.close()

        chunk_index += 1


def signal_clean(signal, window=5):
    return savgol_filter(signal, window, 2)


# add labels to constructed spectrogram frames
# data = (timestamps, frames)
def spectro_labeled(label_file, timestamps):
    all_data = pd.read_csv(label_file, sep=',').values
    labels = []
    current = 0

    for timestamp in timestamps:
        if timestamp > all_data[current][1]:
            current += 1
        if current >= len(all_data):
            break
        labels.append(all_data[current][0])

    while len(labels) != len(timestamps):
        labels.append('out' if all_data[current - 1][0] == 'in' else 'in')

    return labels


def spectro_labeled_loudonly(label_file, timestamps):
    all_data = pd.read_csv(label_file, sep=',').values
    labels = []
    current = 0

    for timestamp in timestamps:
        to_break = False

        while timestamp >= all_data[current][1]:
            current += 1
            if current >= len(all_data):
                to_break = True
                break

        if to_break:
            break

        labels.append(all_data[current][0])

    while len(labels) != len(timestamps):
        labels.append('out' if all_data[current - 1][0] == 'in' else 'in')

    return labels


# draws labeled plot
def pressure_labeled_plot(labels, time, pressure, chunk_size=CHUNK_SIZE):
    labels_extended = []
    x_in = []
    y_in = []
    x_out = []
    y_out = []

    for label in labels:
        for i in range(chunk_size):
            labels_extended.append(label)

    for i in range(len(labels_extended)):
        if labels_extended[i] == 'in':
            x_in.append(time[i])
            y_in.append(pressure[i])
        else:
            x_out.append(time[i])
            y_out.append(pressure[i])
    plt.scatter(x_out, y_out, s=5)
    plt.scatter(x_in, y_in, s=5)
    plt.legend(['out', 'in'])
    plt.show()
