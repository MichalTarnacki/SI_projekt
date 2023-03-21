import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.signal_utils as su
import noisereduce

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


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
    chunked_timestamps = np.arange(0, pressure.shape[0] / fs, chunk_size / fs)

    for i in range(0, len(pressure), chunk_size):
        chunk = pressure[i:i + chunk_size]
        freq = abs(np.fft.rfft(chunk))[:160]
        frames.append(freq)

    return chunked_timestamps, frames


# add labels to constructed spectrogram frames
# data = (timestamps, frames)
def spectro_labeled(label_file, timestamps):
    all_data = pd.read_csv(label_file, sep=',').values
    labels = []
    current = 0

    for timestamp in timestamps:
        if timestamp > all_data[current][1]:
            current += 1
        if current >= len(all_data): break
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

    for i in range(len(time)):
        if labels_extended[i] == 'in':
            x_in.append(time[i])
            y_in.append(pressure[i])
        else:
            x_out.append(time[i])
            y_out.append(pressure[i])

    plt.scatter(x_in, y_in, s=5)
    plt.scatter(x_out, y_out, s=5)
    plt.legend(['in', 'out'])
    plt.show()


def svm_train(filenames):
    x_train = []
    y_train = []

    for filename in filenames:
        file_csv = f'media/train/{filename}.csv'
        file_wav = f'media/train/{filename}.wav'
        x, y = su.wav_to_sample_xy(file_wav)
        timestamps, frames = to_spectro(y, SAMPLE_FREQ)
        labels = spectro_labeled(file_csv, timestamps)
        x_train += frames
        y_train += labels

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = SVC(kernel='linear', verbose=1)
    clf.fit(x_train_std, y_train)
    dump(clf, 'media/models/spectro_svm.joblib')
    dump(scaler, 'media/models/spectro_svm_scaler.joblib')

    return clf, scaler
