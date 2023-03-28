import numpy as np
import src.data_engineering.data_utils as du
import src.data_engineering.spectrogram as sp


def build_train(filenames, previous_state=False):
    if previous_state:
        return build_with_previous_state(filenames, 'train')
    return build(filenames, 'train')


def build_test(filenames, previous_state=False):
    if previous_state:
        return build_with_previous_state(filenames, 'test')
    return build(filenames, 'test')


def build(filenames, dir):
    x_set = []
    y_set = []
    for filename in filenames:
        file_csv = f'media/{dir}/{filename}.csv'
        file_wav = f'media/{dir}/{filename}.wav'
        x, y = du.wav_to_sample_xy(file_wav)
        timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
        labels = sp.spectro_labeled(file_csv, timestamps)
        x_set += frames
        y_set += labels

    return x_set, y_set


def build_with_previous_state(filenames, dir):
    x_set = []
    y_set = []
    for filename in filenames:
        file_csv = f'media/{dir}/{filename}.csv'
        file_wav = f'media/{dir}/{filename}.wav'
        x, y = du.wav_to_sample_xy(file_wav)
        timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
        labels = sp.spectro_labeled(file_csv, timestamps)

        frames[0] = np.append(frames[0], 1)
        for i in range(1, len(frames)):
            frames[i] = np.append(frames[i], 1 if labels[i-1] == 'in' else -1)

        x_set += frames
        y_set += labels

    return np.array(x_set), y_set
