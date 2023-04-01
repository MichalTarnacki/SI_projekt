import numpy as np
import src.data_engineering.data_utils as du
import src.data_engineering.spectrogram as sp


def build(filenames, dir, previous_state=False):
    x_set = []
    y_set = []
    for filename in filenames:
        file_csv = f'{dir}{filename}.csv'
        file_wav = f'{dir}/{filename}.wav'
        x, y = du.wav_to_sample_xy(file_wav)
        timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
        labels = sp.spectro_labeled(file_csv, timestamps)

        if previous_state:
            frames[0] = np.append(frames[0], 1)
            for i in range(1, len(frames)):
                frames[i] = np.append(frames[i], 1 if labels[i - 1] == 'in' else -1)

        x_set += frames
        y_set += labels

    if previous_state:
        return np.array(x_set), y_set

    return x_set, y_set
