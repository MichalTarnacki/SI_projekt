import numpy as np
import src.data_engineering.data_utils as du
import src.data_engineering.spectrogram as sp


def build(filenames, dir, previous_state=False, with_bg=True):
    x_set = []
    y_set = []
    for filename in filenames:
        file_csv = f'{dir}{filename}.csv'
        file_wav = f'{dir}{filename}.wav'
        #x - timestamps y - pressure
        x, y, freq = du.wav_to_sample_xy(file_wav)

        if with_bg:
            file_bg_wav = f'{dir}{filename}.bgwav'
            x_bg, y_bg, freq_bg = du.wav_to_sample_xy(file_wav)
            x_bg = np.mean(x_bg)
            x = [i if i>x_bg else 0 for i in x]




        timestamps, frames = sp.to_spectro(y, freq)
        labels = sp.spectro_labeled(file_csv, timestamps)

        if previous_state:
            frames[0] = np.append(frames[0], 1) # TODO po co??
            for i in range(1, len(frames)):
                frames[i] = np.append(frames[i], 1 if labels[i - 1] == 'in' else -1)

        x_set += frames
        y_set += labels

    if previous_state:
        return np.array(x_set), y_set

    return x_set, y_set
