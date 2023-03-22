import matplotlib.pyplot as plt

import src.svm as svm
import src.signal_utils as su
import src.data_utils as du
import src.model_generics as mg
import src.real_time as rt
import src.spectrogram as sp

from sklearn.preprocessing import StandardScaler
from joblib import load

def record(file):
    file_csv = f'media/train/{file}.csv'
    file_wav = f'media/train/{file}.wav'
    du.data_recorder(file_csv=file_csv, file_wav=file_wav)
    l, t, f = du.file_with_labels(file_csv=file_csv, file_wav=file_wav)
    su.debug_plot(t, f)
    su.debug_plot_marked(t, f, l)


if __name__ == '__main__':

    # rt.detection(
    #     load('media/models/spectro_svm.joblib'),
    #     load('media/models/spectro_svm_scaler.joblib'))
    record('e10')
    # x, y = su.wav_to_sample_xy('media/train/e9.wav')
    # timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
    # labels = sp.spectro_labeled('media/train/e9.csv', timestamps)
    #
    # # timestamps, frames = sp.to_spectro(y, 44100)
    # clf = load('media/models/spectro_svm.joblib')
    # scaler = load('media/models/spectro_svm_scaler.joblib')
    #
    # sp.pressure_labeled_plot(labels, x, y)
    #
    # frames_std = scaler.transform(frames)
    # predictions = clf.predict(frames_std)
    # sp.pressure_labeled_plot(predictions, x, y)
