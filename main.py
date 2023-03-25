import src.data_engineering.data_utils as du
import src.real_time as rt
import src.data_engineering.spectrogram as sp
import src.models.svm as svm

from src.test import test_qualitative, test_quantitative


def record(file):
    file_csv = f'media/train/{file}.csv'
    file_wav = f'media/train/{file}.wav'
    du.data_recorder(file_csv=file_csv, file_wav=file_wav)

    x, y = du.wav_to_sample_xy(file_wav)
    timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
    labels = sp.spectro_labeled(file_csv, timestamps)
    sp.pressure_labeled_plot(labels, x, y)


if __name__ == '__main__':
    svm.svm_train(['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e10', 'e11'], 'svm_custom2')
    test_quantitative(['e9'], 'svm_custom2')
    test_qualitative('svm_custom2')
