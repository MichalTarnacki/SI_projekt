import re

import macros
import src.data_engineering.data_utils as du
import src.real_time as rt
import src.data_engineering.spectrogram as sp
import src.models.svm as svm
import pathlib as pl

from src.test import test_qualitative, test_quantitative
from src.test import test_quantitative_with_previous_state, test_qualitative_with_previous_state


def record():
    folder = pl.Path(macros.train_path)
    files = list(set([i.stem for i in folder.iterdir()]))
    filename = []
    for i in files:
        filename.append(int(re.split('e', i)[1]))
    filename = 'e' + (max(filename)+1).__str__()
    file_csv = f'{macros.train_path}{filename}.csv'
    file_wav = f'{macros.train_path}{filename}.wav'
    du.data_recorder(file_csv=file_csv, file_wav=file_wav)

    if pl.Path.exists(pl.Path(file_wav)):
        x, y = du.wav_to_sample_xy(file_wav)
        timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
        labels = sp.spectro_labeled(file_csv, timestamps)
        sp.pressure_labeled_plot(labels, x, y)

        print(filename)


if __name__ == '__main__':
    while True:
        print("1.Record\n2.Train\n3.Realtime")
        match input():
            case '1':
                record()
            case '2':
                folder = pl.Path(macros.train_path)
                svm.svm_train_with_previous_state(
                    list(set([i.stem for i in folder.iterdir()])),
                    'svm_custom_softmax',
                    True)
            case '3':
               # test_quantitative_with_previous_state(['e9'], 'svm_custom_softmax_prevstate')
                test_qualitative_with_previous_state('svm_custom_softmax_prevstate')
