import re

import macros
import src.data_engineering.data_utils as du
import src.real_time as rt
import src.data_engineering.spectrogram as sp
import src.models.svm as svm
import pathlib as pl

from src.test import test_qualitative, test_quantitative

def show_plot(file_csv, file_wav):
    x, y, sample_rate = du.wav_to_sample_xy(file_wav)
    timestamps, frames = sp.to_spectro(y, sample_rate)
    labels = sp.spectro_labeled(file_csv, timestamps)
    sp.pressure_labeled_plot(labels, x, y)


def record():
    folder = pl.Path(macros.train_path)
    files = list(set([i.stem for i in folder.iterdir()]))
    if files == []:
        filename = f'{macros.train_path}e1'
    else:
        filename = []
        for i in files:
            filename.append(int(re.split('e', i)[1]))
        filename = macros.train_path + 'e' + (max(filename)+1).__str__()
    du.data_recorder(filename)

    if pl.Path.exists(pl.Path(filename + '.wav')):
     #   show_plot(filename + '.csv', filename + '.wav')
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
                test_qualitative('svm_custom_softmax_prevstate', False)
            case '4':
                x=input()
                show_plot(f'{macros.train_path}{x}.csv', f'{macros.train_path}{x}.wav')
