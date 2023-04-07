import re

import macros
import src.data_engineering.data_utils as du
import src.real_time as rt
import src.data_engineering.spectrogram as sp
import src.models.svm as svm
import pathlib as pl

from src.new_realtime import new_realtime
from src.test import test_qualitative, test_quantitative
from src.new_realtime_tensor import new_realtime_tensor

def show_plot(file_csv, file_wav):
    x, y, sample_rate = du.wav_to_sample_xy(file_wav)
    timestamps, frames = sp.to_spectro(y, sample_rate)
    labels = sp.spectro_labeled(file_csv, timestamps)
    sp.pressure_labeled_plot(labels, x, y)


def show_spectrograms(file_wav, filename):
    x, y, sample_rate = du.wav_to_sample_xy(file_wav)
    sp.show_spectrograms(y, sample_rate, filename)


freg = re.compile(r'^e[0-9]+$')


def record():
    directory = pl.Path(macros.train_path)
    files = list(set([i.stem for i in directory.iterdir() if freg.search(i.stem)]))

    if not files:
        filename = f'{macros.train_path}e1'
    else:
        filename = []
        for i in files:
            filename.append(int(re.split('e', i)[1]))
        filename = macros.train_path + 'e' + (max(filename) + 1).__str__()
    du.data_recorder(filename, with_bg=False, seperate=False)

    if pl.Path.exists(pl.Path(filename + '.wav')):
        #   show_plot(filename + '.csv', filename + '.wav')
        print(filename)


if __name__ == '__main__':
    while True:
        print("1. Record\n2. Train\n3. Realtime\n4. Show plot\n5. Test quantitative\n7. New real time\n10. Show spectrograms")
        x = input()
        if x == '1':
            record()
        elif x == '2':
            folder = pl.Path(macros.train_path)
            svm.svm_train_with_previous_state(
                list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])),
                'svm_custom_softmax',
                True, False)
        elif x == '3':
            test_qualitative('svm_custom_softmax_prevstate', with_previous_state=True, with_bg=False)
        elif x == '4':
            x = input('Filename: ')
            show_plot(f'{macros.train_path}{x}.csv', f'{macros.train_path}{x}.wav')
        elif x == '5':
            folder = pl.Path(macros.test_path)
            test_quantitative(list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])), 'svm_custom_softmax_prevstate', True)
        elif x == '7':
            new_realtime('svm_custom_softmax_prevstate')
        elif x == '8':
            new_realtime_tensor()
        elif x == '10':
            x = input('Filename: ')
            show_spectrograms(f'{macros.train_path}{x}.wav', x)
