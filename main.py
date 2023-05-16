# from keras import models
from joblib import load
from keras import models

from macros import freg
import macros
import src.data_engineering.data_utils as du
import src.real_time as rt
import src.data_engineering.spectrogram as sp
import src.models.svm as svm
import pathlib as pl
from TensorFlow import TensorFlow
from src.new_realtime import new_realtime
from src.test import test_quantitative, test_qualitative, test_quantitative_loudonly, test_qualitative_loudonly
from src.new_realtime_tensor import new_realtime_tensor
import re
import src.real_time as realtime


def show_plot(file_csv, file_wav):
    x, y, sample_rate = du.wav_to_sample_xy(file_wav)
    timestamps, frames, chunk_size = sp.to_spectro(y, sample_rate)
    labels = sp.spectro_labeled(file_csv, timestamps)
    sp.pressure_labeled_plot(labels, x, y)


def show_spectrograms(file_wav, filename):
    x, y, sample_rate = du.wav_to_sample_xy(file_wav)
    sp.show_spectrograms(y, sample_rate, filename)


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


NOSE_IN_MOUTH_OUT_MODEL = "media/models/outmodel/nose_in_mouth_out/svm_prevstate"


def menu():
    while True:
        print("1. Nose in, mouth out\n"
              "2. Nose in, nose out\n")
        x = input()
        if x == "1":
            realtime.detection(
                load(NOSE_IN_MOUTH_OUT_MODEL + ".joblib"),
                load(NOSE_IN_MOUTH_OUT_MODEL + "_scaler.joblib"),
                uses_previous_state=True,
                loudonly=False)
        elif x == "2":
            pass


if __name__ == '__main__':
    while True:
        print("1. Record\n2. Train with SVM\n2.1 Train with loud-only SVM\n3. Realtime with SVM"
              "\n3.1 Realtime with loud-only SVM\n4. Show pressure plot\n5. Test quantitative with SVM"
              "\n5.1 Test quantitative with loud-only SVM\n7. New real time with SVM"
              "\n8. Real time tensor\n10. Train with Tensorflow\n11. Show spectrograms")
        x = input()
        if x == '1':
            record()
        elif x == '2':
            folder = pl.Path(macros.train_path)
            svm.svm_train_with_previous_state(
                list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])),
                'svm_trained',
                mouth_out=True, loudonly=False)
        elif x == '2.1':
            folder = pl.Path(macros.train_path)
            svm.svm_train_with_previous_state(
                list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])),
                'svm_trained_loud',
                mouth_out=True, loudonly=True)
        elif x == '3':
            test_qualitative('svm_trained', with_previous_state=True)
        elif x == '3.1':
            test_qualitative_loudonly('svm_trained_loud', with_previous_state=True)
        elif x == '4':
            x = input('Filename: ')
            show_plot(f'{macros.train_path}{x}.csv', f'{macros.train_path}{x}.wav')
        elif x == '5':
            folder = pl.Path(macros.test_path)
            test_quantitative(list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])),
                              'svm_trained', True)
        elif x == '5.1':
            folder = pl.Path(macros.test_path)
            test_quantitative_loudonly(list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])),
                                       'svm_trained_loud', True)
        elif x == '7':
            new_realtime()
        elif x == '8':
            new_realtime_tensor()
        elif x == '9':
            TensorFlow.generate_seperate_files()
        elif x == '10':
            TensorFlow.train(int(input('epochs')))
        elif x == '10.1':
            TensorFlow.train_rnn(int(input('epochs')))
        elif x == '11':
            x = input('Filename: ')
            show_spectrograms(f'{macros.train_path}{x}.wav', x)
        elif x == '12':
            TensorFlow.accuracy(models.load_model(f'{macros.model_path}tensorflow'))
