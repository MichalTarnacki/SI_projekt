import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pathlib as pl
from scipy.io.wavfile import write
import soundfile
import shutil

# from tensorflow.keras import layers
# from tensorflow.keras import models
from IPython import display
from scipy.io import wavfile

import macros


# Set the seed value for experiment reproducibility.


class TensorFlow:
    @staticmethod
    def generate_seperate_files():
        freg = re.compile(r'^e[0-9]+$')
        folder = pl.Path(macros.train_path)

        if os.path.exists(macros.sorted_path):
            folder2 = pl.Path(macros.sorted_path)
            shutil.rmtree(folder2)
        os.mkdir(macros.sorted_path)
        os.mkdir(macros.train_exhales)
        os.mkdir(macros.train_breaths)

        for file in list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)])):
            pressure, sample_rate = soundfile.read(macros.train_path+file+'.wav')
            all_data = pd.read_csv(macros.train_path+file+'.csv', sep=',').values
            iterator = 0
            k=0
            for i in all_data:
                new_file = []
                while iterator < i[1]*sample_rate:
                    new_file.append(pressure[iterator])
                    iterator+=1
                if i[0] == 'in':
                    soundfile.write(macros.train_breaths + file + f'{k}.wav',  np.array(new_file),sample_rate, subtype='PCM_16')
                else:
                    soundfile.write(macros.train_exhales + file + f'{k}.wav',  np.array(new_file),sample_rate, subtype='PCM_16')
                k+=1

    @staticmethod
    def decode_audio(audio_binary):
        # Decode WAV-encoded audio files to `float32` tensors, normalized
        # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
        audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=1)
        # Since all the data is single channel (mono), drop the `channels`
        # axis from the array.
        return tf.squeeze(audio, axis=-1)

    @staticmethod
    def get_label(file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        return parts[-2]

    @staticmethod
    def get_waveform_and_label(file_path):
        label = TensorFlow.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = TensorFlow.decode_audio(audio_binary)
        return waveform, label



    def cos(self):
        data_dir = pathlib.Path(macros.sorted_path)
        # DATASET_PATH = r'C:\Users\theed\Downloads\mini_speech_commands'
        #
        # data_dir = pathlib.Path(DATASET_PATH)
        # if not data_dir.exists():
        #     tf.keras.utils.get_file(
        #         'mini_speech_commands.zip',
        #         origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        #         extract=True,
        #         cache_dir='.', cache_subdir='data')
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        commands = commands[commands != 'README.md']
        commands = commands[commands != 'desktop.ini']
        print('Commands:', commands)

        filenames = tf.io.gfile.glob(str(data_dir) + '*/*')
        filenames = tf.random.shuffle(filenames)

        # train_files = filenames[:int(len(filenames)*8/10)]
        # val_files = filenames[int(len(filenames)*8/10): int(len(filenames)*8/10 + len(filenames)*1/10)]
        # test_files = filenames[int(-len(filenames)*1/10):]

        train_files = filenames[:60]
        val_files = filenames[60: 70]
        test_files = filenames[-8:]

        AUTOTUNE = tf.data.AUTOTUNE

        files_ds = tf.data.Dataset.from_tensor_slices(train_files)

        waveform_ds = files_ds.map(
            map_func=TensorFlow.get_waveform_and_label,
            num_parallel_calls=AUTOTUNE)

        rows = 3
        cols = 3
        n = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        k = enumerate(waveform_ds.take(1))
        for i, (audio, label) in k:
            pass
            # r = i // cols
            # c = i % cols
            # ax = axes[r][c]
            # ax.plot(audio.numpy())
            # ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            # label = label.numpy().decode('utf-8')
            # ax.set_title(label)

        plt.show()


#TensorFlow.generate_seperate_files()
i = TensorFlow()
i.cos()