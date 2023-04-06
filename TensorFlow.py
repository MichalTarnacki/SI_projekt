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
import sounddevice as sd

# from tensorflow.keras import layers
# from tensorflow.keras import models
from IPython import display
from scipy.io import wavfile

import macros


# Set the seed value for experiment reproducibility.


class TensorFlow:
    def __init__(self):
        self.data_dir = pathlib.Path(macros.sorted_path)
        self.seed = 42
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        self.commands = self.commands[self.commands != 'README.md']
        self.commands = self.commands[self.commands != 'desktop.ini']

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

    @staticmethod
    def get_spectrogram(waveform):
        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = 16000
        waveform = waveform[:input_len]
        zero_padding = tf.zeros(
            [16000] - tf.shape(waveform),
            dtype=tf.float32)
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatenate the waveform with `zero_padding`, which ensures all audio
        # clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    @staticmethod
    def plot_spectrogram(spectrogram, ax):
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
        # Convert the frequencies to log scale and transpose, so that the time is
        # represented on the x-axis (columns).
        # Add an epsilon to avoid taking a log of zero.
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)
    @staticmethod
    def get_spectrogram_and_label_id(audio, label):
        data_dir = pathlib.Path(macros.sorted_path)
        commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        commands = commands[commands != 'README.md']
        commands = commands[commands != 'desktop.ini']
        spectrogram = TensorFlow.get_spectrogram(audio)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    @staticmethod
    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            map_func=get_spectrogram_and_label_id,
            num_parallel_calls=AUTOTUNE)
        return output_ds
    def cos(self):

        print('Commands:', self.commands)

        filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
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

        # rows = 3
        # cols = 3
        # n = rows * cols
        # fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        # k = enumerate(waveform_ds.take(rows*cols))
        # for i, (audio, label) in k:
        #     pass
        #     r = i // cols
        #     c = i % cols
        #     ax = axes[r][c]
        #     ax.plot(audio.numpy())
        #     ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        #     label = label.numpy().decode('utf-8')
        #     ax.set_title(label)
        #
        # plt.show()
        #
        # spectrogram = None
        # for waveform, label in waveform_ds.take(5):
        #     label = label.numpy().decode('utf-8')
        #     spectrogram = TensorFlow.get_spectrogram(waveform)
        #     fig, axes = plt.subplots(2, figsize=(12, 8))
        #     timescale = np.arange(waveform.shape[0])
        #     axes[0].plot(timescale, waveform.numpy())
        #     axes[0].set_title('Waveform')
        #     axes[0].set_xlim([0, 16000])
        #
        #     TensorFlow.plot_spectrogram(spectrogram.numpy(), axes[1])
        #     axes[1].set_title('Spectrogram')
        #     plt.show()

        # print('Label:', label)
        # print('Waveform shape:', waveform.shape)
        # print('Spectrogram shape:', spectrogram.shape)
        # print('Audio playback')
        # sd.play(waveform)
        #
        spectrogram_ds = waveform_ds.map(
            map_func=TensorFlow.get_spectrogram_and_label_id,
            num_parallel_calls=AUTOTUNE)
        rows = 3
        cols = 3
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

        for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
          r = i // cols
          c = i % cols
          ax = axes[r][c]
          TensorFlow.plot_spectrogram(spectrogram.numpy(), ax)
          ax.set_title(self.commands[label_id.numpy()])
          ax.axis('off')

        plt.show()


#TensorFlow.generate_seperate_files()
i = TensorFlow()
i.cos()