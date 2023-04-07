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

from keras import layers
from keras import models
from IPython import display
from scipy.io import wavfile

import macros
class TensorFlow:

    data_dir = pathlib.Path(macros.sorted_path)
    test_data_dir = pathlib.Path(macros.test_sorted_path)
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    commands = commands[commands != 'desktop.ini']
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self):
        pass
        # self.data_dir = pathlib.Path(macros.sorted_path)
        # self.seed = 42
        # tf.random.set_seed(self.seed)
        # np.random.seed(self.seed)
        # self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        # self.commands = self.commands[self.commands != 'README.md']
        # self.commands = self.commands[self.commands != 'desktop.ini']

    @staticmethod
    def generate_seperate_files(ratio=8/10):
        freg = re.compile(r'^e[0-9]+$')
        folder = pl.Path(macros.train_path)

        if os.path.exists(macros.sorted_path):
            folder2 = pl.Path(macros.sorted_path)
            shutil.rmtree(folder2)
        os.mkdir(macros.sorted_path)
        os.mkdir(macros.train_exhales)
        os.mkdir(macros.train_breaths)
        if os.path.exists(macros.test_sorted_path):
            folder2 = pl.Path(macros.test_sorted_path)
            shutil.rmtree(folder2)
        os.mkdir(macros.test_sorted_path)
        os.mkdir(macros.test_exhales)
        os.mkdir(macros.test_breaths)

        files = list(set([i.stem for i in folder.iterdir() if freg.search(i.stem)]))
        for file in files:
            pressure, sample_rate = soundfile.read(macros.train_path+file+'.wav')
            all_data = pd.read_csv(macros.train_path+file+'.csv', sep=',').values
            iterator = 0
            k=0
            for i in all_data:
                new_file = []
                while iterator < i[1]*sample_rate:
                    new_file.append(pressure[iterator])
                    iterator+=1
                if k<int(len(files)*ratio):
                    if i[0] == 'in':
                        soundfile.write(macros.train_breaths + file + f'{k}.wav',  np.array(new_file),sample_rate, subtype='PCM_16')
                    else:
                        soundfile.write(macros.train_exhales + file + f'{k}.wav',  np.array(new_file),sample_rate, subtype='PCM_16')
                else:
                    if i[0] == 'in':
                        soundfile.write(macros.test_breaths + file + f'{k}.wav', np.array(new_file), sample_rate,
                                        subtype='PCM_16')
                    else:
                        soundfile.write(macros.test_exhales + file + f'{k}.wav', np.array(new_file), sample_rate,
                                        subtype='PCM_16')
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
    def get_waveform(file_path):
        audio_binary = tf.io.read_file(file_path)
        waveform = TensorFlow.decode_audio(audio_binary)
        return waveform



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
        spectrogram = TensorFlow.get_spectrogram(audio)
        label_id = tf.argmax(label == TensorFlow.commands)
        return spectrogram, label_id

    @staticmethod
    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=TensorFlow.get_waveform_and_label,
            num_parallel_calls=TensorFlow.AUTOTUNE)
        output_ds = output_ds.map(
            map_func=TensorFlow.get_spectrogram_and_label_id,
            num_parallel_calls=TensorFlow.AUTOTUNE)
        return output_ds

    @staticmethod
    def preprocess_dataset2(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=TensorFlow.get_waveform,
            num_parallel_calls=TensorFlow.AUTOTUNE)
        output_ds = output_ds.map(
            map_func=TensorFlow.get_spectrogram,
            num_parallel_calls=TensorFlow.AUTOTUNE)
        return output_ds

    @staticmethod
    def get_files():
        test_validate_ratio = [8, 2]
        ratio = test_validate_ratio[0] / (test_validate_ratio[0] + test_validate_ratio[1])
        filenames = tf.io.gfile.glob(str(TensorFlow.data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        train_files = filenames[:int(len(filenames) * ratio)]
        val_files = filenames[int(len(filenames) * ratio):]
        test_filenames = tf.io.gfile.glob(str(TensorFlow.test_data_dir) + '/*/*')
        test_files = test_filenames
        return train_files, val_files, test_files

        #return waveform_ds, spectrogram_ds

    @staticmethod
    def create_ds(train_files):
        files_ds = tf.data.Dataset.from_tensor_slices(train_files)
        waveform_ds = files_ds.map(
            map_func=TensorFlow.get_waveform_and_label,
            num_parallel_calls=TensorFlow.AUTOTUNE)

        spectrogram_ds = waveform_ds.map(
            map_func=TensorFlow.get_spectrogram_and_label_id,
            num_parallel_calls=TensorFlow.AUTOTUNE)
        return files_ds, waveform_ds, spectrogram_ds

    @staticmethod
    def waveform_plot(waveform_ds, alldata=True, rows=3, cols=3):
        if alldata:
            number_of_el = list(waveform_ds).__len__()
            rows = int(np.sqrt(number_of_el))
            cols = int(np.sqrt(number_of_el))+(number_of_el-rows*rows)//rows + 1
        n = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.axis('off')

        k = enumerate(waveform_ds.take(rows*cols))
        for i, (audio, label) in k:
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(audio.numpy())
           # ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)
        plt.show()

    @staticmethod
    def spectro_plot(waveform_ds, alldata=True, take_num=1):
        if alldata:
            take_num=list(waveform_ds).__len__()
        for waveform, label in waveform_ds.take(take_num):
            plt.ion()
            label = label.numpy().decode('utf-8')
            spectrogram = TensorFlow.get_spectrogram(waveform)
            fig, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(waveform.shape[0])
            axes[0].plot(timescale, waveform.numpy())
            axes[0].set_title(f'Waveform: {label}')
            axes[0].set_xlim([0, 16000])
            TensorFlow.plot_spectrogram(spectrogram.numpy(), axes[1])
            axes[1].set_title('Spectrogram')
            print('Waveform shape:', waveform.shape)
            print('Spectrogram shape:', spectrogram.shape)
            sd.play(waveform)
            plt.show(block=True)
    @staticmethod
    def spectro_plots(spectrogram_ds, alldata=True, rows=3, cols=3):
        if alldata:
            number_of_el = list(spectrogram_ds).__len__()
            rows = int(np.sqrt(number_of_el))
            cols = int(np.sqrt(number_of_el)) + (number_of_el - rows * rows) // rows + 1
        n = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.axis('off')
        for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            TensorFlow.plot_spectrogram(spectrogram.numpy(), ax)
            ax.set_title(TensorFlow.commands[label_id.numpy()])

        plt.show()

    @staticmethod
    def history_plot(history):
        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.show()

    @staticmethod
    def accuracy(model, test_ds):
        test_audio = []
        test_labels = []
        for audio, label in test_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)

        y_pred = np.argmax(model.predict(test_audio), axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')

        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx,
                    xticklabels=TensorFlow.commands,
                    yticklabels=TensorFlow.commands,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
    @staticmethod
    def predict_file(model):
        sample_file = TensorFlow.data_dir / 'in/e10.wav'
        sample_ds = TensorFlow.preprocess_dataset([str(sample_file)])
        for spectrogram, label in sample_ds.batch(1):
            prediction = model(spectrogram)
            percentage = tf.nn.softmax(prediction[0])
            plt.bar(TensorFlow.commands, percentage)
            plt.title(f'Predictions for "{TensorFlow.commands[label[0]]}"')
            plt.show()

    @staticmethod
    def predict_percentage(model, audio_array):
        soundfile.write('media/trash/temp.wav',  np.array(audio_array), 44100, subtype='PCM_16')
        sample_ds = TensorFlow.preprocess_dataset2([str('media/sorted_test/in/e28.wav')])
        for spectrogram in sample_ds.batch(1):
            prediction = model(spectrogram)
            percentage = tf.nn.softmax(prediction[0])
            return TensorFlow.commands, percentage


    @staticmethod
    def train(epochs):

        train_files, val_files, test_files = TensorFlow.get_files()
        files_ds, waveform_ds, spectrogram_ds = TensorFlow.create_ds(train_files)
        #TensorFlow.waveform_plot(waveform_ds)
        #TensorFlow.spectro_plot(waveform_ds, alldata=True, take_num=2)
        #TensorFlow.spectro_plots(spectrogram_ds)

        train_ds = spectrogram_ds
        val_ds = TensorFlow.preprocess_dataset(val_files)
        test_ds = TensorFlow.preprocess_dataset(test_files)

        batch_size = 64
        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)

        train_ds = train_ds.cache().prefetch(TensorFlow.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(TensorFlow.AUTOTUNE)

        for spectrogram, _ in spectrogram_ds.take(1):
            input_shape = spectrogram.shape
       # print('Input shape:', input_shape)
        num_labels = len(TensorFlow.commands)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

        #Warstwy
        model = models.Sequential([
            layers.Input(shape=input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        EPOCHS = epochs
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=epochs),
        )
        if os.path.exists(f'{macros.model_path}tensorflow'):
            folder = pl.Path(f'{macros.model_path}tensorflow')
            shutil.rmtree(folder)
        model.save(f'{macros.model_path}tensorflow')






#TensorFlow.generate_seperate_files()
#i = TensorFlow()
#TensorFlow.predict_file(models.load_model(f'{macros.model_path}tensorflow'))