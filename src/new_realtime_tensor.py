import array
import copy
import threading

import noisereduce
import pyaudio
import numpy as np
import time
import wave
import matplotlib.pyplot as plt
import pygame
from keras import models
from scipy.io.wavfile import write
import src.data_engineering.spectrogram as sp
import sounddevice as sd
from TensorFlow import TensorFlow

import macros


def new_realtime_tensor():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    window = np.blackman(CHUNK)
    model = models.load_model(f'{macros.model_path}tensorflow')
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)

    saved = []
    p = None
    stream = None
    contin = True
    saved_chunks = 20

    def record_thread():
        nonlocal stream
        nonlocal saved
        nonlocal contin
        nonlocal saved_chunks
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        while contin:
            waveData = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

            saved.extend(waveData)
            if saved.__len__() >= (saved_chunks + 1) * CHUNK:
                # sd.play(saved, 44100)
                saved = saved[CHUNK:]
        stream.stop_stream()
        stream.close()
        p.terminate()

    def write_thread():
        nonlocal saved

    def soundPlot():
        nonlocal saved, window, ax1, model, saved_chunks
        while True:
            if saved.__len__() >= saved_chunks * CHUNK:
                # sd.play(saved, 44100)
                # x = sp.signal_clean(saved)
                commands, pred = TensorFlow.new_predict(model, saved)

                color = None
                title = None
                if pred[1] > 0.90:  # and pred[1]<10:
                    color = 'g'  # if commands[0] == 'in' else 'r'
                    title = 'in'  # if commands[0] == 'in' else 'out'
                elif pred[0] > 0.99:  # and pred[0]<10:
                    color = 'r'  # if commands[1] == 'out' else 'g'
                    title = 'out'  # if commands[1] == 'out' else 'in'
                else:
                    title = 'none'
                    color = 'b'

                indata = saved[saved.__len__() - CHUNK:] * window
                fftData = np.abs(np.fft.rfft(indata))
                fftTime = np.fft.rfftfreq(CHUNK, 1. / RATE)
                ax1.cla()
                ax1.plot(indata, color)
                ax1.grid()
                ax1.set_title(title)
                plt.ylim([0,1])
                ax1.bar(commands, pred, color='maroon',
                        width=0.4)
                plt.pause(0.001)

    tr = threading.Thread(target=record_thread)
    tr2 = threading.Thread(target=write_thread)
    tr.start()
    tr2.start()
    soundPlot()
    contin = False
    tr.join()
    tr2.join()
