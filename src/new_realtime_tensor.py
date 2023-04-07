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


def new_realtime_tensor(with_bg=False):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    window = np.blackman(CHUNK)
    model = models.load_model(f'{macros.model_path}tensorflow')
    avgnoise=0
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
   # ax2 = fig.add_subplot(212)
    ax3 = fig.add_subplot(212)

    saved = []
    p = None
    stream = None
    contin = True
    saved_chunks = 1



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
            # sd.play(waveData, 44100)
            saved = saved + list(waveData)
            if saved.__len__() >= (saved_chunks+1)*CHUNK:
                saved = saved[CHUNK:]
        stream.stop_stream()
        stream.close()
        p.terminate()

    def write_thread():
        nonlocal saved

    def soundPlot():
        nonlocal saved, window, ax1, ax3, model, avgnoise, saved_chunks
        i = 0
        k=0
        while True:
            #t1 = time.time()
            if saved.__len__() >= saved_chunks*CHUNK:
                data = sp.signal_clean(saved)
                npArrayData = np.array([i if i>avgnoise else 0 for i in saved[saved.__len__() - CHUNK:]])
                npArrayData_reduced = np.array([i if i>avgnoise else 0 for i in data[data.__len__() - CHUNK:]])

                t_pred = copy.deepcopy(npArrayData)
                commands, pred  = TensorFlow.predict_percentage(model,t_pred)

                indata = npArrayData * window
                fftData = np.abs(np.fft.rfft(indata))

                fftTime = np.fft.rfftfreq(CHUNK, 1. / RATE)

                # indata2 = npArrayData_reduced * window
                # fftData2 = np.abs(np.fft.rfft(indata2))
                # fftData2 = fftData2[fftData2.__len__() - 161:]
                # which = fftData[1:].argmax() + 1

                color = None
                if pred[0] > 0.8:
                    color = 'g'
                elif pred[1] > 0.8:
                    color = 'r'
                else:
                    color = 'b'

                # Plot time domain
                ax1.cla()
                ax1.plot(indata, color)
                ax1.grid()
                # ax3.cla()
                # ax1.plot(indata2, color)
                # ax3.grid()
                if np.mean(fftData) > avgnoise:
                    ax1.set_title(('in' if color == 'g' else 'out') + k.__str__())
                else:
                    ax1.set_title('none')
                ax1.axis([0, sp.CHUNK_SIZE, -1000, 1000])
              #  ax3.axis([0, sp.CHUNK_SIZE, -5000, 5000])
                # Plot frequency domain graph
                # ax2.cla()
                # ax2.plot(fftTime, fftData, 'g' if prev_state == 'in' else 'r')
                # ax2.grid()
                # ax2.axis([0, 5000, 0, 10 ** 6])
                # ax3.cla()
                # ax3.plot(fftTime, fftData2, 'b' if prev_state == 'in' else 'y')
                # ax3.grid()
                # ax3.axis([0, 5000, 0, 10 ** 6])
                plt.pause(0.001)


    if with_bg:
        record_time_s = 25
        record_bg_time_s = 10
        sample_rate = 44100
        channels = 1

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((740, 480))
        font = pygame.font.SysFont(None, 50)
        tr = threading.Thread(target=record_thread, args=(1000,))
        tr.start()
        t0 = time.time()

        while time.time() - t0 < record_bg_time_s:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    tr.join()
                    pygame.quit()
                    return

            screen.fill((0, 0, 0))
            text = font.render(
                f'Recording background, try not to breath: {(record_bg_time_s - time.time() + t0).__str__()}', True,
                (255, 255, 255))
            screen.blit(text, (0, 0))
            pygame.display.update()
        contin = False
        tr.join()
        pygame.quit()
        avgnoise = np.mean(saved)
        saved = []
        #sd.stop()
    contin = True

    tr = threading.Thread(target=record_thread)
    tr2 = threading.Thread(target=write_thread)
    tr.start()
    tr2.start()
    soundPlot()
    contin = False
    tr.join()
    write('test.wav', RATE, np.array(saved))
    tr2.join()
