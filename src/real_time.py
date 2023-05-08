import copy
import sys
import time
import threading

import matplotlib.pyplot as plt
import noisereduce
import numpy as np
import pygame
import pyaudio
import scipy.signal
from scipy.signal import find_peaks, find_peaks_cwt
import sounddevice as sd
import soundcard as sc
from scipy.signal import savgol_filter

import src.data_engineering.spectrogram as sp

from pydub import effects
from scipy.io.wavfile import write


def detection(model, scaler, chunk_size=352, input_size=40, uses_previous_state=False, with_bg=False):
    pygame.init()
    pygame.font.init()

    width = 740
    height = 480
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont(None, 50)
    p = pyaudio.PyAudio()
    fs = 44100
    record_bg_time_s = 10
    channels = 1

    samples = np.array([])

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)

    run_thread = True
    def record_thread():
        # nonlocal stream
        nonlocal samples
        while run_thread:
                data = np.frombuffer(stream.read(512, exception_on_overflow=False), dtype=np.int16)
                data = data / 50
                samples = np.append(samples, data)

    tr = threading.Thread(target=record_thread)
    tr.start()

    state = 'in'
    prev_state = 'in'
    print_state = 'cisza'

    radius = 100

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run_thread = False
                tr.join()
                stream.stop_stream()
                stream.close()
                p.terminate()
                write('test.wav', fs, samples)
                pygame.quit()
                return
        screen.fill((0, 0, 0))

        if samples.shape[0] > chunk_size * (input_size + 200):
            clean = noisereduce.reduce_noise(samples[-176400:], 44100)

            last_frame = abs(np.fft.rfft(clean[len(clean) - sp.CHUNK_SIZE:]))
            last_frame = sp.signal_clean(last_frame)

            if sum(last_frame) < 200:
                print_state = "cisza"
            else:
                print_state = ""

            if uses_previous_state:
                last_frame = np.append(last_frame, 1 if prev_state == 'in' else -1)
                prev_state = state

            color = (0, 0, 255)
            if print_state == "cisza":
                color = (0, 255, 0)
            elif state == "out":
                color = (255, 0, 0)
                radius -= 1
            else:
                radius += 1

            # for x, y in enumerate(last_frame[:-1]):
            #     pygame.draw.line(screen, color, (x*3, 480 - y), (x*3 + 3, 480 - last_frame[x+1]))



            pygame.draw.circle(screen, color, (width/2, height/2), radius)

            last_frame_std = scaler.transform(last_frame.reshape(-1, 1).T)
            state = model.predict(last_frame_std)


        # if print_state == "cisza":
        #     text = font.render('cisza', True, (255, 255, 255))
        # else:
        #     text = font.render('teraz wdychasz' if state == 'in' else 'teraz wydychasz', True, (255, 255, 255))
        # screen.blit(text, (0, 0))
        pygame.display.update()


def detection_loudonly(model, scaler, chunk_size=352, input_size=40, uses_previous_state=False, with_bg=False):
    pygame.init()
    pygame.font.init()

    width = 740
    height = 480
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont(None, 50)
    p = pyaudio.PyAudio()
    fs = 44100
    record_bg_time_s = 10
    channels = 1

    samples = np.array([])

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)

    run_thread = True
    def record_thread():
        # nonlocal stream
        nonlocal samples
        while run_thread:
                data = np.frombuffer(stream.read(512, exception_on_overflow=False), dtype=np.int16)
                data = data / 50
                samples = np.append(samples, data)

    tr = threading.Thread(target=record_thread)
    tr.start()

    state = 'in'
    prev_state = 'in'
    print_state = 'cisza'

    radius = 100
    fx = 0
    pred_history = []
    pred_time = []
    resprate = ""
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run_thread = False
                tr.join()
                stream.stop_stream()
                stream.close()
                p.terminate()
                write('test.wav', fs, samples)
                pygame.quit()
                return
        screen.fill((0, 0, 0))

        if samples.shape[0] > chunk_size * (input_size + 200):
            clean = noisereduce.reduce_noise(samples[-176400:], 44100)

            last_frame = abs(np.fft.rfft(clean[len(clean) - sp.CHUNK_SIZE:]))
            last_frame = sp.signal_clean(last_frame)

            if sum(last_frame) < 400:
                print_state = "cisza"
            else:
                print_state = ""

            if uses_previous_state:
                last_frame = np.append(last_frame, 1 if prev_state == 'in' else -1)
                prev_state = state

            color = (0, 0, 255)
            if print_state == "cisza":
                color = (0, 255, 0)
            elif state == "out":
                color = (255, 0, 0)
                fx -= 0.05
                pred_history.append(fx)
                pred_time.append(time.time())
                radius -= 0.5
            else:
                fx += 0.1
                pred_history.append(fx)
                pred_time.append(time.time())
                radius += 1

            if len(pred_history) > 0:
                peaks = find_peaks_cwt(pred_history, widths=np.arange(5, 15))
                if len(peaks) > 1:
                    sumdist = 0
                    for j in range(len(peaks)-1):
                        sumdist += pred_time[peaks[j+1]] - pred_time[peaks[j]]
                    avgT = sumdist / (len(peaks) - 1)
                    resprate = 60/avgT

            if len(pred_history) > width:
                pred_history = pred_history[-width:]
                pred_time = pred_time[-width:]

            for x, y in enumerate(pred_history[:-1]):
                pygame.draw.line(screen, color, (x, 400 - y * 5), (x+ 1, 400 - pred_history[x + 1] * 5))
                # if x in peaks:
                #     pygame.draw.circle(screen, (0, 255, 0), (x, 400-y*5), 5)
            # for x, y in enumerate(last_frame[:-1]):
            #     pygame.draw.line(screen, color, (x*3, 480 - y), (x*3 + 3, 480 - last_frame[x+1]))

            pygame.draw.circle(screen, color, (width/2, height/2), radius)

            last_frame_std = scaler.transform(last_frame.reshape(-1, 1).T)
            state = model.predict(last_frame_std)

        text = font.render(f"RR: {resprate}", True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()
