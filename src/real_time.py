import copy
import sys
import time
import threading

import matplotlib.pyplot as plt
import noisereduce
import numpy as np
import pygame
import pyaudio
import sounddevice as sd
import soundcard as sc
from scipy.signal import savgol_filter

import src.data_engineering.spectrogram as sp

from pydub import effects
from scipy.io.wavfile import write


def detection(model, scaler, chunk_size=352, input_size=40, uses_previous_state=False, with_bg=False):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((740, 480))
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
            # clean = noisereduce.reduce_noise(samples[-1024:], 44100)

            clean = samples
            last_frame = abs(np.fft.rfft(clean[len(clean) - sp.CHUNK_SIZE:]))[:160]
            last_frame = sp.signal_clean(last_frame)
            if uses_previous_state:
                last_frame = np.append(last_frame, 1 if prev_state == 'in' else -1)
                prev_state = state
            # clean_frame = sp.signal_clean(last_frame[:-1])
            for x, y in enumerate(last_frame[:-1]):
                pygame.draw.line(screen, (255, 0, 0) if state == 'in' else (0, 255, 0), (x*3, 480 - y), (x*3 + 3, 480 - last_frame[x+1]))

            last_frame_std = scaler.transform(last_frame.reshape(-1, 1).T)
            state = model.predict(last_frame_std)

            # if samples.shape[0] > 44100:
                # samples = samples[1024:]

        text = font.render('teraz wdychasz' if state == 'in' else 'teraz wydychasz', True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()
