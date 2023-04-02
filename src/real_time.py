import sys
import time
import threading

import noisereduce
import numpy as np
import pygame
import pyaudio
import sounddevice as sd
import src.data_engineering.spectrogram as sp

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


    s = 0
    if with_bg:
        rec = sd.rec(int(record_bg_time_s * fs), samplerate=fs, channels=channels)
        t0 = time.time()

        while time.time() - t0 < record_bg_time_s:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill((0, 0, 0))
            text = font.render(
                f'Recording background, try not to breath: {(record_bg_time_s - time.time() + t0).__str__()}', True,
                (255, 255, 255))
            screen.blit(text, (0, 0))
            pygame.display.update()

        sd.stop()
        s = [i[0] for i in rec]
        s = np.mean(s)

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1)
    samples = np.array([])

    # lock = threading.Lock()
    run_thread = True
    def record_thread():
        nonlocal stream
        nonlocal samples
        while run_thread:
            data = np.frombuffer(stream.read(1, exception_on_overflow=False), dtype=np.int16)
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
        if samples.shape[0] > chunk_size * (input_size + 200):
            if with_bg:
                samples = [i if i > s else 0 for i in samples]
            clean = noisereduce.reduce_noise(samples, 44100)
            last_frame = abs(np.fft.rfft(clean[len(clean) - sp.CHUNK_SIZE:]))[:160]
            if uses_previous_state:
                last_frame = np.append(last_frame, 1 if prev_state == 'in' else -1)
                prev_state = state

            last_frame_std = scaler.transform(last_frame.reshape(-1, 1).T)
            state = model.predict(last_frame_std)

        screen.fill((0, 0, 0))
        text = font.render('teraz wdychasz' if state == 'in' else 'teraz wydychasz', True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()
