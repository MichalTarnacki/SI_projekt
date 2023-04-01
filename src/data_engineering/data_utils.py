import time
import pygame
import sounddevice as sd
import pandas as pd
import numpy as np
import src.old.signal_utils as su

from scipy.io.wavfile import write
from scipy.io import wavfile

def wav_to_sample_xy(filename):
    sample_rate, pressure = wavfile.read(filename)
    timestamps = np.arange(0, pressure.shape[0]/sample_rate, 1/sample_rate)
    return timestamps, pressure, sample_rate


def data_recorder(file_csv, file_wav):
    record_time_s = 25
    sample_rate = 44100
    channels = 1

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((740, 480))
    font = pygame.font.SysFont(None, 50)
    timestamps = []
    state = 'in'

    rec = sd.rec(int(record_time_s * sample_rate), samplerate=sample_rate, channels=channels)
    t0 = time.time()

    while time.time() - t0 < record_time_s:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                timestamps.append((state, time.time() - t0))
                state = 'in' if state == 'out' else 'out'

        screen.fill((0, 0, 0))
        text = font.render('teraz wdychasz' if state == 'in' else 'teraz wydychasz', True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()

    pygame.quit()
    sd.wait()

    pd.DataFrame({'type': [y[0] for y in timestamps],
                  'time_right': [y[1] for y in timestamps]
                  }).to_csv(file_csv, index=False)
    write(file_wav, sample_rate, rec)
