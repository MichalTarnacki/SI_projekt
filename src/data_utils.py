import time
import pygame
import sounddevice as sd
import pandas as pd
import src.signal_utils as su

from scipy.io.wavfile import write


def buffer_get_labels(filename, timestamps):
    all_data = pd.read_csv(filename, sep=',').values
    current = 0
    labels = []

    for timestamp in timestamps:
        if timestamp > all_data[current][1]:
            current += 1
        if current >= len(all_data): break
        labels.append(all_data[current][0])

    while len(labels) != len(timestamps):
        labels.append('out' if all_data[current-1][0] == 'in' else 'in')

    return labels


def file_with_labels(file_csv, file_wav):
    x, y = su.wav_to_sample_xy(file_wav)
    t, f = su.to_dominant_freq(x, y)

    return buffer_get_labels(file_csv, t), t, su.signal_clean(f)


def data_recorder(file_csv, file_wav):
    record_time_s = 30
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
                break
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
