import pygame
import sounddevice as sd


from pandas import read_csv
from scipy.io.wavfile import write


def label_to_sample(filename, timestamps):
    all_data = read_csv(filename, sep=';').values
    current = 0
    labels = []

    for timestamp in timestamps:
        if timestamp > all_data[current][1]:
            current += 1
        labels.append(all_data[current][0])

    return labels


def data_recorder(filename):

    pygame.init()
    screen = pygame.display.set_mode((740, 480))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYUP:
                pass

        screen.fill((0, 0, 0))
        pygame.display.update()
    # sample_rate = 44100
    # chunk = 1024
    # channels = 1
    # seconds = 3
    # rec = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=channels)
    # sd.wait()
    # write('output.wav', sample_rate, rec)
