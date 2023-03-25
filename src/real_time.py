import noisereduce
import numpy as np
import pygame
import pyaudio
import src.data_engineering.spectrogram as sp

from scipy.io.wavfile import write


def detection(model, scaler, chunk_size=352, input_size=40):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((740, 480))
    font = pygame.font.SysFont(None, 50)
    p = pyaudio.PyAudio()
    fs = 44100

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    samples = np.array([])
    state = 'in'

    while True:
        data = np.frombuffer(stream.read(1024, exception_on_overflow=False), dtype=np.int16)
        samples = np.append(samples, data)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stream.stop_stream()
                stream.close()
                p.terminate()
                write('test.wav', fs, samples)
                return
        if samples.shape[0] > chunk_size * (input_size + 200):
            clean = noisereduce.reduce_noise(samples, 44100)
            last_frame = abs(np.fft.rfft(clean[len(clean) - sp.CHUNK_SIZE:]))[:160]

            last_frame_std = scaler.transform(last_frame.reshape(-1,1).T)
            state = model.predict(last_frame_std)


        screen.fill((0, 0, 0))
        text = font.render('teraz wdychasz' if state == 'in' else 'teraz wydychasz', True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()