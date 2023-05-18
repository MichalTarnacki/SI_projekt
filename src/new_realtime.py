import copy
import random
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
from keras import models
from scipy.signal import savgol_filter

from TensorFlow import TensorFlow
import macros
import src.data_engineering.spectrogram as sp

from pydub import effects
from scipy.io.wavfile import write
from src.real_time import StateMachine

def new_realtime():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    model = models.load_model(f'{macros.model_path}tensorflow')
    saved = []
    p = None
    stream = None
    contin = True
    saved_chunks = 40

    pygame.init()
    pygame.font.init()

    width = 740
    height = 480
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont(None, 50)
    run_thread = True

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

    tr = threading.Thread(target=record_thread)
    tr.start()

    radius = 100
    wdech_s = 0.99
    wydech_s = 0.7
    minute = pygame.USEREVENT + 1
    pygame.time.set_timer(minute, 60000)
    wdechy = 0
    wydechy = 0
    inne = 0
    w_p = 0
    i_p = 0
    wy_p = 0
    R = random.randint(20, 240)
    points = 0
    avg = 50
    sm = StateMachine()
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run_thread = False
                tr.join()
                stream.stop_stream()
                stream.close()
                p.terminate()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                wydech_s += 0.01
            if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                wydech_s -= 0.01
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                wdech_s -= 0.01
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                wdech_s += 0.01
            if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                avg += 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                avg -= 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                avg -= 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                avg += 1
            if event.type == minute:
                w_p = wdechy / (wdechy + wydechy + inne)
                wy_p = wydechy / (wdechy + wydechy + inne)
                i_p = inne / (wdechy + wydechy + inne)
                wdechy = 0
                wydechy = 0
                inne = 0
        screen.fill((0, 0, 0))
        text = font.render('nic', True, (255, 255, 255))
        pred = [0,0]
        if saved.__len__() >= saved_chunks * CHUNK:
            commands, pred = TensorFlow.new_predict(model, saved)

            color = (0, 0, 255)

            if pred[2] > wdech_s:  # and np.average(saved) > avg:  # and pred[1]<10:
                sm.feed("in")
            elif pred[1] > wydech_s:  # and np.average(saved) > avg:  # and pred[0]<10:
                sm.feed("out")
            else:
                sm.feed("silence")


            match sm.get_state():
                case 'in':
                    color = (255, 0, 0)
                    radius -= 2 if radius > 10 else 0
                    text = font.render('teraz wdychasz', True, (255, 255, 255))
                    wdechy += 1
                case 'out':
                    color = (0, 255, 0)
                    radius += 1 if radius < 250 else 0
                    text = font.render('teraz wydychasz', True, (0, 255, 255))
                    wydechy += 1

            # if pred[2] > wdech_s: #and np.average(saved) > avg:  # and pred[1]<10:
            #     color = (255, 0, 0)
            #     radius -= 2 if radius > 10 else 0
            #     text = font.render('teraz wdychasz', True, (255, 255, 255))
            #     wdechy += 1
            # elif pred[1] > wydech_s: #and np.average(saved) > avg:  # and pred[0]<10:
            #     color = (0, 255, 0)
            #     radius += 1 if radius < 250 else 0
            #     text = font.render('teraz wydychasz', True, (0, 255, 255))
            #     wydechy += 1
            # else:
            #     inne += 1



            # radius -= 1

            # for x, y in enumerate(last_frame[:-1]):
            #     pygame.draw.line(screen, color, (x*3, 480 - y), (x*3 + 3, 480 - last_frame[x+1]))
            if R == radius:
                points +=1
                R = random.randint(20, 240)

            if R > radius:
                pygame.draw.circle(screen, (124, 143, 31), (width / 2, height / 2), R)
            pygame.draw.circle(screen, color, (width/2, height/2), radius)
            if R<= radius:
                pygame.draw.circle(screen, (124, 143, 31), (width / 2, height / 2), R)


        text2 = font.render(f'wydech_s {int(100*wydech_s)}%', True, (0, 255, 255))
        text3 = font.render(f'wdech_s {int(100*wdech_s)}%', True, (0, 255, 255))
        if wdechy+wydechy+inne > 0:
            text4 = font.render(f'wdechy {round(100 * w_p)}%'
                                f'wydechy {round(100 * wy_p)}%'
                                f'inne {round(100 * i_p)}%', True, (0, 255, 255))
            screen.blit(text4, (0, 90))
        text5 = font.render(f'punkty {points}', True, (0, 255, 255))
        # text6 = font.render(f'avg_p {avg}', True, (0, 255, 255))
        # text7 = font.render(f'avgerasge {np.max(saved) if len(saved) > 0 else 0}', True, (0, 255, 255))
        # text8 = font.render(f'predykcje {pred[0]}, {pred[1]}', True, (0, 255, 255))
        screen.blit(text, (0, 0))
        screen.blit(text2, (0, 30))
        screen.blit(text3, (0, 60))
        screen.blit(text5, (0, 120))
        # screen.blit(text6, (0, 150))
        # screen.blit(text7, (0, 180))
        # screen.blit(text8, (0, 210))
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
                radius -= 2
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
