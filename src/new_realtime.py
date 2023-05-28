import random
import threading
import numpy as np
import pygame
import pyaudio
from keras import models
from TensorFlow import TensorFlow
import macros
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
    saved_chunks = 30

    pygame.init()
    pygame.font.init()

    width = 740
    height = 480
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont(None, 50)

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
    active = False
    active_s = False
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                contin = False
                tr.join()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                active = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                active = False if active_s else active
                active_s = True if not active_s else False
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
        if saved.__len__() >= saved_chunks * CHUNK:
            commands, pred = TensorFlow.new_predict(model, saved)

            color = (0, 0, 255)

            if pred[list(commands).index("in")] > wdech_s:  # and np.average(saved) > avg:  # and pred[1]<10:
                sm.feed("in")
            elif pred[list(commands).index("out")] > wydech_s:  # and np.average(saved) > avg:  # and pred[0]<10:
                sm.feed("out")
            else:
                sm.feed("silence")


            match sm.get_state():
                case 'in':
                    color = (255, 0, 0)
                    radius -= 1 if radius > 10 else 0
                    text = font.render('teraz wdychasz', True, (255, 255, 255))
                    wdechy += 1
                case 'out':
                    color = (0, 255, 0)
                    radius += 1 if radius < 250 else 0
                    text = font.render('teraz wydychasz', True, (0, 255, 255))
                    wydechy += 1

            text2 = font.render(f'wydech_s {int(100 * wydech_s)}%', True, (0, 255, 255))
            text3 = font.render(f'wdech_s {int(100 * wdech_s)}%', True, (0, 255, 255))
            if active_s:
                if R == radius:
                    points +=1
                    R = random.randint(20, 240)

                if R > radius:
                    pygame.draw.circle(screen, (124, 143, 31), (width / 2, height / 2), R)
                pygame.draw.circle(screen, color, (width/2, height/2), radius)
                if R<= radius:
                    pygame.draw.circle(screen, (124, 143, 31), (width / 2, height / 2), R)



                if wdechy+wydechy+inne > 0:
                    text4 = font.render(f'wdechy {round(100 * w_p)}%'
                                        f'wydechy {round(100 * wy_p)}%'
                                        f'inne {round(100 * i_p)}%', True, (0, 255, 255))
                    screen.blit(text4, (0, 120))
                text5 = font.render(f'punkty {points}', True, (0, 255, 255))
                screen.blit(text5, (0, 90))
            else:
                pygame.draw.circle(screen, color, (width / 2, height / 2), radius)
        screen.blit(text, (0, 0))
        if active:
            screen.blit(text2, (0, 30))
            screen.blit(text3, (0, 60))
        pygame.display.update()