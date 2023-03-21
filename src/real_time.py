import pygame
import src.signal_utils as su

from sklearn.preprocessing import StandardScaler


def real_time_detection(model, chunk_size=352, input_size=40):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((740, 480))
    font = pygame.font.SysFont(None, 50)
    fs = 44100
    state = 'in'
    samples = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        if len(samples) > chunk_size * input_size:
            t, f = su.to_dominant_freq(fs, samples)
            f = su.signal_clean(f)
            f = StandardScaler().fit_transform(f)
            input = f[len(f)-input_size:]
            state = model.predict(input)

        screen.fill((0, 0, 0))
        text = font.render('teraz wdychasz' if state == 'in' else 'teraz wydychasz', True, (255, 255, 255))
        screen.blit(text, (0, 0))
        pygame.display.update()
