import src.signal_utils as su


if __name__ == '__main__':
    x, y = su.wav_to_sample_xy('./media/output.wav')
    su.debug_plot(x, y)
    x, y = su.to_dominant_freq(x, y)
    su.debug_plot(x, y)
    su.debug_plot(x, su.signal_clean(y))
