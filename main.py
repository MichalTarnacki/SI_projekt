import src.signal_utils as su
import src.data_utils as du

if __name__ == '__main__':
    # x, y = su.wav_to_sample_xy('./media/output.wav')
    # su.debug_plot(x, y)
    # t, f = su.to_dominant_freq(x, y, chunk_size=128)
    # labels = du.label_to_sample('./media/output.csv', t)
    #
    # # su.debug_plot_marked(t, su.signal_clean(f), labels)
    # # su.debug_plot(t, f)
    # su.debug_plot(t, su.signal_clean(f))
    du.data_recorder('test.wav')
