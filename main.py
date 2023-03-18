import src.signal_utils as su
import src.data_utils as du


def record(file):
    file_csv = f'media/train/{file}.csv'
    file_wav = f'media/train/{file}.wav'
    du.data_recorder(file_csv=file_csv, file_wav=file_wav)
    l, t, f = du.file_with_labels(file_csv=file_csv, file_wav=file_wav)
    su.debug_plot(t, f)
    su.debug_plot_marked(t, f, l)


def debug_recorded(file):
    l, t, f = du.file_with_labels(file_csv=f'media/train/{file}.csv', file_wav=f'media/train/{file}.wav')
    su.debug_plot(t, f)
    su.debug_plot_marked(t, f, l)


if __name__ == '__main__':
    # debug_recorded('e2')
    record('e3')
