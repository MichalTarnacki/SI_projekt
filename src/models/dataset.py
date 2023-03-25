import src.data_engineering.data_utils as du
import src.data_engineering.spectrogram as sp


def build_train(filenames):
    return build(filenames, 'train')


def build_test(filenames):
    return build(filenames, 'test')


def build(filenames, dir):
    x_set = []
    y_set = []
    for filename in filenames:
        file_csv = f'media/{dir}/{filename}.csv'
        file_wav = f'media/{dir}/{filename}.wav'
        x, y = du.wav_to_sample_xy(file_wav)
        timestamps, frames = sp.to_spectro(y, sp.SAMPLE_FREQ)
        labels = sp.spectro_labeled(file_csv, timestamps)
        x_set += frames
        y_set += labels

    return x_set, y_set
