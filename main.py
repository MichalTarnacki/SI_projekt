import src.svm as svm
import src.signal_utils as su
import src.data_utils as du

from sklearn.preprocessing import StandardScaler


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
    x_train, y_train = svm.create_train_dataset(['e1', 'e2', 'e3', 'e4', 'e5', 'e7'])
    sv, scaler = svm.train(x_train, y_train)

    l, t, f = du.file_with_labels(file_csv=f'media/train/e7.csv', file_wav=f'media/train/e7.wav')
    su.debug_plot_marked(t, svm.filter_NaN(f), l)
    t, f, l, fp = svm.prepare(t, l, f)

    pred = sv.predict(StandardScaler().fit_transform(fp))
    su.debug_plot_marked(t, f, pred)

