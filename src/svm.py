import math
import numpy as np
import src.data_utils as du
import src.signal_utils as su

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def filter_NaN(x):
    avg = np.average([0 if math.isnan(y) else y for y in x])
    return [avg if math.isnan(y) else y for y in x]


# reshape all data and return it back
def prepare(t, labels, f):
    f = filter_NaN(f)
    t = filter_NaN(t)

    dimensions = 225
    return \
        t[:len(t) - dimensions], \
        f[:len(f) - dimensions], \
        labels[:len(labels) - dimensions], \
        [f[i:i + dimensions] for i in range(len(f) - dimensions)]


def create_train_dataset(train_files, train_dir='media/train'):
    x_train = []
    y_train = []

    for tag in train_files:
        l, t, f = du.file_with_labels(
            file_csv=f'{train_dir}/{tag}.csv',
            file_wav=f'{train_dir}/{tag}.wav')
        su.debug_plot_marked(t, filter_NaN(f), l)
        t, f, l, fp = prepare(t, l, f)
        x_train += fp
        y_train += l

    return x_train, y_train


def train(x_train, y_train):
    clf = SVC(kernel='linear', verbose=1)
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf.fit(x_train_std, y_train)
    return clf, scaler
