import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def prepare(t, labels, f):
    dimensions = 10
    return \
        t[:len(t) - dimensions], \
        f[:len(f) - dimensions], \
        labels[:len(labels) - dimensions], \
        [f[i:i + dimensions] for i in range(len(f) - dimensions)]


def train(x_train, y_train):
    clf = SVC(kernel='linear', verbose=1)
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf.fit(x_train_std, y_train)
    return clf
