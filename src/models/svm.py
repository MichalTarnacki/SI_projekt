import math

import numpy as np
import src.models.dataset as dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from joblib import dump

import numpy as np


class SVM:

    def __init__(self, C=1.0):
        self.C = C
        self.w = []
        self.b = 0

    def hingeloss(self, w, b, x, y):
        reg = 0.5 * (w * w)
        for i in range(x.shape[0]):
            opt_term = y[i] * ((np.dot(w, x[i])) + b)
            loss = reg + self.C * max(0, 1 - opt_term)

        return loss[0][0]

    def fit(self, X, Y, batch_size=400, learning_rate=0.001, epochs=100):
        number_of_features = X.shape[1]
        number_of_samples = X.shape[0]
        c = self.C

        ids = np.arange(number_of_samples)
        np.random.shuffle(ids)

        w = np.zeros((1, number_of_features))
        b = 0

        for i in range(epochs):
            print(f'{i}: {self.hingeloss(w, b, X, Y)}')

            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c * Y[x] * X[x]
                            gradb += c * Y[x]

                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb

        self.w = w
        self.b = b

        return self.w, self.b

    def predict(self, X):
        prediction = np.dot(X, self.w[0]) + self.b  # w.x + b
        return 'in' if np.sign(prediction) == 1 else 'out'


def transform_to_binary(y):
    decisions = {'in': 1, 'out': -1}

    return [decisions[t] for t in y]


def transform_to_extended(x):
    return np.array([extended(xi) for xi in x])


def extended(x):
    return np.append(x, 1)


def library_svm_train(filenames, modelname):
    x_train, y_train = dataset.build_train(filenames)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = SVC(kernel='linear', verbose=1)
    clf.fit(x_train_std, y_train)
    dump(clf, f'media/models/{modelname}.joblib')
    dump(scaler, f'media/models/{modelname}_scaler.joblib')

    return clf, scaler


def svm_train(filenames, modelname):
    x_train, y_train = dataset.build_train(filenames)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    y_train = transform_to_binary(y_train)
    clf = SVM()
    clf.fit(x_train_std, y_train)
    dump(clf, f'media/models/{modelname}.joblib')
    dump(scaler, f'media/models/{modelname}_scaler.joblib')

    return clf, scaler

