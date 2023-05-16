import math
from abc import ABC, abstractmethod

import numpy as np

import macros
import src.models.dataset as dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from joblib import dump

import numpy as np


class SVM:

    def __init__(self, C=100, batch_size=200, learning_rate=0.0001, epochs=200):
        self.C = C
        self.w = []
        self.b = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def hingeloss(self, w, b, x, y):
        reg = 0.5 * (w * w)
        for i in range(x.shape[0]):
            opt_term = y[i] * ((np.dot(w, x[i])) + b)
            loss = reg + self.C * max(0, 1 - opt_term)

        return loss[0][0]

    def fit(self, X, Y):
        epochs = self.epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate

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


class SVMWrapper(ABC):

    def __init__(self):
        self.svm = SVM()

    @abstractmethod
    def select_key_frequencies(self, X):
        pass

    @staticmethod
    def to_string(prediction):
        if prediction == 1 or prediction == -1 or prediction == 0:
            return "in" if prediction == 1 else "out"
        return prediction

    def fit(self, X, Y):
        modified_X = self.select_key_frequencies(X)
        self.svm.fit(modified_X, Y)

    def predict(self, X):
        modified_X = self.select_key_frequencies(X)
        return SVMWrapper.to_string(self.svm.predict(modified_X))


class StandardScalerIgnorePreviousState(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X[:, :-1])
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[:, :-1])
        return np.concatenate((X_head, X[:, -1:]), axis=1)


class MouthOutSVMWrapper(SVMWrapper):
    def select_key_frequencies(self, X):
        return np.array([np.concatenate([x[:169], [x[len(x) - 1]]]) for x in X])


class NoseOutSVMWrapper(SVMWrapper):
    def select_key_frequencies(self, X):
        return np.array([np.concatenate([x[:371], [x[len(x) - 1]]]) for x in X])


class MouthOutLoudonlySVMWrapper(MouthOutSVMWrapper):
    def __init__(self):
        super().__init__()
        self.svm = SVM(C=1, learning_rate=0.001, batch_size=1)


def transform_to_binary(y):
    decisions = {'in': 1, 'out': -1}

    return [decisions[t] for t in y]


def library_svm_train(filenames, modelname):
    x_train, y_train = dataset.build(filenames, macros.train_path)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = SVC(kernel='linear', verbose=1)
    clf.fit(x_train_std, y_train)
    dump(clf, f'media/models/{modelname}.joblib')
    dump(scaler, f'media/models/{modelname}_scaler.joblib')

    return clf, scaler


def svm_train_basic(filenames, modelname):
    x_train, y_train, chunk_size = dataset.build(filenames, macros.train_path)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    y_train = transform_to_binary(y_train)
    clf = SVM()
    clf.fit(x_train_std, y_train)
    dump(clf, f'{macros.model_path}{modelname}.joblib')
    dump(scaler, f'{macros.model_path}{modelname}_scaler.joblib')

    return clf, scaler


def svm_train_with_previous_state(filenames, modelname, mouth_out=True, loudonly=False):
    if loudonly:
        x_train, y_train, chunk_size = dataset.build_loudonly(filenames, macros.train_path, previous_state=True)
    else:
        x_train, y_train, chunk_size = dataset.build(filenames, macros.train_path, previous_state=True)

    scaler = StandardScalerIgnorePreviousState()
    x_train_std = scaler.fit(x_train).transform(x_train)
    y_train = transform_to_binary(y_train)

    if loudonly:
        clf = MouthOutLoudonlySVMWrapper()
    else:
        if mouth_out:
            clf = MouthOutSVMWrapper()
        else:
            clf = NoseOutSVMWrapper()

    clf.fit(x_train_std, y_train)
    dump(clf, f'{macros.model_path}{modelname}.joblib')
    dump(scaler, f'{macros.model_path}{modelname}_scaler.joblib')

    return clf, scaler
