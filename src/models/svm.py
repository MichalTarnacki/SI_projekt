import math

import numpy as np

import macros
import src.models.dataset as dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
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
    x_train, y_train = dataset.build(filenames, macros.train_path)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = SVC(kernel='linear', verbose=1)
    clf.fit(x_train_std, y_train)
    dump(clf, f'media/models/{modelname}.joblib')
    dump(scaler, f'media/models/{modelname}_scaler.joblib')

    return clf, scaler


def svm_train_basic(filenames, modelname):
    x_train, y_train = dataset.build(filenames, macros.train_path)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    y_train = transform_to_binary(y_train)
    clf = SVM()
    clf.fit(x_train_std, y_train)
    dump(clf, f'{macros.model_path}{modelname}.joblib')
    dump(scaler, f'{macros.model_path}{modelname}_scaler.joblib')

    return clf, scaler


def svm_train_with_previous_state(filenames, modelname, softmax=False, with_bg=False):
    x_train, y_train = dataset.build(filenames, macros.train_path, True, with_bg)

    scaler = StandardScalerIgnorePreviousState()
    x_train_std = scaler.fit(x_train).transform(x_train)
    y_train = transform_to_binary(y_train)

    if softmax:
        clf = SoftmaxSVM()
    else:
        clf = SVM()

    clf.fit(x_train_std, y_train)
    dump(clf, f'{macros.model_path}{modelname}_prevstate.joblib')
    dump(scaler, f'{macros.model_path}{modelname}_prevstate_scaler.joblib')

    return clf, scaler


def svm_train_with_previous_state_with_wide_spectro(filenames, modelname, softmax=False, with_bg=False):
    x_train, y_train, chunk_size = dataset.build_wide_spectro(filenames, macros.train_path, True, with_bg)

    scaler = StandardScalerIgnorePreviousState()
    x_train_std = scaler.fit(x_train).transform(x_train)
    y_train = transform_to_binary(y_train)

    if softmax:
        clf = SoftmaxSvmWithWideSpectro()
    else:
        clf = SVM()

    clf.fit(x_train_std, y_train)
    dump(clf, f'{macros.model_path}{modelname}_prevstate.joblib')
    dump(scaler, f'{macros.model_path}{modelname}_prevstate_scaler.joblib')

    return clf, scaler


class StandardScalerIgnorePreviousState(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X[:, :-1])
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[:, :-1])
        return np.concatenate((X_head, X[:, -1:]), axis=1)


class SoftmaxSVM:
    def __init__(self):
        self.in_SVM = SVM()
        self.out_SVM = SVM()
        self.classifier_SVM = SVM()

    def fit(self, X, Y):
        modified = np.array([np.concatenate([x[:30], x[80:]]) for x in X])
        # self.in_SVM.fit(np.concatenate([X[:, :30], X[:, 50:]]), Y)
        self.in_SVM.fit(modified, Y)
        return (self.in_SVM.w, self.in_SVM.b)
        # print("breathe in SVM")
        # Xin = X[:, 70:]
        # self.in_SVM.fit(Xin, Y)
        # print("breathe out SVM")
        # Xout = np.concatenate([X[:, :30], [X[:, 160]]])
        # self.out_SVM.fit(Xout, Y)
        # # self.out_SVM.fit(np.concatenate([X[,:80], [X[160]]]), Y)
        # print("classifier SVM")
        # X_new = np.array([self.to_softmax(x) for x in X])
        # self.classifier_SVM.fit(X_new, Y, batch_size=100)

        # return (self.in_SVM.w, self.in_SVM.b), (self.out_SVM.w, self.out_SVM.b)

    def to_softmax(self, X):
        prediction_in = np.dot(X, self.in_SVM.w[0]) + self.in_SVM.b  # 1 - in, -1 - not in
        prediction_out = -1 * (np.dot(X, self.out_SVM.w[0]) + self.out_SVM.b)  # 1 - out, -1 - not out

        e_in = math.exp(prediction_in - max(prediction_in, prediction_out))
        e_out = math.exp(prediction_out - max(prediction_in, prediction_out))
        su = e_in + e_out
        return e_in/su, e_out/su

    def predict(self, X):
        modified = np.array([np.concatenate([x[:30], x[80:]]) for x in X])
        # modified = X
        return self.in_SVM.predict(modified)
        # softmax_in, softmax_out = self.to_softmax(X)
        # prediction = np.dot(np.array([softmax_in, softmax_out]), self.classifier_SVM.w[0]) + self.classifier_SVM.b

        # return 'in' if np.sign(prediction) == 1 else 'out'


class SoftmaxSvmWithWideSpectro:
    def __init__(self):
        self.in_SVM = SVM()
        self.out_SVM = SVM()
        self.classifier_SVM = SVM()

    def fit(self, X, Y):
        modified = np.array([np.concatenate([x[:371], [x[len(x) - 1]]]) for x in X])
        self.in_SVM.fit(modified, Y)
        return (self.in_SVM.w, self.in_SVM.b)
        # print("breathe in SVM")
        # Xin = X[:, 70:]
        # self.in_SVM.fit(Xin, Y)
        # print("breathe out SVM")
        # Xout = np.concatenate([X[:, :30], [X[:, 160]]])
        # self.out_SVM.fit(Xout, Y)
        # # self.out_SVM.fit(np.concatenate([X[,:80], [X[160]]]), Y)
        # print("classifier SVM")
        # X_new = np.array([self.to_softmax(x) for x in X])
        # self.classifier_SVM.fit(X_new, Y, batch_size=100)

        # return (self.in_SVM.w, self.in_SVM.b), (self.out_SVM.w, self.out_SVM.b)

    def to_softmax(self, X):
        prediction_in = np.dot(X, self.in_SVM.w[0]) + self.in_SVM.b  # 1 - in, -1 - not in
        prediction_out = -1 * (np.dot(X, self.out_SVM.w[0]) + self.out_SVM.b)  # 1 - out, -1 - not out

        e_in = math.exp(prediction_in - max(prediction_in, prediction_out))
        e_out = math.exp(prediction_out - max(prediction_in, prediction_out))
        su = e_in + e_out
        return e_in/su, e_out/su

    def predict(self, X):
        modified = np.array([np.concatenate([x[:371], [x[len(x) - 1]]]) for x in X])
        return self.in_SVM.predict(modified)
        # softmax_in, softmax_out = self.to_softmax(X)
        # prediction = np.dot(np.array([softmax_in, softmax_out]), self.classifier_SVM.w[0]) + self.classifier_SVM.b

        # return 'in' if np.sign(prediction) == 1 else 'out'