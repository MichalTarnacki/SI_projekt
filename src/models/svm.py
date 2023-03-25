import math

import numpy as np
import src.models.dataset as dataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from joblib import dump


class SVM:
    def __init__(self, decisions):
        self.w = np.array([])
        self.C = 10000
        self.kappa = 0.000001
        self.epochs = 100
        self.decisions = {decisions[0]: 1, decisions[1]: -1}

    def fit(self, x_train, y_train):
        y = self.transform_to_binary(y_train)
        x = self.transform_to_extended(x_train)
        self.w = np.zeros(x_train.shape[1] + 1)

        for i in range(0, self.epochs):
            X, Y = shuffle(x, y)
            # self.w = self.w - self.kappa * self.compute_gradient(self.w, x, y)
            for ind, xi in enumerate(X):
                self.w = self.w - (self.kappa * self.compute_gradient(self.w, xi, Y[ind]))


            print(f'{i}: {self.compute_cost(self.w, x, y)}')

        print(self.compute_cost(self.w, x, y))

    def transform_to_binary(self, y):
        return [self.decisions[t] for t in y]

    def transform_to_extended(self, x):
        return np.array([self.extended(xi) for xi in x])

    def extended(self, x):
        return np.append(x, 1)

    def compute_cost(self, w, x, y):
        N = x.shape[0]
        distances = 1 - y * (x @ w)
        distances[distances < 0] = 0
        return 1/2 * w.T @ w + self.C * (np.sum(distances)/N)

    def compute_gradient(self, w, x, y):
        if type(y) == np.int:
            x = np.array([x])
            y = np.array([y])

        distances = 1 - y * (x @ w)
        grad = np.zeros(w.shape)

        for i, d in enumerate(distances):
            if max(0, d) == 0:
                grad += w
            else:
                grad += w - self.C * y[i] * x[i]
        grad /= x.shape[0]
        return grad

    def predict(self, x):
        xe = self.extended(x)
        pred = 1 if self.w.T @ xe else -1
        for k in self.decisions:
            if self.decisions[k] == pred:
                return k
        return 0


def library_svm_train(filenames, modelname):
    x_train, y_train = dataset.build_train(filenames)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = SVC(kernel='linear', verbose=1)
    clf.fit(x_train_std, y_train)
    dump(clf, f'media/models/{modelname}.joblib')
    dump(scaler, f'media/models/{modelname}_scaler.joblib')

    return clf, scaler


def transform_to_binary(y):
    decisions = {'in': 1, 'out': -1}

    return [decisions[t] for t in y]

def transform_to_extended(x):
    return np.array([extended(xi) for xi in x])

def extended(x):
    return np.append(x, 1)


def svm_train(filenames, modelname):
    x_train, y_train = dataset.build_train(filenames)
    scaler = StandardScaler()




    # x_train.insert(loc=len(x_train.columns), column='intercept', value=1)
    x_avg = []

    for x in x_train:
        av = []
        for i in range(0, 160, 16):
            av.append(np.average(x[i:i+10]))
        x_avg.append(av)

    x_train = scaler.fit_transform(x_train)
    x_train = transform_to_extended(x_train)
    y_train = transform_to_binary(y_train)

    sgd(x_train, y_train)

    # x_train_std = scaler.fit_transform(x_train)
    # #
    # #
    # #
    # clf = SVM(['in', 'out'])
    # clf.fit(x_train_std, y_train)
    # dump(clf, f'media/models/{modelname}.joblib')
    # dump(scaler, f'media/models/{modelname}_scaler.joblib')

    # return clf, scaler

regularization_strength = 10000
learning_rate = 0.000001

def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


# I haven't tested it but this same function should work for
# vanilla and mini-batch gradient descent as well
def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.int:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    return dw


def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights
