from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from joblib import dump
import src.models.dataset as dataset
import macros
import numpy as np
from tensorflow import keras

class NNWrapper:

    def __init__(self, model):
        self.model = model

    def select_key_frequencies(self, X):
        return np.array([np.concatenate([x[:371], [x[len(x) - 1]]]) for x in X])

    @staticmethod
    def to_string(prediction):
        if prediction == 1 or prediction == -1 or prediction == 0:
            return "in" if prediction == 1 else "out"
        return prediction

    def fit(self, X, Y):
        modified_X = self.select_key_frequencies(X)
        Y = Y.reshape(Y.shape[0], 1)
        self.model.fit(modified_X, Y, epochs=10, batch_size=100)

    def predict(self, X):
        modified_X = self.select_key_frequencies(X)
        return NNWrapper.to_string(self.model.predict(modified_X)[0][0])

def transform_to_binary(y):
    decisions = {'in': 1, 'out': -1}

    return [decisions[t] for t in y]

class StandardScalerIgnorePreviousState(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X[:, :-1])
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[:, :-1])
        return np.concatenate((X_head, X[:, -1:]), axis=1)


def build(filenames):
    x_train, y_train, chunk_size = dataset.build_loudonly(filenames, macros.train_path, previous_state=True)

    scaler = StandardScalerIgnorePreviousState()
    x_train_std = scaler.fit(x_train).transform(x_train)
    y_train = transform_to_binary(y_train)

    model = keras.Sequential([
        keras.layers.Conv1D(32,kernel_size=3, activation='relu', padding='same', input_shape=(372, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(),
        keras.layers.Conv1D(64,kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(),
        keras.layers.Conv1D(128,kernel_size=3,activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(1, activation='tanh')
    ])

    x_train_std = np.array(x_train_std)
    y_train = np.array(y_train)
    y_train.resize(y_train.shape[0])
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    wrapper = NNWrapper(model)
    wrapper.fit(x_train_std, y_train)

    wrapper.predict(x_train_std[0].reshape(1, -1))
    dump(wrapper, f'{macros.model_path}neural_network.joblib')
    dump(scaler,  f'{macros.model_path}neural_network_scaler.joblib')