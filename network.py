import numpy as np
from layers import *
from time import perf_counter

class Network(object):
    def __init__(self, epochs: int, learning_rate: float):
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_pred) ** 2)

    def loss_prim(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        n_samples = len(input_data)
        result = []
        for i in range(n_samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_prop(output)

            result.append(output)

        return result

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n_samples = len(y_train)

        for epoch in range(self.epochs):
            error_sum = 0
            start = perf_counter()
            for j in range(n_samples):
                output = X_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                error_sum += self.loss(y_train[j], output)

                error = self.loss_prim(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, self.learning_rate)

            error_sum /= n_samples
            print('Epoch {}/{}  MSE={} Time={}s' .format(epoch + 1, self.epochs, error_sum, perf_counter() - start))

