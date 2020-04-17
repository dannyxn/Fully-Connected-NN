import numpy as np
import abc
import sys
from activations import *


class Layer(abc.ABC):
    def __init__(self):
        self.input = None
        self.output = None

    def forward_prop(self, input_data):
        raise NotImplementedError

    def backward_prop(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.random((input_size, output_size)) - 0.5
        self.bias = np.random.random((1, output_size)) - 0.5

    def forward_prop(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return input_data @ self.weights + self.bias

    def backward_prop(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = output_error @ self.weights.T
        weights_error = self.input.T @ output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation: str):
        self.activation, self.activation_prim = activations_mapping.get(activation, None)
        if not self.activation:
            sys.exit()

    def forward_prop(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_prop(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return self.activation_prim(self.input) * output_error

