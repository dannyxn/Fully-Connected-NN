import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prim(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.where(x >= 0, x, 0)


def relu_prim(x):
    return np.where(x >= 0, 1, 0)


activations_mapping = {"tanh": (tanh, tanh_prim), "relu": (relu, relu_prim)}
