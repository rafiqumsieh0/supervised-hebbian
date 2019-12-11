import numpy as np
import math


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def tanh(z, c):
    return (math.exp(c*z) - math.exp(-c*z))/(math.exp(c*z) + math.exp(-c*z))


def modified_tanh(z, c=1.0):
    if z <= 0:
        return 0
    else:
        return tanh(z, c)


def relu(z):
    if z <= 0:
        return 0
    else:
        return z


def relu_tanh(z, c):
    if z <= 0:
        return 0
    elif 0 < z < 1:
        return relu(z)
    elif z >= 1:
        return tanh(z, c)