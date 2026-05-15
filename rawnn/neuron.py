# Simple Neuron Implmentation
import numpy as np

class Neuron:
    def __init__(self, n_inputs, activation):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = np.random.randn()
        self.afn = activation

        self.__last_ip = None
        self.__last_total = None
        self.__last_op = None

    def feedforward(self, inputs):
        self.__last_ip = inputs
        self.__last_total = inputs @ self.weights + self.bias
        self.__last_op = self.afn(self.__last_total)
        return self.__last_op

    def backprop(self, learn_rate, err_grad):
        delta = err_grad * self.afn.derv(self.__last_total)
        ip_error = delta * self.weights
        self.weights -= learn_rate * delta * self.__last_ip
        self.bias -= learn_rate * delta
        return ip_error

