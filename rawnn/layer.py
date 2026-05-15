# Implementation of Various Layers for Neural Network Creation
import numpy as np
from .neuron import Neuron
from .activation_functions import ReLU

class Neurons:
    def __init__(self, n, activation=ReLU()):
        self.n_neurons = n
        self.afn = activation
        self.neurons = []

    def build(self, ip_shape):
        ip_per_neuron = ip_shape[0]
        self.neurons = [Neuron(ip_per_neuron, self.afn) for _ in range(self.n_neurons)]

    def feedforward(self, inputs):
        return np.array([n.feedforward(inputs) for n in self.neurons])

    def backprop(self, learn_rate, op_grads):
        total_ip_grad = np.zeros_like(self.neurons[0].weights)

        for i, neuron in enumerate(self.neurons):
            total_ip_grad += neuron.backprop(learn_rate, op_grads[i])

        return total_ip_grad

    def get_output_shape(self):
        return (self.n_neurons,)

class ConvNeurons:
    def __init__(self, n_filters, kernel_size = 3, activation = ReLU()):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.afn = activation
        self.filters = None
        self.biases = None

    def build(self, ip_shape):
        h, w, d = ip_shape
        self.filters = np.random.randn(self.n_filters, self.kernel_size, self.kernel_size, d) * 0.1
        self.biases = np.zeros(self.n_filters)
        self.out_h, self.out_w = h - self.kernel_size + 1, w - self.kernel_size + 1
        self.output_shape = (self.out_h, self.out_w, self.n_filters)
        self.n_neurons = np.prod(self.output_shape)

    def feedforward(self, inputs):
        if len(inputs.shape) == 2:
            inputs = inputs[:, :, np.newaxis]

        self.__last_ip = inputs
        self.__last_op = np.zeros((self.out_h, self.out_w, self.n_filters))

        for (i, j), window in self.__iterate(inputs):
            self.__last_op[i, j] = np.sum(window * self.filters, axis = (1, 2, 3)) + self.biases

        return self.afn(self.__last_op)

    def backprop(self, learn_rate, grad):
        delta = grad * self.afn.derv(self.__last_op)
        ip_grad = np.zeros_like(self.__last_ip)
        filter_grads = np.zeros_like(self.filters)
        bias_grads = np.zeros_like(self.biases)

        for (i, j), window in self.__iterate(self.__last_ip):
            for f in range(self.n_filters):
                filter_grads[f] += window * delta[i, j, f]
                bias_grads[f] += delta[i, j, f]
                ip_grad[i:i + self.kernel_size, j:j + self.kernel_size] += self.filters[f] * delta[i, j, f]

        self.filters -= learn_rate * filter_grads
        self.biases -= learn_rate * bias_grads

        return ip_grad

    def get_output_shape(self):
        return self.output_shape

    def __iterate(self, inputs):
        for i in range(self.out_h):
            for j in range(self.out_w):
                yield (i, j), inputs[i:i + self.kernel_size, j:j + self.kernel_size]


class MaxPool:
    def __init__(self, size = 2):
        self.size = size
        self.__last_ip = None

    def build(self, ip_shape):
        h, w, d = ip_shape
        self.output_shape = (h // self.size, w // self.size, d)
        self.n_neurons = np.prod(self.output_shape)

    def feedforward(self, inputs):
        self.__last_ip = inputs
        h, w, d = inputs.shape
        output = np.zeros((h // self.size, w // self.size, d))

        for (i, j), window in self.__iterate(inputs):
            output[i, j] = np.max(window, axis=(0, 1))
        
        return output

    def backprop(self, learn_rate, grad):
        h, w, d = self.__last_ip.shape
        ip_grad = np.zeros_like(self.__last_ip)

        for (i, j), window in self.__iterate(self.__last_ip):
            for k in range(d):
                chan_size = window[:, :, k]
                m, n = np.unravel_index(np.argmax(chan_size), chan_size.shape)
                ip_grad[i * self.size + m, j * self.size + n, k] = grad[i, j, k]

        return ip_grad

    def __iterate(self, inputs):
        h, w, _ = inputs.shape

        for i in range(h // self.size):
            for j in range(w // self.size):
                rs, re = i * self.size, (i + 1) * self.size
                cs, ce = j * self.size, (j + 1) * self.size
                yield (i, j), inputs[rs:re, cs:ce]

    def get_output_shape(self):
        return self.output_shape


class Flatten:
    def __init__(self):
        self.input_shape = None

    def build(self, ip_shape):
        self.input_shape = ip_shape
        self.n_neurons = np.prod(ip_shape)

    def feedforward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.flatten()

    def backprop(self, learn_rate, grad):
        return grad.reshape(self.input_shape)

    def get_output_shape(self):
        return (self.n_neurons,)
