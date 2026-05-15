# Implementation of some commonly used activation functions
from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def fn(self, x):
        pass

    @abstractmethod
    def derv(self, x):
        pass

    def __call__(self, x):
        return self.fn(x)

class Sigmoid(Activation):
    def fn(self, x):
        return 1 / (1 + np.exp(-x))

    def derv(self, x):
        f = self.fn(x)
        return f * (1 - f)

class ReLU(Activation):
    def fn(self, x):
        return np.maximum(0, x)

    def derv(self, x):
        return (x > 0).astype(float)

class Tanh(Activation):
    def fn(self, x):
        return np.tanh(x)

    def derv(self, x):
        return 1 - self.fn(x) ** 2

