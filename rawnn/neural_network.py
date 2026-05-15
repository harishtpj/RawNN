# Neural Network implementation
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def build(self, ip_shape):
        for layer in self.layers:
            layer.build(ip_shape)
            ip_shape = layer.get_output_shape()

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, XTrain, YTrain, epochs = 10000, learn_rate = 0.1, op_freq = 100):
        self.build(XTrain.shape[1:])
        for epoch in range(epochs):
            totalErr = 0

            for x, y in zip(XTrain, YTrain):
                output = self.predict(x)
                gradient = -2 * (y - output)

                for layer in reversed(self.layers):
                    gradient = layer.backprop(learn_rate, gradient)

            totalErr += ((output - y) ** 2).sum()

            if epoch % op_freq == 0:
                print(f"Epoch: {epoch}, Error: {totalErr / len(XTrain) * 100:.2f}%")

