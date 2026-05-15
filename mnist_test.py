# This code requires scikit-learn for mnist dataset
# The model itself is independent of the actual tf modules
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from rawnn.neural_network import NeuralNetwork
from rawnn import layer
from rawnn.activation_functions import Sigmoid

if __name__ == "__main__":
    digits = load_digits()
    X, Y = digits.images, digits.target
    X = X.reshape(-1, 8, 8, 1) / 16.0
    Y_one_hot = np.eye(10)[Y]

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y_one_hot, test_size = 0.2)

    model = NeuralNetwork([
        layer.ConvNeurons(n_filters=2, kernel_size=3, activation = Sigmoid()),
        layer.Flatten(),
        layer.Neurons(10, activation = Sigmoid()),
    ])

    print("Starting MNIST (lite) Training: ")
    model.train(xTrain, yTrain, epochs = 50, learn_rate = 0.01, op_freq = 5)

    test_img = xTest[0]
    pred = model.predict(test_img)
    print(f"Predicted digit: {np.argmax(pred)} - Actual: {np.argmax(yTest[0])}")

