# This code requires scikit-learn for utilizing the
# iris dataset.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from rawnn.neural_network import NeuralNetwork 
from rawnn import layer
import rawnn.activation_functions as afns

iris = load_iris()
X, Y = iris.data, iris.target

X = X / np.max(X, axis=0)
Y_one_hot = np.eye(3)[Y]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y_one_hot, test_size=0.2)

model = NeuralNetwork([
    layer.Neurons(8),
    layer.Neurons(3, activation = afns.Sigmoid()),
])

print("Training on Iris dataset:")
model.train(xTrain, yTrain, epochs = 500, learn_rate = 0.05)

crt = 0
for x, y_true in zip(xTest, yTest):
    pred = model.predict(x)
    if np.argmax(pred) == np.argmax(y_true):
        crt += 1

print(f"Test Accuracy: {crt / len(yTest) * 100:.2f}%")
