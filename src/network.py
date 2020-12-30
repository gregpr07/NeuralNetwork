import numpy as np
from kernel import kernels
from cost import cost


class Network():
    def __init__(self, cost='meanSquare'):

        # L weight matrix connecrs L and L+1 layer
        # dimensions of each are dim(L-1),dim(L)
        self.weights = []

        # layers represents z values
        self.layers = []

        # each layer has a specific kernel
        # None is set so that layer and kernel have the same index
        self.kernels = [None, ]

        # same dimensions as the layer
        # None is set so that layer and kernel have the same index
        self.biases = [None, ]

        self.costFunction = cost

        # naming from https://en.wikipedia.org/wiki/Backpropagation
        self.deltas = [None, ]
        self.weightGrads = []

    def __repr__(self):
        return str((self.layers))

    def addInputLayer(self, dimension):
        self.layers.append(np.zeros(dimension))

    def addLayer(self, dimension, kernel):
        layer_id = len(self.layers)

        if not layer_id:
            raise Exception('Input layer not defined.')

        if not kernel in kernels:
            raise Exception(
                f'Kernel not available. Choose from the following: {list(kernels.keys())}')

        self.layers.append(np.zeros(dimension))
        self.biases.append((np.random.rand(dimension)-0.5)*2)
        self.kernels.append(kernel)
        self.deltas.append(False)
        self.weightGrads.append(False)

        self.weights.append((np.random.rand(
            self.layers[layer_id - 1].size, dimension)-0.5)*2)

    def setInput(self, arr):
        arr = np.array(arr)
        if not arr.size == self.layers[0].size:
            raise Exception('First layer and input size do not match')

        self.layers[0] = arr

    def outputLayer(self, i):
        if i == 0:
            return self.layers[i]

        return kernels[self.kernels[i]](self.layers[i] + self.biases[i])

    def propagateForwards(self, i):
        self.layers[i+1] = np.matmul(self.outputLayer(i), self.weights[i])

    def calculateOutput(self, inp):
        self.setInput(inp)
        num_layers = len(self.layers)-1
        for i in range(num_layers):
            self.propagateForwards(i)

        return self.outputLayer(num_layers)

    def calculateCost(self, x, y):
        pred_y = self.calculateOutput(x)
        return cost[self.costFunction](y, pred_y)

    def propagateBackwards(self, x, y):
        pred_y = self.calculateOutput(x)

        gradC = cost[self.costFunction](y, pred_y, derivative=True)

        self.deltas[-1] = kernels[self.kernels[-1]](gradC, derivative=True)

        for i in reversed(range(1, len(self.deltas) - 1)):
            prod = np.matmul(self.weights[i], self.deltas[i+1])
            self.deltas[i] = kernels[self.kernels[i]](prod, derivative=True)

        for i in range(len(self.weightGrads)):
            dC_dWl = np.outer(self.deltas[i + 1], self.outputLayer(i))
            self.weightGrads[i] = dC_dWl.T

    def train(self, alpha=0.1):
        for i in range(len(self.weightGrads)):
            self.weights[i] -= self.weightGrads[i] * alpha

        # for i in reversed(range(1, len(self.biases))):
        #    self.biases[i] -= self.deltas[i] * alpha

    def visualizeTrain(self, x, y, length, alpha=0.1):
        for i in range(length):
            self.propagateBackwards(x, y)
            self.train(alpha=alpha)
            print(self.calculateCost(x, y))
