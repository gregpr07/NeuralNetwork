import numpy as np
from kernel import kernels
from cost import cost
from tqdm import tqdm


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
        self.biasGrads = [None, ]

        self.costFunction = cost

        # naming from https://en.wikipedia.org/wiki/Backpropagation
        self.deltas = [None, ]
        self.weightGrads = []

        # this will be matrices of size (dim(Al),batch)
        self.Al_Caches = []

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
        self.biasGrads.append(False)

        self.weights.append((np.random.rand(
            self.layers[layer_id - 1].size, dimension) - 0.5) * 2)

    def setInput(self, arr):
        arr = np.array(arr)
        if not arr.size == self.layers[0].size:
            raise Exception('First layer and input size do not match')

        self.layers[0] = arr

    def kernelFunction(self, x, i, derivative=False):
        return kernels[self.kernels[i]](x, derivative=derivative)

    def outputLayer(self, i):
        if i == 0:
            return self.layers[i]

        return self.kernelFunction(self.layers[i] + self.biases[i], i)

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

    def propagateBackwards(self, Y, m):
        L = len(self.layers) - 1

        gradC = cost[self.costFunction](self.Al_Caches[-1], Y, derivative=True)

        self.biasGrads[-1] = np.sum(gradC, axis=1) / m
        self.weightGrads[-1] = np.dot(gradC, self.Al_Caches[-1].T) / m

        self.deltas[L] = np.dot(self.weights[L-1], gradC)

        for l in reversed(range(1, L)):

            dZ = self.kernelFunction(
                self.deltas[l + 1], l + 1, derivative=True)

            #! check the index
            self.deltas[l] = np.dot(self.weights[l-1], dZ)

            self.weightGrads[l - 1] = np.dot(dZ, self.deltas[l].T) / m
            self.biasGrads[l] = np.sum(dZ, axis=1) / m

    # X,Y are matrices (arrays of inpts, outputs)
    def train_batch(self, X, Y, alpha=0.01):

        X = np.array(X)
        Y = np.array(Y)

        # batch size
        m = X.shape[0]

        self.Al_Caches = []
        for layer in self.layers:
            self.Al_Caches.append(np.zeros((layer.size, m)))

        # to cache all layers in all batches
        for i, (x, y) in enumerate(zip(X, Y)):
            self.calculateOutput(x)

            for l in range(len(self.layers)):
                self.Al_Caches[l][:, i] = self.outputLayer(l)

        # because it now represents vectors in column
        self.propagateBackwards(Y.T, m)

        self.applyGrad(alpha=alpha)

        # print(self.calculateCost(X[0], Y[0]))

    def applyGrad(self, alpha):

        for i in range(len(self.weightGrads)-1):

            self.weights[i] = self.weights[i] - self.weightGrads[i].T * alpha

            self.biases[i + 1] = self.biases[i + 1] - \
                self.biasGrads[i + 1] * alpha

    def visualizeTrain(self, x, y, length, alpha=0.1):
        print(self.calculateCost(x, y))
        for i in range(length):
            self.propagateBackwards(x, y)
            self.train(alpha=alpha)
        print(self.calculateCost(x, y))

    def predict(self, x):
        return self.calculateOutput(x)

    def train(self, X_train, Y_train, epochs, batch_size=20, leaning_rate=0.01):

        for batch in range(epochs):

            for i in tqdm(range(0, len(X_train), batch_size)):

                self.train_batch(X_train[i:i+batch_size],
                                 Y_train[i:i+batch_size])

    def accuracy(self, x_val, y_val):

        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.calculateOutput(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)
