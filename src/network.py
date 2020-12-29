import numpy as np
from kernel import kernels


class Network():
    def __init__(self):

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

    def __repr__(self):
        return str((self.layers, self.kernels, self.biases))

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
        self.biases.append(np.random.rand(dimension))
        self.kernels.append(kernel)
