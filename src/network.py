# implementation of simple dumb network
import numpy as np
from .kernel import kernels
from .architecture import *
from .cost import meanSquare as costFunction

# !! the big problem now is that first layer has activation fucntion as well


class Network():
    #! first layer is always the input layer
    def __init__(self, layerDimensions):
        def connectAllLayers(network):
            if len(network) > 1:
                return np.array([ConnectionLayer(network[i], network[i+1]) for i in range(len(network)-1)])

        self.network = np.array([Layer(x) for x in layerDimensions])
        self.connectors = connectAllLayers(self.network)

        self.dimensions = layerDimensions

    def __repr__(self):
        def format_str(layerName, nrNeurons):
            names = ['Input', 'Hidden Layer', 'Output']
            return f'{names[layerName]}: {nrNeurons} nevronov \n'

        reprs = ["V celoti povezana mreža\n"]
        for i, dim in enumerate(self.dimensions):
            name = 0
            if i:
                name = 1
            if i == len(self.dimensions) - 1:
                name = -1
            reprs.append(format_str(name, dim))

        return ''.join(reprs)

        # def

    def showNeuronValues(self):
        return (self.network)

    def calculateOutput(self):
        for conn_layer in self.connectors:
            conn_layer.updateChildrenNeurons()

        return self.network[-1].values()

    def setInputValues(self, values):
        self.network[0].updateLayer(values)

    def cost(self, y):
        return costFunction(np.array(y), self.calculateOutput())
