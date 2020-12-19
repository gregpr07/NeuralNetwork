# implementation of simple dumb network
import numpy as np
from .kernels import kernels

# !! the big problem now is that first layer has activation fucntion as well


class Neuron():
    def __init__(self, value, index, kernel='ReLu'):
        self.value = value
        self.kernel = kernel
        self.index = index

    def __repr__(self):
        return f'Nevron, z indexom {self.index}, vrednost {self.value} in {self.kernel} kernel'

    def output(self):
        return kernels[self.kernel](self.value)

    def updateValue(self, newValue):
        self.value = newValue


class Synapse():
    def __init__(self, parent, child, weight=1):
        self.parent = parent
        self.child = child
        self.weight = weight

    def __repr__(self):
        return f'Sinapsa, parent: {self.parent}, child: {self.child} ,weight {self.weight}'

    def output(self):
        return self.weight * self.parent.output()


class Layer():
    def __init__(self, size, kernels='ReLu'):
        self.size = size
        self.layer = np.array([Neuron(0, x, kernel='ReLu')
                               for x in range(size)])

    def __repr__(self):
        return ' '.join([str(round(neuron.value, 3)) for neuron in self.layer])

    # ? funkcija, ki se uporablja samo za update prvega layerja - za input function
    def updateLayer(self, values):
        [neuron.updateValue(val) for neuron, val in zip(self.layer, values)]


#! this is fully connected layer - not bothering with other connections
class ConnectionLayer():
    def __init__(self, parentLayer, childLayer):
        # ? connections is n*m matrix, n=parent, m=child, each connection is a Synapse
        # ? connections[i] = konstanten parent
        # ? connections[:,i] = konstanten child
        self.connections = np.array(
            [[Synapse(parent, child, weight=np.random.random()) for child in childLayer.layer] for parent in parentLayer.layer])
        self.parentLayer = parentLayer
        self.childLayer = childLayer

    def childrenNodes(self, i):
        return self.connections[i]

    def parentNodes(self, i):
        return self.connections[:, i]

    # ? s to funkcijo lahko zdaj cisto vsak nevron, ki ni v input layerju spreminjamo
    # ?  uporabljal bom zato, da izračunam vrednosti v mreži
    def updateChildNeuron(self, i):
        neuron = self.childLayer.layer[i]
        new = sum([synapse.output() for synapse in self.parentNodes(i)])
        neuron.updateValue(new)

    def updateChildrenNeurons(self):
        for i in range(len(self.childLayer.layer)):
            self.updateChildNeuron(i)


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
        return self.network

    def calculateOutput(self):
        for conn_layer in self.connectors:
            conn_layer.updateChildrenNeurons()

        return self.network[-1]

    def setInputValues(self, values):
        self.network[0].updateLayer(values)
