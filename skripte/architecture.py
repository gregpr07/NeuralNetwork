import numpy as np
from .kernels import kernels


class Neuron():
    def __init__(self, bias, index, kernel):
        self.value = 0
        self.kernel = kernel
        self.index = index
        self.bias = bias
        self.biasDerivative = 0

    def __repr__(self):
        return f'Nevron, z indexom {self.index}, vrednost {self.value} in {self.kernel} kernel'

    def output(self):
        return kernels[self.kernel](self.value)

    def updateValue(self, newValue):
        self.value = newValue + self.bias

    def forceValue(self, newValue):
        self.value = newValue

    def changeBias(self, newBias):
        self.bias = newBias

    def changeBiasDerivative(self, biasDerivative):
        self.biasDerivative = biasDerivative


class Synapse():
    def __init__(self, parent, child, weight=1):
        self.parent = parent
        self.child = child
        self.weight = weight
        self.weightDerivative = 0

    def __repr__(self):
        return f'Sinapsa, parent: {self.parent}, child: {self.child}, weight {self.weight}'

    def output(self):
        return self.weight * self.parent.output()

    def changeWeight(self, newWeight):
        self.weight = newWeight

    def changeWeightDerivative(self, newWeightDerivative):
        self.weightDerivative = newWeightDerivative


class Layer():
    def __init__(self, size, kernels='Sigmoid'):
        self.size = size
        self.layer = np.array([Neuron(np.random.randint(-1, 1), x, kernel=kernels)
                               for x in range(size)])

    def __repr__(self):
        return ' '.join([str(round(neuron.value, 3)) for neuron in self.layer])

    # ? funkcija, ki se uporablja samo za update prvega layerja - za input function
    def updateLayer(self, values):
        [neuron.forceValue(val) for neuron, val in zip(self.layer, values)]

    def values(self):
        return np.array([neuron.output() for neuron in self.layer])


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

    def getChild(self, i):
        return self.childLayer.layer[i]

    # ? s to funkcijo lahko zdaj cisto vsak nevron, ki ni v input layerju spreminjamo
    # ?  uporabljal bom zato, da izračunam vrednosti v mreži
    def updateChildNeuron(self, i):
        neuron = self.getChild(i)
        new = sum([synapse.output() for synapse in self.parentNodes(i)])
        neuron.updateValue(new)

    def updateChildrenNeurons(self):
        for i in range(len(self.childLayer.layer)):
            self.updateChildNeuron(i)

    def calculateChildDerivative(self, i):
        neuron = self.getChild(i)
        for parent in self.parentNodes(i):
