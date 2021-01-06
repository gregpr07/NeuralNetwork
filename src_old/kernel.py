import numpy as np


def sigmoid(x, derivative=False):
    # Sigmoida in odvod
    s = 1/(1 + np.exp(-x))
    if not derivative:
        return s
    else:
        return s * (1 - s)


def ReLu(x, derivative=False):
    if not derivative:
        return x if x > 0 else 0,
    else:
        return 1 if x > 0 else 0,


kernels = {
    "ReLu": ReLu,
    'Sigmoid': sigmoid
}
