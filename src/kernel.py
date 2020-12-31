import numpy as np


def sigmoid(X, derivative=False):
    # Sigmoid and derivative
    s = 1/(1 + np.exp(-X))
    if not derivative:
        return s
    else:
        return s * (1 - s)


def ReLU(X, alpha=0, derivative=False):
    # ReLU function and derivative
    if derivative == False:
        return np.where(X < 0, alpha*X, X)
    elif derivative == True:
        X_relu = np.ones_like(X)
        X_relu[X < 0] = alpha
        return X_relu


def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


kernels = {
    "ReLu": ReLU,
    'Sigmoid': sigmoid,
    'Softmax': softmax
}
