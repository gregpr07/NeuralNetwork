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


kernels = {
    "ReLu": ReLU,
    'Sigmoid': sigmoid
}
