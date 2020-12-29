import numpy as np


def meanSquare(A, B, derivative=False):
    if not derivative:
        return np.sqrt(np.sum((A - B) ** 2))
    elif derivative:
        return np.sum(2 * (A - B))
