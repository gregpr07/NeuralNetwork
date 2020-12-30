import numpy as np


def meanSquare(A, B, derivative=False):
    if not derivative:
        return (np.sum((A - B) ** 2))/len(A)
    elif derivative:
        return 2 * (A - B)


cost = {
    'meanSquare': meanSquare
}
