import numpy as np


def meanSquare(A, B):
    return np.sqrt(np.sum((A-B)**2))
