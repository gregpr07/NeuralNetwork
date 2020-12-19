import numpy as np

kernels = {
    "ReLu": lambda x: x if x > 0 else 0,
    'Sigmoid': lambda x: 1/(1 + np.exp(-x))
}
