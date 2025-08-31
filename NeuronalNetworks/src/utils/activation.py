import numpy as np

"""
recommendations for activation functions:
sigmoid: low learning rate: [0.001, 0.01] and architecture with few layers and not too big layer width, sigmoid is prone
to saturation, usually sigmoid + MSE isn't a good choice, better alternative: sigmoid + cross-entropy
relu: 
"""
def activation(x, type="sigmoid"):
    if type == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif type == "relu":
        return np.maximum(0, x)


def derivative_activation(x, type="sigmoid"):
    if type == "sigmoid":
        s = activation(x, "sigmoid")
        return s * (1 - s)
    elif type == "relu":
        return (x > 0).astype(float)
