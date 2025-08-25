import numpy as np

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

def prediction(x): # wandelt Sigmoid werte in binäre Klassifikation um
    return np.round(x)

def cost(error):
    return 0.5 * error**2