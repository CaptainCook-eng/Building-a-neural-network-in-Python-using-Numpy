import numpy as np

"""
Function that returns a weight matrix based on the number of neurons in the previous and the current layer 
(fan_in and fan_out respectively). It uses the uniform distribution for weight variance in an intervall depending on fan_in and fan_out.
"""
def glorot(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))