# Optimizer

import numpy as np
from utils.optimizer import *


"""WIP
import Layer
import NeuralNetwork

class Momentum(Layer):

    def __init__(self, input_dim, output_dim, activation="sigmoid"): # neue Instanzvariable delta_W
        super().__init__(input_dim, output_dim, activation)
        self.delta_W = np.zeros((output_dim, input_dim))
        self.delta_B = np.zeros((1, output_dim))

    def update_val(self, eta, self.alpha):
        self.weight_matrix += (self.alpha - 1) * (eta * (self.delta.T @ self.input)) + self.alpha * self.delta_W
        self.delta_W = eta * (self.delta.T @ self.input)
        self.bias_vector += (self.alpha - 1) * (eta * self.delta).sum(axis=0, keepdims=True) + self.alpha * self.delta_B
        self.delta_B = (eta * self.delta).sum(axis=0, keepdims=True)

class SGD(NeuralNetwork):

    def loop(self, data, batch_size):
        np.random.shuffle(data) # in-place Funktion
        input = data[:, :2]
        target = data[:, 2]
        for i in range(len(data[:, 0]), step=batch_size):
            self.forward(input[i : i + batch_size, :])
            self.backward(target[i : i + batch_size, :])
            self.update_vals()

# AdaGrad ist eine Kombination aus SGD mit batch_size = 1 und AdaGrad(Layer)

class AdaGrad(Layer):

    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        super().__init__(input_dim, output_dim, activation)
        self.gradient_sum_weights = None
        self.gradient_sum_bias = None

    def update_val(self, eta):
        self.weight_matrix += eta / np.sqrt(self.gradient_sum_weights) * (self.delta.T @ self.input)
        self.gradient_sum_weights += (self.delta.T @ self.input)**2
        self.bias_vector += eta / np.sqrt(self.gradient_sum_bias) * (self.delta).sum(axis=0, keepdims=True)
        self.gradient_sum_bias += (self.delta).sum(axis=0, keepdims=True)**2
"""


# für alle Optimizer eine Klasse da Optimizer immer nur die update-Funktion verändern und die update Funktion nur vom jeweiligen layer abhängig ist
class Optimizer:

    def update_val(self, layer):
        raise NotImplementedError
        # updatet die values von layer auf optimizer Art

class Momentum(Optimizer):

    def __init__(self, eta, alpha):
        self.eta = eta
        self.alpha = alpha
        self.velocities_W = {}
        self.velocities_B = {}

    def update_val(self, layer):
        if layer not in self.velocities_W:
            self.velocities_W[layer] = np.zeros_like(layer.weight_matrix)
            self.velocities_B[layer] = np.zeros_like(layer.bias_vector)

        # define gradients
        grad_w = layer.delta.T @ layer.input
        grad_b = layer.delta.sum(axis=0, keepdims=True)

        layer.weight_matrix -= (1 - self.alpha) * (self.eta * grad_w) + self.alpha * self.velocities_W[layer]
        self.velocities_W[layer] = self.eta * grad_w
        layer.bias_vector -= (1 - self.alpha) * self.eta * grad_b + self.alpha * self.velocities_B[layer]
        self.velocities_B[layer] = self.eta * grad_b


class Adam(Optimizer):

    def __init__(self, eta, forgetting_factor1, forgetting_factor2):
        self.eta = eta
        self.forgetting_factor1 = forgetting_factor1
        self.forgetting_factor2 = forgetting_factor2
        self.momentum_weights = None
        self.second_moment_weights = None
        self.momentum_bias = None
        self.second_moment_bias = None

    def update_val(
        self, layer
    ):  # funktioniert so nicht weil bei jedem Aufruf self.momentum_weights usw. überschrieben werden mit np.zeros_like(layer.weight_matrix)
        self.momentum_weights = np.zeros_like(layer.weight_matrix)
        self.second_moment_weights = np.zeros_like(layer.weight_matrix)
        self.momentum_bias = np.zeros_like(layer.bias_vector)
        self.second_moment_bias = np.zeros_like(layer.bias_vector)
        epsilon = 1e-7
        gradient_weight_matrix = layer.delta.T @ layer.input
        gradient_bias_vector = layer.delta.sum(axis=0, keepdims=True)
        # Funktion momentumhat speichert momentum
        layer.weight_matrix += (
            self.eta
            * momentum_hat(
                self.forgetting_factor1, gradient_weight_matrix, self.momentum_weights
            )
            / (
                np.sqrt(
                    second_moment_hat(
                        self.forgetting_factor2,
                        gradient_weight_matrix,
                        self.second_moment_weights,
                    )
                )
                + epsilon
            )
        )
        layer.bias_vector += (
            self.eta
            * second_moment_hat(
                self.forgetting_factor2, gradient_bias_vector, self.momentum_bias
            )
            / (
                np.sqrt(
                    second_moment_hat(
                        self.forgetting_factor1,
                        gradient_bias_vector,
                        self.second_moment_bias,
                    )
                )
                + epsilon
            )
        )


# Instanz von Adam: optimizer = Adam(eta, forgetting_factor1, forgetting_factor2, layer)
# in update_vals optimizer.update_val
