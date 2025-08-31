from utils.activation import * # importiert numpy as np mit
from utils.loss import *
from utils.weightInit import *

class Layer:

    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initializing weights and biases
        self.weight_matrix = glorot(input_dim, output_dim)
        self.bias_vector = np.zeros((1, output_dim))


        # Variablen für Forward / Backward Pass speichern
        self.input = None
        self.z = None
        self.a = None
        self.delta = None

        # modifying the network
        self.activation = activation
        self.loss_type = "mse"

    # ------------------
    # Forward Pass
    # ------------------
    # Layers speichern die Aktivierungen zeilenweise pro Sample also so (activation(sample1), activation(sample2), ..., activation(sample_n)).T # Das ist die übliche Konvention der Input muss die shape (batch_size, features) haben
    def forward(self, input):
        self.input = input
        self.z = input @ self.weight_matrix.T + self.bias_vector # ((batch_size, input_dim) @ (input_dim, output_dim)) + (1, output_dim) = (batch_size, output_dim) + (1, output_dim)
        self.a = activation(self.z, self.activation) # (batch_size, output_dim)
        return self.a

    # ------------------
    # Backward Pass
    # ------------------
    def backward(self, delta_next=None, w_next=None, target=None):
       if delta_next is None: # Output-Layer
           # Ich habs komplett vercheckt und die Kettenregel falsch angewendet, wodurch ich die ganze Zeit ein extra Minus-Zeichen berechnet hab
           # -> negativer gradient wird -> zeigt in Richtung Minima -> wird addiert statt subtrahiert
           self.delta =  derivative_activation(self.z, type=self.activation) * self.derivative_loss(target, self.a) # (batch_size, output_dim) * (batch_size, output_dim) # Wenn es nur einen "Gesamt"-Output gibt (batch_size, 1)
           return self.delta
       else: # Hidden-Layer
           self.delta = derivative_activation(self.z, type=self.activation) * (delta_next @ w_next) # (batch_size, output_dim) * (batch_size, output_dim_next) @ (output_dim_next, output_dim) = (batch_size, output_dim)
           return self.delta

    # -------------------------
    # Gewichte und Bias updaten
    # -------------------------
    def update_val(self, eta):
        # Gradient (zeigt Richtung Maxima) wird abgezogen
        self.weight_matrix -= eta * (self.delta.T @ self.input) # (1,) * (output_dim, batch_size) @ (batch_size, input_dim) = (output_dim, input_dim)
        # damit np.sum() den array nicht abflacht keepdims=True
        self.bias_vector -= (eta * self.delta).sum(axis=0, keepdims=True) # (batch_size, output_dim) --> (1, output_dim) (mit der sum-Funktion werden alle Zeilen von self.delta addiert

    # ---------------------------
    # Ableitung der loss-Funktion
    # ---------------------------
    def derivative_loss(self, target, output):
        if self.loss_type == "mse":
            return derivative_mse(target, output)
        elif self.loss_type == "bce":
            return derivative_bce(target, output)
        elif self.loss_type == "cce":
            return derivative_cce(target, output)