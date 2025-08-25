import numpy as np

from utils import activation, derivative_activation

class Layer:

    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        self.weight_matrix = np.random.randn(output_dim, input_dim) * 0.1 # (output_dim, input_dim)
        self.bias_vector = np.zeros((1, output_dim))
        self.activation = activation

        # Variablen für Forward / Backward Pass speichern
        self.input = None
        self.z = None
        self.a = None
        self.delta = None

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
    def backward(self, delta_next=None, W_next=None, target=None):
       if delta_next is None: # Output-Layer
           # hier wird der Gradient entweder als positiv mit (output - target) oder negativ mit (target - output) definiert
           # Da der Gradient hier negativ definiert ist wird delta_W zu W_alt addiert und nicht subtrahiert
           self.delta =  derivative_activation(self.z, type=self.activation) * (target - self.a) # (batch_size, output_dim) * (batch_size, output_dim) # Wenn es nur einen "Gesamt"-Output gibt (batch_size, 1)
           return self.delta
       else: # Hidden-Layer
           self.delta = derivative_activation(self.z, type=self.activation) * (delta_next @ W_next) # (batch_size, output_dim) * (batch_size, output_dim_next) @ (output_dim_next, output_dim) = (batch_size, output_dim)
           return self.delta

    # -------------------------
    # Gewichte und Bias updaten
    # -------------------------
    def update_val(self, eta):
        # Addition von delta_W, weil der Gradient schon negativ definiert ist
        self.weight_matrix += eta * (self.delta.T @ self.input) # (1,) * (output_dim, batch_size) @ (batch_size, input_dim) = (output_dim, input_dim)
        # damit np.sum() den array nicht abflacht keepdims=True
        self.bias_vector += (eta * self.delta).sum(axis=0, keepdims=True) # (batch_size, output_dim) --> (1, output_dim) (mit der sum-Funktion werden alle Zeilen von self.delta addiert



class NeuralNetwork:

    def __init__(self, listLayers):
        self.listLayers = listLayers
        self.length = len(self.listLayers)

    def forward(self, input):
        output = input
        for i in range(self.length):
            output = self.listLayers[i].forward(output) # rekursive definition von output
        return output

    def backward(self, target):
        # Man könnte alternativ zu reversed(range) auch list.reverse() benutzen um von hinten nach vorne durch die Liste zu iterieren
        # Output-Layer
        self.listLayers[-1].delta = self.listLayers[-1].backward(target)
        # Hidden-Layers
        for i in reversed(range(self.length - 1)):
            self.listLayers[i].delta = self.listLayers[i].backward(
                self.listLayers[i + 1].delta,
                self.listLayers[i + 1].weight_matrix
            ) # delta_next, W_next

    def update_vals(self, eta):
        for i in range(self.length):
            self.listLayers[i].weight_matrix += eta * (self.listLayers[i].delta.T @ self.listLayers[i].input) # Layer.input wird mit Layer.forward aktualisiert d. h. mit NeuralNetwork.forward auch
            self.listLayers[i].bias_vector += (eta * self.listLayers[i].delta).sum(axis=0, keepdims=True)