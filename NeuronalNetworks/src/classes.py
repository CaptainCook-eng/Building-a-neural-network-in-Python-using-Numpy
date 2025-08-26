import numpy as np
import json
from pathlib import Path
from utils import activation, derivative_activation, mean_squared_error, binary_cross_entropy, categorical_cross_entropy, derivative_mse,derivative_bce, derivative_cce

current_file = Path(__file__)
root_file = current_file.parent.parent

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
    def backward(self, delta_next=None, W_next=None, target=None):
       if delta_next is None: # Output-Layer
           # hier wird der Gradient entweder als positiv mit (output - target) oder negativ mit (target - output) definiert
           # Da der Gradient hier negativ definiert ist wird delta_W zu W_alt addiert und nicht subtrahiert
           self.delta =  derivative_activation(self.z, type=self.activation) * self.derivative_loss(target, self.a) # (batch_size, output_dim) * (batch_size, output_dim) # Wenn es nur einen "Gesamt"-Output gibt (batch_size, 1)
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

    def derivative_loss(self, target, output):
        if self.loss_type == "mse":
            return derivative_mse(target, output)
        elif self.loss_type == "bce":
            return derivative_bce(target, output)
        elif self.loss_type == "cce":
            return derivative_cce(target, output)


class NeuralNetwork:

    def __init__(self, listLayers, loss_type="mse"):
        self.listLayers = listLayers
        self.length = len(self.listLayers)
        self.loss_type = loss_type
        # ändern vom loss_type in der Output-Layer
        self.listLayers[-1].loss_type = self.loss_type
        # Prediction / Output für loss-Methode
        self.output = None
        self.filename = root_file / "data" / "Networks" / "Net.json"
        # json kann keine arrays speichern deswegen .tolist()
        self.value_dict = {i: {"weight": Layer.weight_matrix.tolist(), "bias": Layer.bias_vector.tolist()} for i, Layer in enumerate(self.listLayers)}

    def forward(self, input):
        output = input
        for i in range(self.length):
            output = self.listLayers[i].forward(output)# rekursive definition von output
        self.output = output
        return output

    def backward(self, target):
        # Man könnte alternativ zu reversed(range) auch list.reverse() benutzen um von hinten nach vorne durch die Liste zu iterieren
        # Output-Layer
        self.listLayers[-1].delta = self.listLayers[-1].backward(None, None, target)
        # Hidden-Layers
        for i in reversed(range(self.length - 1)):
            self.listLayers[i].delta = self.listLayers[i].backward(
                self.listLayers[i + 1].delta,
                self.listLayers[i + 1].weight_matrix
            ) # delta_next, W_next

    def update_vals(self, eta):
        for i in range(self.length):
            self.listLayers[i].weight_matrix += eta * (self.listLayers[i].delta.T @ self.listLayers[i].input) # Layer.input wird mit Layer.forward aktualisiert d. h. mit NeuralNetwork.forward auch
            self.listLayers[i].bias_vector += (eta * self.listLayers[i].delta).sum(axis=0, keepdims=True) # hier kann man die Performance mit np.mean() verbessern

    def loss(self, target):
        if self.loss_type == "mse":
            return mean_squared_error(target, self.output)
        elif self.loss_type == "bce":
            return binary_cross_entropy(target, self.output)
        elif self.loss_type == "cce":
            return categorical_cross_entropy(target, self.output)
# json kann keine ndarrays speichern
    def save_vals(self):
        self.value_dict.update({i: {"weight": Layer.weight_matrix.tolist(), "bias": Layer.bias_vector.tolist()} for i, Layer in enumerate(self.listLayers)}) # dictionary muss upgedatet werden um neue Werte zu laden
        with open(self.filename, "w") as json_file:
            json.dump(self.value_dict, json_file, indent=4)

    def load_vals(self):
        with open(self.filename, "r") as f:
            data = json.load(f)
        for i in range(self.length):
            self.listLayers[i].weight_matrix = np.array(data[f"{i}"]["weight"])
            self.listLayers[i].bias_vector = np.array(data[f"{i}"]["bias"])

class Optimizer(Layer):

    def __init__(self, input_dim, output_dim, activation="sigmoid"): # neue Instanzvariable delta_W
        super().__init__(input_dim, output_dim, activation)
        self.delta_W = np.zeros((output_dim, input_dim))
        self.delta_B = np.zeros((1, output_dim))

    def update_val(self, eta, alpha):

        self.weight_matrix += (alpha - 1) * (eta * (self.delta.T @ self.input)) + alpha * self.delta_W
        self.delta_W = eta * (self.delta.T @ self.input)
        self.bias_vector += (alpha - 1) * (eta * self.delta).sum(axis=0, keepdims=True) + alpha * self.delta_B
        self.delta_B = (eta * self.delta).sum(axis=0, keepdims=True)