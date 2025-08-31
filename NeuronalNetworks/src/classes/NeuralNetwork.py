from pathlib import Path
import json

from classes.Optimizer import Momentum
from utils.loss import * # imports all variables, functions and classes from utils

current_file = Path(__file__)
root_file = current_file.parent.parent.parent

class NeuralNetwork:

    def __init__(self, list_layers, loss_type="mse", eta=0.1, optimizer=None):
        self.list_layers = list_layers
        # nur einmal verwendete Instanzvariable könnte vielleicht rausgelassen werden
        self.length = len(self.list_layers)
        self.loss_type = loss_type
        # ändern vom loss_type in der Output-Layer
        self.list_layers[-1].loss_type = self.loss_type
        # Prediction / Output für loss-Methode
        self.output = None
        self.filename = root_file / "data" / "Networks" / "Net.json"
        # json kann keine arrays speichern deswegen .tolist()
        self.value_dict = {i: {"weight": layer.weight_matrix.tolist(), "bias": layer.bias_vector.tolist()} for i, layer in enumerate(self.list_layers)}

        # Variable die den Gradient descent tweaken
        self.eta = eta
        self.optimizer = optimizer

    # -------------------
    # Forward Pass
    # -------------------
    def forward(self, inputs):
        output = inputs
        for layer in self.list_layers:
            output = layer.forward(output)  # rekursive definition von output
        self.output = output
        return output

    # -------------------
    # Backward Pass
    # -------------------
    def backward(self, target):
        # Man könnte alternativ zu reversed(range) auch list.reverse() benutzen um von hinten nach vorne durch die Liste zu iterieren
        # Output-Layer
        self.list_layers[-1].delta = self.list_layers[-1].backward(None, None, target)
        # Hidden-Layers
        for i in reversed(range(self.length - 1)):
            self.list_layers[i].delta = self.list_layers[i].backward(
                self.list_layers[i + 1].delta,
                self.list_layers[i + 1].weight_matrix
            ) # delta_next, W_next

    # -------------------------------
    # Gewichte und Bias aktualisieren
    # -------------------------------
    def update_vals(self): # falsch gebaut !!!!
        for layer in self.list_layers:
            if self.optimizer is None:
                layer.update_val(self.eta) # hier kann man die Performance mit np.mean() verbessern
            else:
                self.optimizer.update_val(layer)

    # ---------------------------
    # Definition loss-Funktion
    # ---------------------------
    def loss(self, target):
        if self.loss_type == "mse":
            return mean_squared_error(target, self.output)
        elif self.loss_type == "bce":
            return binary_cross_entropy(target, self.output)
        elif self.loss_type == "cce":
            return categorical_cross_entropy(target, self.output)


#Falls man kein full batch training machen will sondern die batch_size selbst bestimmen muss kann man diese Funktion benutzen:
    # --------------------------
    # SGD
    # --------------------------
    def loop(self, training_inputs, training_labels, batch_size):
        observations = np.hstack((training_inputs, training_labels))
        np.random.shuffle(observations) # in-place Funktion
        # training_labels ist immer eindimensional, daher kann targets einfach als die letzte Spalte von observations definiert werden
        inputs = observations[:, :-1]
        # keepdims slicing Methode: [-1] wird als untermatrix behandelt wodurch der output array (rows, 1) wird
        target = observations[:, [-1]]

        for i in range(0, observations.shape[0], batch_size):
            self.forward(inputs[i : i + batch_size, :])
            self.backward(target[i : i + batch_size])
            self.update_vals()

# json kann keine ndarrays speichern
    # -------------------------------
    # Weights und Biases speichern
    # -------------------------------
    def save_vals(self):
        self.value_dict.update({i: {"weight": layer.weight_matrix.tolist(), "bias": layer.bias_vector.tolist()} for i, layer in enumerate(self.list_layers)}) # dictionary muss upgedatet werden um neue Werte zu laden
        with open(self.filename, "w") as json_file:
            json.dump(self.value_dict, json_file, indent=4)

    # --------------------------------------
    # Gespeicherte Weights und Biases laden
    # --------------------------------------
    def load_vals(self, filename):
        self.filename = filename
        with open(self.filename, "r") as f:
            data = json.load(f)
        for i, layer in enumerate(self.list_layers):
            layer.weight_matrix = np.array(data[f"{i}"]["weight"])
            layer.bias_vector = np.array(data[f"{i}"]["bias"])
