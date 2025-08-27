from pathlib import Path
import json
from utils.loss import * # imports all variables, functions and classes from utils

current_file = Path(__file__)
root_file = current_file.parent.parent.parent

class NeuralNetwork:

    def __init__(self, list_layers, loss_type="mse", learning_rate=0.1, momentum=None):
        self.list_layers = list_layers
        self.length = len(self.list_layers)
        self.loss_type = loss_type
        # ändern vom loss_type in der Output-Layer
        self.list_layers[-1].loss_type = self.loss_type
        # Prediction / Output für loss-Methode
        self.output = None
        self.filename = root_file / "data" / "Networks" / "Net.json"
        # json kann keine arrays speichern deswegen .tolist()
        self.value_dict = {i: {"weight": Layer.weight_matrix.tolist(), "bias": Layer.bias_vector.tolist()} for i, Layer in enumerate(self.list_layers)}

        # Variablen die wichtig für den Optimizer sind
        self.eta = learning_rate
        self.alpha = momentum

    # -------------------
    # Forward Pass
    # -------------------
    def forward(self, input):
        output = input
        for i in range(self.length):
            output = self.list_layers[i].forward(output)# rekursive definition von output
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
    def update_vals(self, optimizer=None): # falsch gebaut !!!!
        for i in range(self.length):
            if optimizer is None:
                self.list_layers[i].update_val(self.eta, self.alpha) # hier kann man die Performance mit np.mean() verbessern
            else:
                optimizer.update_val(self.list_layers[i])

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
# json kann keine ndarrays speichern

    # -------------------------------
    # Weights und Biases speichern
    # -------------------------------
    def save_vals(self):
        self.value_dict.update({i: {"weight": Layer.weight_matrix.tolist(), "bias": Layer.bias_vector.tolist()} for i, Layer in enumerate(self.list_layers)}) # dictionary muss upgedatet werden um neue Werte zu laden
        with open(self.filename, "w") as json_file:
            json.dump(self.value_dict, json_file, indent=4)

    # --------------------------------------
    # Gespeicherte Weights und Biases laden
    # --------------------------------------
    def load_vals(self):
        with open(self.filename, "r") as f:
            data = json.load(f)
        for i in range(self.length):
            self.list_layers[i].weight_matrix = np.array(data[f"{i}"]["weight"])
            self.list_layers[i].bias_vector = np.array(data[f"{i}"]["bias"])
