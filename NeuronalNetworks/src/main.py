import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from classes import Layer

# Auf vorher generierte synthetische Datensätze zugreifen mit pathlib
current_file = Path(__file__)

root_file = current_file.parent.parent

synthetischer_Datensatz_file = root_file / "data" / "synthetischer_Datensatz_mit_2_Klassen.csv"

# ==========================
# Datensatz laden
# ==========================
data = np.loadtxt(synthetischer_Datensatz_file, delimiter=",", skiprows=1)
X = data[:, :2]   # Features # (100, 2)
labels = data[:, 2].reshape(-1, 1)  # Zielwerte als Spaltenvektor # (100, 1) # (batch_size, output_dim)

np.random.seed(42) # wofür genau braucht man das

# Trainingsvariablen setzen
eta = 0.1
epochs = 100

# ==========================
# Training (Feedforward + Backpropagation)
# ==========================

# Neuronales Netzwerk mit Layer Klasse: 2-2-1

Layer1 = Layer(2, 2, "sigmoid")

Layer2 = Layer(2, 1, "sigmoid")

for epoch in range(epochs):
    output1 = Layer1.forward(X) #aktualisiert self.z und self.a für backward-Funktion
    output2 = Layer2.forward(output1)

    delta2 = Layer2.backward(target=labels) # berechnet delta2 aus error = Layer2.a - labels
    delta1 = Layer1.backward(delta2, Layer2.weight_matrix) # berechnet delta1 aus delta2, Layer2.weight_matrix und Layer1.z

    Layer2.update_val(eta) # updatet die Instanzvariablen self.weight_matrix und self.bias mithilfe von vorher berechneten Instanzvariable self.delta und der unabhängigen Lernrate eta
    Layer1.update_val(eta)

# --------------------------------------
# Visualisierung der Entscheidungsgrenze
# --------------------------------------

# 100 x 100 Meshgrid

x = np.linspace(-0.5, 1.5, 100)

y = np.linspace(-0.5, 1.5, 100)

xv, yv = np.meshgrid(x, y) #zwei (100, 100) arrays das Koordinatensystem wird von links unten nach rechts oben indiziert

pairs = np.stack((xv, yv), axis=-1) # entsprechende Einträge aus xv und yv als Tupel in (100, 100)-array # (100, 100, 2)

pairs_flat = pairs.reshape(-1, pairs.shape[2]) # neuer flacher (10000, 2) array der kompatibel mit feed_forward(x) ist

o1 = Layer1.forward(pairs_flat)
output = Layer2.forward(o1)

flat = output.ravel() # (10000,) array

z_ready = flat.reshape(100, 100) # (100, 100) array

plt.contour(xv, yv, z_ready, levels=[0.5]) # Niveaulinie auf der Höhe = 0.5

class_0 = X[labels[:, 0] == 0]
class_1 = X[labels[:, 0] == 1]

plt.scatter(class_0[:, 0], class_0[:, 1], color="blue", label="Klasse 0")
plt.scatter(class_1[:, 0], class_1[:, 1], color="red", label="Klasse 1")
plt.xlabel("x-Achse")
plt.ylabel("y-Achse")
plt.title("Trainiertes 2-2-1 Netz: Feedforward")
plt.legend()
plt.grid(True)
plt.show()
