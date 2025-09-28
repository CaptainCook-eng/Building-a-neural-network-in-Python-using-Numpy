import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from classes.Layer import Layer
from classes.NeuralNetwork import NeuralNetwork
from classes.Optimizer import *
from utils.OneHotEncoding import *

# Auf vorher generierte synthetische Datensätze zugreifen mit pathlib
current_file = Path(__file__)
root_file = current_file.parent.parent
synthetischer_Datensatz_file = root_file / "data" / "synthetischer_Datensatz_mit_2_Klassen.csv" # in ML wird diese Tabelle mit features | label auch "observations" genannt

# ==========================
# Datensatz laden
# ==========================
data = np.loadtxt(synthetischer_Datensatz_file, delimiter=",", skiprows=1)
X = data[:, :2]   # Features # (100, 2)
labels = data[:, 2].reshape(-1, 1)  # Zielwerte als Spaltenvektor # (100, 1) # (batch_size, output_dim)

#============================
#MNIST Datensatz
#============================
import tensorflow as tf
from PIL import Image


mnist = tf.keras.datasets.mnist

# training input: an array of 60.000 gray scale 255 images in 28x28 format
(training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()

training_inputs = np.reshape(training_inputs, shape=(60000, 784))
# keep weights and biases from exploding (activation function saturation)
training_inputs = training_inputs / 255.0

training_labels = np.reshape(training_labels, shape=(60000, 1))
training_labels = as_one_hot(training_labels)

# actually really important for testing
# seeding the RNG (random number generator) lets you reproduce an experiment based on "random" numbers multiple times
# here it manages the weights and bias initialization; the RNG always produces the same weights and biases after initialization
# This lets you compare e.g. different optimizers without leeway for randomness
np.random.seed(42)

# Anzahl der Trainingsdurchläufe
epochs = 20
batch_size = 64

# ========================================
# Training (Feedforward + Backpropagation)
# ========================================

# Neuronales Netzwerk mit Architektur: 784-16-16-10

Layer1 = Layer(784, 16, "sigmoid")
Layer2 = Layer(16, 16, "sigmoid")
Layer3 = Layer(16, 10, activation="sigmoid")
output_dim = Layer3.output_dim

optimizer = Momentum(eta=0.01, alpha=0.1)


Net = NeuralNetwork([Layer1, Layer2, Layer3],eta=0.001, loss_type="cce", optimizer=None) # standardmäßig ist MSE als loss-function ausgewählt

mean_errors = []

# Net.load_vals(Net.filename)

observations = np.hstack((training_inputs, training_labels))
samples, features_plus_labels = observations.shape

for epoch in range(epochs):
    np.random.shuffle(observations)

    inputs = observations[:, :-output_dim]
    target = observations[:, -output_dim:]
    print(target.shape)
    for sample in range(0, samples, batch_size):
        Net.forward(inputs[sample : sample + batch_size, :])
        Net.backward(target[sample : sample + batch_size, :])
        Net.update_vals()
        # mittlerer Fehler pro batch -> hat epochs * round(60.000 / 64) Einträge
        mean_errors.append(float(Net.loss(target[sample : sample + batch_size])))

Net.save_vals()

print(len(mean_errors))
print(mean_errors)

# --------------------------------------
# Visualisierung der Entscheidungsgrenze
# --------------------------------------
"""
# 100 x 100 Meshgrid

x = np.linspace(-0.5, 1.5, 100)

y = np.linspace(-0.5, 1.5, 100)

xv, yv = np.meshgrid(x, y) #zwei (100, 100) arrays das Koordinatensystem wird von links unten nach rechts oben indiziert

pairs = np.stack((xv, yv), axis=-1) # entsprechende Einträge aus xv und yv als Tupel in (100, 100)-array # (100, 100, 2)

pairs_flat = pairs.reshape(-1, pairs.shape[2]) # neuer flacher (10000, 2) array der kompatibel mit feed_forward(x) ist

output = Net.forward(pairs_flat)

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
"""