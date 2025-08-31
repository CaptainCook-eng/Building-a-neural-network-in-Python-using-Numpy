import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
"""
import tensorflow as tf
MNIST = tf.keras.datasets.mnist
(training_inputs, training_labels), (test_inputs, test_labels) = MNIST.load_data()
"""
img = Image.open("chess-bishop-black.png")
img = np.array(img)

fig, ax = plt.subplots()
ax.show()
