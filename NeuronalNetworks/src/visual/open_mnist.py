import tensorflow as tf
from PIL import Image
import numpy as np


test_variance = np.random.randn(16, 784)
print(np.min(test_variance))
mnist = tf.keras.datasets.mnist

# training input: an array of 60.000 gray scale 255 images in 28x28 format
(training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()

np.reshape(training_inputs, newshape=(60000, 784))

img_array = training_inputs[4, :, :]

img = Image.fromarray(img_array)
img.show()