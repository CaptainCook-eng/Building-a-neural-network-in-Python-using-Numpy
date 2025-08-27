import numpy as np

# target, output: (batch_size, output_dim)
# np.mean() flattet das array zuerst und berechnet dann den Durchschnitt

def mean_squared_error(target, output): # positiv definierter Gradient
    return 0.5 * np.mean((target - output)**2) # statt Standard: 0.5 * np.sum(target - output)**2  np.mean(), weil der Fehler so nicht mit der Batch Größe wächst (ist so Standard in neueren Frameworks)

def derivative_mse(target, output):
    return target - output
"""
cross entropy:
cross entropy is a measure for how dissimilar two probability distributions are over the same set of events.
It is calculated with the expected value operator: H(q, p) := - Ep[log(q)]. In the binary case there are only two possible outcomes for X, 1 and 0. 
Furthermore there are only two probability for p and q because of that. p is an element of {y, 1 - y} and q is an element of {yhat, 1 - yhat}. 
This simplifies the equation to  
"""
def binary_cross_entropy(target, output): # für binäre Klassifikation mit einem output also target, output: (batch_size, 1)
    return -np.mean(target * np.log(output + 1e-12) + (1 - target) * np.log(1 - output + 1e-12)) # durch Broadcasting gibt es hier keine Probleme

def derivative_bce(target, output):
    return -target / output + (1 - target) / (1 - output)

def categorical_cross_entropy(target, output): # hier ist das target für ein sample ein One-Hot-Vektor, also ein Vektor mit einer Komponente gleich 1 und dem Rest Null
    # target, output: (batch_size, number_of_classes)
    return -np.mean(target * np.log(output + 1e-12))

def derivative_cce(target, output):
    return - target / output