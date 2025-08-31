import numpy as np

# Vektorisierte Funktionen für Klassifikationen mit mehreren Kategorien;
# wandeln den Index der Kategorie in einen One-Hot-Vektor um

def as_one_hot(label_vector: np.ndarray, output_dim: int = 10) -> np.ndarray:
    """Convert a number to a one-hot-like binary array of length `output_dim`."""
    batch_size = label_vector.shape[0]
    array = np.zeros((batch_size, output_dim))

    for j in range(batch_size):
        """the number in label_vector at index j is a one in column j in array"""
        array[j][label_vector[j]] = 1

    return array

def as_integer(array: np.ndarray) -> np.ndarray:
    """Convert a one-hot-encoded vector to an integer."""
    batch_size, output_dim = array.shape
    label_vector = np.zeros((batch_size, 1))
    for j in range(batch_size):
        for i in range(output_dim):
            if array[j][i] == 1:
                label_vector[j] = i
    return label_vector