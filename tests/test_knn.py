#test_knn.py
import numpy as np
import pytest
from custom_knn.knn import KNN

def test_establish_neighbors():
    neighbors = 2
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 1])

    knn = KNN(k=neighbors)
    knn.fit(X, y)

    sample = np.array([0.9, 0.9])
    indices, labels = knn.establish_neighbors(sample)

    assert len(indices) == neighbors, "establish_neighbors() does not return required number of k indices"
    assert len(labels) == neighbors, "establish_neighbors() does not return required number of k labels"