#test_knn.py
import numpy as np
import pytest
from custom_knn.knn import KNN


def knn_data():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 1])

    return X,y

def test_establish_neighbors():
    """
    This test establishes that the number of neighbours returned is as required
    """
    neighbors = 2
    X,y = knn_data()

    knn = KNN(k=neighbors)
    knn.fit(X, y)

    sample = np.array([0.9, 0.9])
    indices, labels = knn.establish_neighbors(sample)

    assert len(indices) == neighbors, "establish_neighbors() does not return required number of k indices"
    assert len(labels) == neighbors, "establish_neighbors() does not return required number of k labels"

def test_fit():
    """
    Tests the fit method of the KNN classifier
    """ 
    X,y = knn_data()
    knn = KNN(k=3)
    knn.fit(X,y)
    np.testing.assert_array_equal(knn._X_train, X), "X array not equal"
    np.testing.assert_array_equal(knn._y_train, y), "y array not equal"

def test_predict_single_majority_vote():
    X = np.array([[0,0],[1,1],[2,2]])
    y = np.array([0,0,1])
    knn = KNN(k=2)
    knn.fit(X, y)

    sample = np.array([0.9, 0.9])
    prediction = knn._predict_single(sample)

    # Closest neighbors: [1,0] -> labels [0,0] -> prediction should be 0
    assert prediction == 0, f"Expected prediction 0, got {prediction}"