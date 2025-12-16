# knn.py
import numpy as np
from collections import Counter


class KNN():
    """
    KNN - K-Nearest Neighbours
    This class is a custom implementation of KNN algorithm

    Parameters
    ----------
    _k : int - K number of nearest neighbours to check
    _X_train/_y_train - already "labelled" data
    """
    
    def __init__(self, k):
        self._k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self, X, y):
        """
        This method stores feature vectors and their labels.
        This method actually only stores data for later comparison,
        but keeping the unfiorm ML Algorithms name - fit()

        Input Parameters
        ----------
        X - training feature vectors
        y - target labels
        """

        self._X_train = np.array(X)
        self._y_train = np.array(y)
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X

        Input Parameters:
        ----------------
        X - input samples
        """
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    

    def _predict_single(self, x):
        # Get Euclidean distance from input to each point in X_train
        distances = np.linalg.norm(self._X_train - x, axis=1)

        # Find index of the closest neighbours 
        k_indexes = np.argsort(distances)[:self._k]

        # Retrieve the k indexes labels (y_train corresponds to them)
        k_labels = self._y_train[k_indexes] # Classes of the neighbours

        # Get most frequent predicitions
        prediction = Counter(k_labels).most_common(1)[0][0]
        return prediction