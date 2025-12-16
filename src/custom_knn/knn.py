# knn.py
import numpy as np


class KNN:
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
        This method actually only stores data for **later comparison**,
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
        return np.array([self._predict_single(sample) for sample in X])
    
    def establish_neighbors(self, sample):
        """
        Finds neighbours distances and labels
        """
        # Get Euclidean distance from input to each point in X_train
        distances = np.linalg.norm(self._X_train - sample, axis=1)

        # Find index of the closest neighbours 
        k_indexes = np.argsort(distances)[:self._k]

        # Retrieve the k indexes labels (y_train corresponds to them)
        k_labels = self._y_train[k_indexes] # Classes of the neighbours

        return k_indexes, k_labels

    def _predict_single(self, sample):
        """
        Establishes class label for a single sample
        """
        k_indexes, k_labels = self.establish_neighbors(sample)

        # Get most frequent predicitions (labels)
        label_counts = {} # dictionary
        for label in k_labels:
            if label in label_counts:
                label_counts[label] += 1 # Increase value of key if already exists
            else:
                label_counts[label] = 1 # Set value to one if key is first created (added)

        # Find the label with the highest count
        prediction = max(label_counts, key=label_counts.get)
        return prediction