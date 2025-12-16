# test.py
from knn import KNN
import numpy as np
from sklearn.datasets import load_wine #Dataset from scikit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class TestKNN:
    """
    TestKNN - Test Class for K-Nearest Neighbours KNN class implementation
    This class contains of methods which purpose is to test out KNN on different datasets

    Parameters
    ----------
    do_print : bool - printing ON/OFF
    """
    def __init__(self, do_print = True):
        self.do_print = do_print    

    def accuracy_test(self, neighbors, X, y, test_size = 0.2,random_state = 42, dataset_name = "?"):
        """
        Simple accuracy test comparing custom implementation
        with scikit one
        :param neighbors: number of k neighbours
        :param X: feature matrix from scikit dataset
        :param y: target labels matrix from scikit dataset
        :param test_size: proportion of dataset to include in the test split
        :param random_state: random seed for reproducibility in train-test splitting
        :param dataset_name: 
        """
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
        # Custom k-NN implementation
        knn = KNN(k=neighbors)
        knn.fit(X_train,y_train)

        y_predicts = knn.predict(X_test) # Predict class labels
        accuracy = accuracy_score(y_test, y_predicts) # Proportion of correctly predicted samples

        # Scikit k-NN implementation
        sk_knn = KNeighborsClassifier(n_neighbors=neighbors)
        sk_knn.fit(X_train, y_train)

        y_predicts_sk = sk_knn.predict(X_test) # Predict class labels
        sk_accuracy = accuracy_score(y_test, y_predicts_sk)  # Proportion of correctly predicted samples

        accuracy_diff = abs(accuracy-sk_accuracy)

        #Results
        if self.do_print == True:
            print(f"Dataset: {dataset_name} \n"
                f"Custom k-NN accuracy: {accuracy} \n"
                f"Scikit implementation k-NN accuracy: {sk_accuracy} \n"
                f"Accuracy difference: {accuracy_diff} \n"
                f"---------------------\n")
        
        return accuracy,sk_accuracy,accuracy_diff
    
    def plot_classification(self, neighbors, X, y, data, test_size = 0.2,random_state = 42, dataset_name ="?", sample=0):
        """
        Visualizes k-NN classification on a 2D plot, 
        using **the first two features of the dataset**.
        
        :param neighbors: number of k neighbours
        :param X: feature matrix from scikit dataset
        :param y: target labels matrix from scikit dataset
        :param data: input data from scikit
        :param test_size: proportion of dataset to include in the test split
        :param random_state: random seed for reproducibility in train-test splitting
        :param dataset_name: 
        :param sample: index of the test sample
        """

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)

        feature_names = data.feature_names #names of axis
        target_names = data.target_names #names of plots

        # Pick first two features by default
        idx1, idx2 = 0, 1
        if len(feature_names) >= 2:
            idx1, idx2 = 0, 1  # first two features
        else:
            raise ValueError("Dataset must have at least 2 features for plotting.")
        
        # Use only two features for visualization - idx1 and idx2
        X_train_plot = X_train[:, [idx1, idx2]]
        X_test_plot = X_test[:, [idx1, idx2]]

        knn_plot = KNN(k=neighbors)
        knn_plot.fit(X_train_plot, y_train)

        sample = int(sample)
        x_sample = X_test_plot[sample]

        k_indices, distances = knn_plot.establish_neighbors(x_sample)

        predicted_class = knn_plot.predict([x_sample])[0]
        print(f"Predicted class for the sample: {predicted_class}")

        # Plot training points by class
        classes = np.unique(y_train)

        for cls in classes:
            label_selection = target_names[cls] if target_names is not None else f"Class {cls}"
            plt.scatter(
                X_train_plot[y_train == cls, 0],
                X_train_plot[y_train == cls, 1],
                alpha=0.4,
                label=f"{dataset_name} class {label_selection}"
            )

        # Plot test sample
        plt.scatter(
            x_sample[0], # first feature (x axis)
            x_sample[1], # second feature (y axis)
            color='red',
            marker='x',
            s=100,
            label=f'Test sample,\n predicted class = {predicted_class}'
        )

        # Highlight nearest neighbors
        plt.scatter(
            X_train_plot[k_indices, 0],
            X_train_plot[k_indices, 1],
            edgecolors='black',
            facecolors='none',
            s=120,
            label='Nearest neighbors'
        )

        plt.xlabel(feature_names[idx1])
        plt.ylabel(feature_names[idx2])
        plt.legend()
        plt.title(f"k-NN classification visualization ({dataset_name} dataset)")
        plt.show()


if __name__ == "__main__":
    # Parameters
    neighbors = 3
    test_size = 0.2
    random_state=42
    

    ### Wine Dataset ###
    # Prepare data using scikit sets
    data = load_wine()      
    X, y = data.data, data.target 
    # For plotting   
    dataset_name = "Wine"
    sample_index = 25

    # Perform tests
    test = TestKNN(do_print=True)

    test.accuracy_test(neighbors, X, y, test_size=test_size, random_state=random_state, dataset_name = dataset_name)
    
    test.plot_classification(
    neighbors, X, y,
    data=data,
    test_size=test_size,
    random_state=random_state,
    sample=sample_index,
    dataset_name=dataset_name
    )   


    ### Breast Cancer Dataset ###
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()     
    X, y = data.data, data.target 
    # For plotting   
    dataset_name = "Breast Cancer"
    sample_index = 25

    # Perform tests
    test = TestKNN(do_print=True)

    test.accuracy_test(neighbors, X, y, test_size=test_size, random_state=random_state, dataset_name = dataset_name)
    
    test.plot_classification(
    neighbors, X, y,
    data=data,
    test_size=test_size,
    random_state=random_state,
    sample=sample_index,
    dataset_name=dataset_name
    )