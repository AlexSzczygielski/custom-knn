# test.py
from knn import KNN
import numpy as np
from sklearn.datasets import load_wine #Dataset from scikit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    neighbours = 3

    # Prepare data using scikit sets        
    X, y = load_wine(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Custom k-NN implementation
    knn = KNN(k=neighbours)
    knn.fit(X_train,y_train)

    y_predicts = knn.predict(X_test) # Predict class labels
    accuracy = accuracy_score(y_test, y_predicts) # Proportion of correctly predicted samples

    # Scikit k-NN implementation
    sk_knn = KNeighborsClassifier(n_neighbors=neighbours)
    sk_knn.fit(X_train, y_train)

    y_predicts_sk = sk_knn.predict(X_test) # Predict class labels
    sk_accuracy = accuracy_score(y_test, y_predicts_sk)  # Proportion of correctly predicted samples

    #Results
    print(f"Custom k-NN accuracy: {accuracy} \n"
          f"Scikit implementation k-NN accuracy: {sk_accuracy} \n"
          f"Accuracy difference: {abs(accuracy-sk_accuracy)} \n")
    
    # Get feature names
    feature_names = load_wine().feature_names

    # Indices of features to visualize
    idx1 = feature_names.index("alcohol")
    idx2 = feature_names.index("flavanoids")

    # Use only two features for visualization
    X_vis = X[:, [idx1, idx2]]

    X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
        X_vis, y, test_size=0.2, random_state=42
    )

    knn_vis = KNN(k=neighbours)
    knn_vis.fit(X_train_vis, y_train_vis)

    x_sample = X_test_vis[10]

    distances = np.linalg.norm(X_train_vis - x_sample, axis=1)
    k_indices = np.argsort(distances)[:neighbours]

    # Plot training points by class
    classes = np.unique(y_train_vis)

    for cls in classes:
        plt.scatter(
            X_train_vis[y_train_vis == cls, 0],
            X_train_vis[y_train_vis == cls, 1],
            alpha=0.4,
            label=f"Wine class {cls}"
        )

    # Plot test sample
    plt.scatter(
        x_sample[0],
        x_sample[1],
        color='red',
        marker='x',
        s=100,
        label='Test sample'
    )

    # Highlight nearest neighbors
    plt.scatter(
        X_train_vis[k_indices, 0],
        X_train_vis[k_indices, 1],
        edgecolors='black',
        facecolors='none',
        s=120,
        label='Nearest neighbors'
    )

    plt.xlabel("Alcohol")
    plt.ylabel("Flavanoids")
    plt.legend()
    plt.title("k-NN classification visualization (Wine dataset)")
    plt.show()

    """
    X, y = load_wine(return_X_y=True)
    features = load_wine().feature_names

    idx1 = features.index("alcohol")
    idx2 = features.index("flavanoids")

    plt.scatter(X[:, idx1], X[:, idx2], c=y)
    plt.xlabel("Alcohol")
    plt.ylabel("Flavanoids")
    plt.title("Wine Dataset: Alcohol vs Flavanoids")
    plt.show()
    """
