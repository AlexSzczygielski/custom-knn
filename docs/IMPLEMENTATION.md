# Algorithm Implementation

The K-Nearest Neighbors (K-NN) algorithm implemented in this project follows a straightforward procedure for classifying new data points based on their proximity to labeled training data. The process is visualized in the [State Diagram](./STATE_DIAGRAM.md).

---

## Steps of Implementation

1. **Initialization**
   - Create a KNN instance with a specified number of neighbors `k`.
   - Initialize internal variables for storing training data (`_X_train`) and labels (`_y_train`).

2. **Training (`fit` method)**
   - Accept a set of training feature vectors `X` and corresponding labels `y`.
   - Store them as NumPy arrays for efficient computation.
   - This step does not perform traditional model fitting; it simply prepares the data for distance calculations.

3. **Prediction (`predict` method)**
   - Convert incoming test samples into NumPy arrays.
   - Iterate over each sample and call `_predict_single` to determine its class.

4. **Single Sample Prediction (`_predict_single` method)**
   - Compute the distance from the test sample to all training points using **Euclidean distance**.
   - Identify the `k` nearest neighbors by sorting distances and selecting the closest points.
   - Retrieve the class labels of these neighbors.
   - Count the frequency of each label.
   - Assign the label with the highest frequency as the predicted class for the sample.

5. **Neighbor Calculation (`establish_neighbors` method)**
   - Returns both the indexes and labels of the `k` nearest neighbors for a given sample.
   - Used internally by `_predict_single` and optionally for visualization.

6. **Output**
   - Return an array of predicted labels for all input test samples.
   - For visualization, the nearest neighbors can be highlighted in plots (see the `plot_classification` method).

---

> **Visualization Reference:**  
> See the complete algorithm flow in the [K-NN State Diagram](./STATE_DIAGRAM.md).

---

## Notes

- Choosing the right `k` is critical:
  - Too low → overfitting
  - Too high → smoothing, possible underfitting
