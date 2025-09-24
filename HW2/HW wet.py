#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import numpy as np
import pandas as pd

class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.ids = (11111111, 2222222)
        self.classes = set()
        self.weight_vectors = []
        self.max_iterations = 0
        self.num_features = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding bias term
        self.classes = set(y)
        self.num_features = X_with_bias.shape[1]
        self.weight_vectors = np.zeros((len(self.classes), self.num_features), dtype=np.float32)
        self.max_iterations = X_with_bias.shape[0]

        while True:
            find_wrong_sample = False
            for j in range(self.max_iterations):
                xi = X_with_bias[j]
                yi = y[j]
                scores = np.dot(self.weight_vectors, xi)
                predicted_class = np.argmax(scores)
                if predicted_class != yi:
                    self.weight_vectors[yi] += xi
                    self.weight_vectors[predicted_class] -= xi
                    find_wrong_sample = True
            if not find_wrong_sample:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding bias term
        scores = np.dot(X_with_bias, self.weight_vectors.T)
        predictions = np.argmax(scores, axis=1)
        return predictions.astype(np.uint8)

if __name__ == "__main__":
    print("*" * 20)
    print("Started HW2_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)


# In[ ]:




