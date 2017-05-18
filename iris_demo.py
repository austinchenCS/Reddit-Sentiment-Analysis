# Demonstration of the iris classification dataset

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


# Iris Classification
def run_iris_demo():
    iris = load_iris()

    X = iris.data
    y = iris.target

    print(X.shape)
    print(y.shape)

    print(pd.DataFrame(X, columns=iris.feature_names).head())

    knn = KNeighborsClassifier()
    knn.fit(X, y)

    print(knn.predict([[3, 5, 4, 2]]))
