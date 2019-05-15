from sklearn.model_selection import train_test_split

from load_datasets import load_datasets
import numpy as np

X, y = load_datasets()

def load_test_data():
    X_test = X[650:]
    y_test = y[650:]
    return X_test, y_test

def load_train_valid_data(test_split):
    X_tmp = X[:650]
    y_tmp = y[:650]
    X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=test_split, random_state=1)

    return  X_train, X_valid, y_train,  y_valid






