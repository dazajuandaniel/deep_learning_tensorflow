"""
Support file that holds functions used across notebooks
"""
import time
import os
from tensorflow import keras

def get_run_logdir(folder_name: str = 'my_logs'):
    """
    Function that creates a new directory with the current time

    Args:
        folder_name(str)

    Returns:
        (str): The path of the created folder
    """
    root_logdir = os.path.join(os.curdir, folder_name)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def get_fashion_mnist_data():
    """
    Support Function that gets the MNIST Fashion data
    """
    # Load Data from Keras
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Get Validation/Test set
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0

    return X_train, X_test, X_valid, y_train, y_test, y_valid
    