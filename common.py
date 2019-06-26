"""A set of functions for running machine learning programs from the console.

Includes utility functions for getting validated float and int inputs
(get_float_input and get_int_input) along with functions for getting a csv
dataset and splitting it into training, validation, and test sets
(split_data, get_dataset).
"""

import numpy as np
import pandas as pd


def get_dataset(directory: str) -> np.ndarray:
    """Get all values in a csv dataset.

    Args:
        directory (str): A path to a CSV file

    Returns:
        np.ndarray: An array containing rows each for features and a label.
    """
    return pd.read_csv(directory).values


def split_data(data: np.ndarray, amounts: tuple = None) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """Split a dataset into a training, validation, and test dataset.

    Args:
        data (np.ndarray): An array containing rows for each data point and columns for each feature.
            The last column must be the label.
        amounts (list): Percent for training/validation/test. If not provided, a 70/20/10 split will be used.

    Returns:
        tuple: A tuple containing the split training, validation, and test datasets.
    """
    if amounts is None:
        amounts = (0.7, 0.2, 0.1)
    total_length = len(data)
    features = []
    labels = []
    for example in data:
        features.append(example[:-1])
        labels.append(example[-1])
    train_percent, validation_percent, test_percent = amounts
    validation_lower_bound = int(total_length * train_percent)
    test_lower_bound = validation_lower_bound + int(
        total_length * validation_percent)
    training = {
        'x': features[:validation_lower_bound],
        'y': labels[:validation_lower_bound]
    }
    validation = {
        'x': features[validation_lower_bound:test_lower_bound],
        'y': labels[validation_lower_bound:test_lower_bound]
    }
    test = {
        'x': features[test_lower_bound:],
        'y': labels[test_lower_bound:]
    }
    return training, validation, test


def get_int_input(message: str = 'Input a number:') -> int:
    """Returns a validated integer user input.

    Args:
        message (str): A message for the console

    Returns:
        int: A number with user input.
    """
    while True:
        try:
            inputted = input(message)
            number = int(inputted)
            return number
        except ValueError:
            print('That is not a number.')


def get_float_input(message: str = 'Input a number:') -> float:
    """Returns a validated long user input.

    Args:
        message (str): A message for the console

    Returns:
        int: A number with user input.
    """
    while True:
        try:
            inputted = input(message)
            number = float(inputted)
            return number
        except ValueError:
            print('That is not a number.')
