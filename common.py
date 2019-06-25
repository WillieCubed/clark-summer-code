"""A set of functions for running machine learning programs from the console.

Includes utility functions for getting validated float and int inputs
(get_float_input and get_int_input) along with functions for getting a csv
dataset and splitting it into training, validation, and test sets.
"""

import numpy as np


def get_dataset(directory: str, include_feature_labels: bool = True,
                label_first: bool = False) -> (
        np.ndarray, np.ndarray, list):
    """Get all values in a csv dataset.

    Args:
        directory (str): A path to a CSV file
        include_feature_labels (bool): True if labels for features are the first line of the file
        label_first (bool): True if the output label is at the beginning of each example, False if it is at the end
    Returns:
        A tuple containing a numpy array for the x, and y values for the
        dataset along with a list for the labels of the dataset if
        include_labels was True.
    """
    features_list = np.array([[]])
    labels = np.array([])

    with open(directory) as f:
        if include_feature_labels:
            feature_labels = f.readline().split(',')
        else:
            feature_labels = None
        for example in f.readlines():
            values = example.split(',')
            values[-1] = values[-1][:-1]  # Remove the newline because I'm stupid
            print(values)
            for value in values:
                if label_first:
                    label = int(value[0])
                    features = float(value[1:])
                else:
                    label = int(value[-1])
                    features = float(value[0:])
                features_list = np.append(features_list, features)
                labels = np.append(labels, label)
    return features_list, labels, feature_labels


def split_data(x: np.ndarray, y: np.ndarray, amounts: tuple = None,
               split_randomly: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    """Split a dataset into a training, validation, and test dataset.

    Args:
        x (np.ndarray): Values for the features of a dataset.
        y (np.ndarray): Values for the labels of a dataset. Must be same size as x.
        amounts (list): Percent for training/validation/test. If not provided, a 70/20/10 split will be used.
        split_randomly (bool): True if sample order should be randomized

    Returns:
        A tuple containing the split training, validation, and test datasets.
    """
    # TODO: Find some more efficient way to do this.
    if amounts is None:
        amounts = (0.7, 0.2, 0.1)
    total_length = len(y)
    train_percent, validation_percent, test_percent = amounts
    validation_lower_bound = int(total_length * train_percent)
    test_lower_bound = validation_lower_bound + int(
        total_length * validation_lower_bound)
    if split_randomly:
        group = np.random.rand(list(zip(x, y)))
    else:
        group = list(zip(x, y))
    return group[:validation_lower_bound], \
           group[validation_lower_bound:test_lower_bound], \
           group[test_lower_bound:]


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
