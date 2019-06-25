"""A perceptron implemented almost from scratch.

This perceptron implementation uses the numpy library to reduce
mathematical boilerplate.
"""

import numpy as np

from common import get_dataset, split_data

DATA_DIRECTORY = 'data/mystery.data'

DEFAULT_STEP_SIZE = 0.01

DEFAULT_DIFF_THRESHOLD = 0.01  # TODO: Tune hyperparameters


def should_stop_iterating(last_value: float, current_value: float,
                          threshold: float = DEFAULT_DIFF_THRESHOLD) -> bool:
    """Determine if a learner should stop learning.

    This checks if the absolute difference between the last value and current
    value is less than the given threshold.

    Returns:
        bool: True if the learner should stop iterating, False otherwise.
    """
    return abs(last_value - current_value) < threshold


def calculate_i(condition) -> int:
    """Return 0 or 1 based on the truth of the condition."""
    if condition():
        return 1
    return 0


def check_classification(y: int, w: np.ndarray, x: float, b: float) -> bool:
    """Determine if the input was classified incorrectly.

    FIXME: Find out why this looks so weird.
    """
    return y * w * x + b < 0


def determine_stop(last_value: float, current_value: float,
                   threshold: float = DEFAULT_DIFF_THRESHOLD) -> bool:
    """Determine if a learner should stop learning.

    This checks if the absolute difference between the last value and current
    value is less than the given threshold.

    Returns:
        bool: True if the learner should stop iterating, False otherwise.
    """
    return abs(last_value - current_value) < threshold


def sign(thing: int) -> int:
    """Performs a sign calculation.

    Returns:
        int: -1 if the input is negative, 1 if the input is positive,
        and 0 otherwise.
    """
    if thing < 0:
        return -1
    elif thing > 0:
        return 1
    else:
        return 0


def compute_w(last_weights: np.ndarray, last_bias: np.ndarray,
              inputs: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the next set of weights given the current input.

    A part of gradient descent.

    Returns:
        np.ndarray: An array of weights the same size as `last_weights`.
    """
    diff_sum = 0
    m = len(inputs)
    for i in range(m):
        x_m, y_m = inputs[i]
        diff_sum += y_m * x_m * calculate_i(lambda: check_classification(y=y_m, w=last_weights, x=x_m, b=last_bias))
    return last_weights.transpose() + alpha * diff_sum


def compute_b(last_biases: np.ndarray, inputs: tuple, alpha: float, last_weights: np.ndarray) -> np.ndarray:
    """Compute the next set of biases given the current input.

    A part of gradient descent.

    Returns:
        np.ndarray: An array of weights the same size as `last_biases`.
    """
    diff_sum = 0
    m = len(inputs)
    for i in range(m):
        x_m, y_m = inputs[i]
        diff_sum += y_m * calculate_i(lambda: check_classification(y=y_m, w=last_weights, x=x_m, b=last_biases))
    return last_biases + alpha * diff_sum


class Perceptron:
    """A perceptron implementation used for linear regression.

    This perceptron can be trained using `train` and make an inference
    using `infer`.

    Training uses gradient descent to
    """

    def __init__(self, x_array: np.ndarray, y_array: np.ndarray):
        """Create a perceptron and initialize it with the given data.

        Args:
            x_array (np.ndarray): Features from the dataset
            y_array (np.ndarray): Labels from the dataset
        """
        self._x = x_array
        self._y = y_array
        self.weights = np.zeros((len(y_array), len(x_array)))
        self.biases = np.zeros((len(y_array), 1))

    @staticmethod
    def _predict(weights: np.ndarray, x: np.ndarray,
                 biases: np.ndarray) -> int:
        """Get the value of this perceptron with the given inputs.

        Args:
            weights (np.ndarray): A 1 x N array of weights
            x (np.ndarray): An N dimensional array of feature values
            biases (np.ndarray): A 1 x N array of biases

        Returns:
            int: The predicted value (1 or 0).
        """
        return sign(np.dot(weights.transpose(), x) + biases)

    def _check(self, y_i: int, weights: np.ndarray, x: np.ndarray,
               biases: np.ndarray) -> bool:
        """Check if the given example matches this perceptron's prediction.

        Args:
            y_i : Either 1 or 0
            weights (np.ndarray): An array of weights
            x (np.ndarray): An array of x values
            biases (np.ndarray): An array of biases

        Returns:
            The predicted value (True or False).
        """
        return self._predict(weights, x, biases) == y_i

    def train(self, alpha=DEFAULT_STEP_SIZE):
        """Minimize loss by continuously adjusting weights and biases.

        This training function performs standard gradient descent to
        minimize the loss function L(w,b).
        """
        for x_batch, y_batch in zip(self._x, self._y):
            new_weights = compute_w(self.weights, self.biases, self._x, alpha)
            new_biases = compute_b(self.biases, (self._x, self._y), alpha, self.weights)
            self.weights = new_weights
            self.biases = new_biases

    def infer(self, x: np.ndarray) -> bool:
        """Perform inference with the given data.

        Should be done after calling `train`.

        Args:
            x (np.ndarray): A vector representing the features of the data.

        Returns:
            bool: The output of this perceptron.
            True if this perceptron labels the
        """
        return True if self._predict(self.weights, x, self.biases) == 1 else False


def determine_accuracy(perceptron: Perceptron, test_data: tuple) -> float:
    """Determine the accuracy of a perceptron.

    Args:
        perceptron (Perceptron): The perceptron used to evaluate accuracy
        test_data (tuple): A pair of np.ndarray used to store the test data

    Returns:
        float: The amount of correctly classified examples divided by the total
        amount of examples.
    """
    correct = 0
    for x, y in test_data:
        inference = perceptron.infer(x)
        if inference == y:
            correct += 1
    return correct / len(test_data)


def main():
    """Train and test a perceptron implementation.

    This takes data at the location defined by DATA_DIRECTORY, splits it into and creates
    a perceptron to predict values.
    """
    x, y, labels = get_dataset(directory=DATA_DIRECTORY, include_feature_labels=True,
                               label_first=False)
    training, validation, test = split_data(x, y)
    perceptron = Perceptron(training[0], training[1])
    # TODO: Use validation set to tune alpha
    perceptron.train(alpha=DEFAULT_STEP_SIZE)
    accuracy = determine_accuracy(perceptron, test)
    print(f'The perceptron is {accuracy}% accurate. Woo.')


if __name__ == '__main__':
    main()
