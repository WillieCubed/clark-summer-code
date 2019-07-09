"""A perceptron implemented almost from scratch.

This perceptron implementation uses the numpy library to reduce
mathematical boilerplate.
"""

import numpy as np

from common import get_dataset, split_data

DATA_DIRECTORY = 'data/mystery.data'

DEFAULT_STEP_SIZE = 0.03

DEFAULT_DIFF_THRESHOLD = 0.001  # TODO: Tune hyperparameters

NUMBER_OF_EPOCHS = 1000


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
    result = y * w.transpose().dot(x) + b < 0
    return result


def determine_stop(last_value: float, current_value: float,
                   threshold: float = DEFAULT_DIFF_THRESHOLD) -> bool:
    """Determine if a learner should stop learning.

    This checks if the absolute difference between the last value and current
    value is less than the given threshold.

    Returns:
        bool: True if the learner should stop iterating, False otherwise.
    """
    return abs(last_value - current_value) < threshold


def compute_new_weights(last_weights: np.ndarray, last_bias: float,
                        inputs: tuple, alpha: float) -> np.ndarray:
    """Compute the next set of weights given the current input.

    A part of gradient descent.

    Returns:
        np.ndarray: An array of weights the same size as `last_weights`.
    """
    diff_sum = 0
    for x, y in zip(inputs[0], inputs[1]):
        result = y * last_weights.transpose().dot(x) + last_bias
        i = 0 if result < 0 else 1
        diff_sum += x * y * i
        # diff_sum += x * y * calculate_i(
        #     lambda: check_classification(y=y, w=last_weights, x=x, b=last_bias))
    return last_weights.transpose() + alpha * diff_sum


def compute_new_bias(last_weights: np.ndarray, last_bias: float, inputs: tuple,
                     alpha: float) -> float:
    """Compute the next set of biases given the current input.

    A part of gradient descent.

    Returns:
        np.ndarray: An array of weights the same size as `last_biases`.
    """
    diff_sum = 0
    for x, y in zip(inputs[0], inputs[1]):
        result = y * last_weights.transpose().dot(x) + last_bias
        i = 0 if result < 0 else 1
        diff_sum += y * i
        # diff_sum += y * calculate_i(
        #     lambda: check_classification(y=y, w=last_weights, x=x, b=last_bias))
    # print(f'last_bias: {type(last_bias)}, alpha: {type(alpha)}, diff_sum: {type(diff_sum)}')
    return last_bias + alpha * diff_sum


class Perceptron:
    """A perceptron implementation used for linear regression.

    This perceptron can be trained using `train` and make an inference
    using `infer`.

    Training uses gradient descent.
    """

    def __init__(self):
        """Initialize a perceptron with weights of 0 and a bias of 0."""
        self.weights = None
        self.bias = 0.

    @staticmethod
    def _predict(weights: np.ndarray, x: np.ndarray,
                 bias: float) -> int:
        """Get the value of this perceptron with the given inputs.

        Args:
            weights (np.ndarray): A 1 x N array of weights.
            x (np.ndarray): An N dimensional array of feature values.
            bias (float): A scalar bias.

        Returns:
            int: The predicted value (1 or 0).
        """
        print(f'Weights: {weights.dot(x)}, bias: {bias}')
        return np.sign(weights.transpose().dot(x) + bias)

    def _check(self, y_i: int, weights: np.ndarray, x: np.ndarray,
               bias: float) -> bool:
        """Check if the given example matches this perceptron's prediction.

        Args:
            y_i : Either 1 or 0
            weights (np.ndarray): An array of weights
            x (np.ndarray): An array of x values
            biases (np.ndarray): An array of biases

        Returns:
            The predicted value (True or False).
        """
        return self._predict(weights, x, bias) == y_i

    def train(self, x: np.ndarray, y: np.ndarray, alpha: float = DEFAULT_STEP_SIZE, epochs: int = NUMBER_OF_EPOCHS):
        """Minimize loss by continuously adjusting weights and biases.

        This training function performs standard gradient descent to
        minimize the loss function L(w,b).

        Args:
            x (np.ndarray): Features from the dataset.
            y (np.ndarray): Labels from the dataset.
            alpha (float): The learning rate, or how serverely weights are altered for each iteration of training data.
            epochs (int): The number of iterations to train.
        """
        len_x = len(x)
        len_y = len(y)
        if len_x != len_y:
            raise ValueError('x should be same length as y.')
        self.weights = np.zeros((len(x[0])))
        for _ in range(epochs):
            inputs = (x, y)
            new_weights = compute_new_weights(self.weights, self.bias, inputs, alpha)
            self.weights = new_weights
            new_bias = compute_new_bias(self.weights, self.bias, inputs, alpha)
            self.bias = new_bias

    def infer(self, x: np.ndarray) -> int:
        """Perform inference with the given data.

        Should be done after calling `train`.

        Args:
            x (np.ndarray): A vector representing the features of the data.

        Returns:
            int: The output of this perceptron. (1 indicating True and -1 if false)
        """
        if self.weights is None:
            raise ValueError('Train must be called before inference.')

        return np.sign(self._predict(self.weights, x, self.bias))


def determine_accuracy(perceptron: Perceptron, test_data: dict) -> float:
    """Determine the accuracy of a perceptron.

    Args:
        perceptron (Perceptron): The perceptron used to evaluate accuracy
        test_data (tuple): A pair of np.ndarray used to store the test data

    Returns:
        float: The amount of correctly classified examples divided by the total
        amount of examples.
    """
    correct = 0
    features = test_data['x']
    labels = test_data['y']
    total = len(labels)
    if len(features) != len(labels):
        raise ValueError('x must be the same length as y.')
    for i in range(len(features)):
        x = features[i]
        y = labels[i]
        inference = perceptron.infer(x)
        print(f'Does {inference} == {y}?')
        if inference == y:
            print('Yep!')
            correct += 1
        print(f'{correct}/{i}')
    return correct / total


def main():
    """Train and test a perceptron implementation.

    This takes data at the location defined by DATA_DIRECTORY, splits it into and creates
    a perceptron to predict values.
    """
    data = get_dataset(DATA_DIRECTORY)
    training, validation, test = split_data(data)
    perceptron = Perceptron()
    # TODO: Use validation set to tune alpha
    x, y = training['x'], training['y']
    perceptron.train(x, y, alpha=DEFAULT_STEP_SIZE)
    accuracy = determine_accuracy(perceptron, test)
    print(f'The perceptron is {int(accuracy * 100)}% accurate. Woo.')


if __name__ == '__main__':
    main()
