"""Gradient descent implemented from scratch."""

import numpy as np

from common import get_float_input, get_int_input

DEFAULT_LEARNING_RATE = 0.01

DEFAULT_DIFF_THRESHOLD = 0.001


def f_prime(x: float) -> float:
    """The derivative for a function (x-2)^4."""
    return 4. * (x - 2) ** 3


def should_stop_iterating(last_value: float, current_value: float, threshold: float) -> bool:
    """Return True if the """
    return abs(last_value - current_value) < threshold


def descend(initial_x: float, step_count: int, step_size: float = DEFAULT_LEARNING_RATE,
            callback=None) -> list:
    """Perform gradient descent.

    Args:
        initial_x (float):
        step_size (float):
        step_count (int): The number of times to descend.
        callback (function): A function called upon each step.
    """
    values = []
    x_i = initial_x
    for x in range(1, step_count):
        new_x_i = x_i - step_size * f_prime(x_i)
        if should_stop_iterating(x_i, new_x_i, DEFAULT_DIFF_THRESHOLD):
            break
        x_i = new_x_i
        values.append(new_x_i)
        callback(new_x_i)
    return values


def partial_derivative() -> float:
    """Return d(Wx+b)/dw"""
    pass


def calculate_i(condition) -> int:
    """Return 0 or 1 based on the truth of the condition."""
    if condition():
        return 1
    return 0


def check_classification(y, w, x, b) -> bool:
    return y * w * x + b < 0


def compute_w(w_transposed: np.ndarray, w_t: int, last_bias: int, inputs: list, alpha: float) -> np.ndarray:
    y_sum = 0
    m = len(inputs)
    for i in range(m):
        x_m, y_m = inputs[i]
        y_sum += y_m * x_m * calculate_i(lambda: check_classification(y=y_m, w=w_t, x=x_m, b=last_bias))
    return w_transposed + alpha * y_sum


def compute_b(last_bias: float, y: tuple, alpha: float, w_t: int) -> float:
    y_sum = 0
    m = len(y)
    for i in range(m):
        x_m, y_m = y[i]
        y_sum += y_m * calculate_i(lambda: check_classification(y=y_m, w=w_t, x=x_m, b=last_bias))
    return last_bias + alpha * y_sum


def print_update(x: float):
    print(f'New value: {x}')


def compute_delta() -> float:
    pass


class Perceptron:

    def __init__(self):
        pass

    def go(self):
        pass


def main():
    """Perform gradient descent using user-provided parameters."""
    learning_rate = get_float_input('Input a learning rate, a=')
    initial_x = get_float_input('Input an initial value, x0=')
    step_count = get_int_input('Input the number of steps:')
    print('Now descending...')
    descend(initial_x=initial_x, step_count=step_count, step_size=learning_rate, callback=print_update)


if __name__ == '__main__':
    main()
