"""Gradient descent implemented from scratch.

Gradient descent is the process by which the output of a loss function is
minimized by successively calculating the gradient of a function with a given
input and reducing the it until the difference is met.

This example uses a hard-coded `f_prime` function as the "loss" function.
To perform gradient descent with another function, replace the contents
of `f_prime` with another one.
"""

from common import get_float_input, get_int_input

DEFAULT_LEARNING_RATE = 0.01

DEFAULT_DIFF_THRESHOLD = 0.001


def f_prime(x: float) -> float:
    """The derivative for a function (x-2)^4."""
    return 4. * (x - 2) ** 3


def should_stop_iterating(last_value: float, current_value: float,
                          threshold: float = DEFAULT_DIFF_THRESHOLD) -> bool:
    """Determine if a learner should stop learning.

    This checks if the absolute difference between the last value and current
    value is less than the given threshold.

    Returns:
        bool: True if the learner should stop iterating, False otherwise.
    """
    return abs(last_value - current_value) < threshold


def descend(initial_x: float, step_count: int, step_size: float = DEFAULT_LEARNING_RATE,
            callback=None) -> list:
    """Perform gradient descent.

    Args:
        initial_x (float): The starting x value
        step_size (float): The size of the step
        step_count (int): The number of times to descend
        callback (function): A function called upon each step

    Returns:
        list: A list of all values calculated during gradient descent.
    """
    values = []
    x_i = initial_x
    for x in range(1, step_count):
        new_x_i = x_i - step_size * f_prime(x_i)
        if should_stop_iterating(x_i, new_x_i):
            break
        x_i = new_x_i
        values.append(new_x_i)
        callback(new_x_i)
    return values


def print_update(x: float):
    """Logs a new iteration to standard output."""
    print(f'New value: {x}')


def main():
    """Perform gradient descent using user-provided parameters.

    This asks the user to provide a learning rate, an initial starting value,
    and a maximum step count to avoid infinite iteration.
    """
    learning_rate = get_float_input('Input a learning rate, a=')
    initial_x = get_float_input('Input an initial value, x0=')
    step_count = get_int_input('Input the number of steps:')
    print('Now descending...')
    descend(initial_x=initial_x, step_count=step_count, step_size=learning_rate,
            callback=print_update)


if __name__ == '__main__':
    main()
