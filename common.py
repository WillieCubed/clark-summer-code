"""A set of utility functions for running programs from the console."""


def get_int_input(message: str = 'Input a number:') -> int:
    """
    Args:
        message (str): A message for the console

    Returns:
        A number (int) with user input.
    """
    while True:
        try:
            inputted = input(message)
            number = int(inputted)
            return number
        except ValueError:
            print('That is not a number.')


def get_float_input(message: str = 'Input a number:') -> float:
    """
    Args:
        message (str): A message for the console

    Returns:
        A number (int) with user input.
    """
    while True:
        try:
            inputted = input(message)
            number = float(inputted)
            return number
        except ValueError:
            print('That is not a number.')


