import functools
from time import perf_counter
from typing import Any, Callable


def timer(enabled: bool = True, decimal_places: int = 4):
    """
    A function decorator that times the execution of the decorated function and prints the time taken.

    Parameters:
    function (callable): The function to be decorated.
    enabled (bool): A flag to enable/disable the timer. Default is True.
    decimal_places (int): The number of decimal places to round the time to. Default is 3.

    Returns:
    A decorated function that prints the execution time if the timer is enabled.
    """

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if enabled:
                before = perf_counter()
                value = function(*args, **kwargs)
                after = perf_counter()
                time_diff = after - before
                formatted_time = f"{time_diff:.{decimal_places}f}"
                print(
                    f"Function {function.__name__} took {formatted_time} secs"
                    " to run."
                )
            else:
                value = function(*args, **kwargs)
            return value

        return wrapper

    return decorator
