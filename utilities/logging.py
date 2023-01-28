import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("tennis_project")


def with_logging(logger: logging.Logger = logger, enabled: bool = True):
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if enabled:
                logger.info(f"Calling {function.__name__}")
                value = function(*args, **kwargs)
                logger.info(f"Finished calling {function.__name__}")
            else:
                value = function(*args, **kwargs)
            return value

        return wrapper

    return decorator


# with_default_logging = functools.partial(with_logging, logger=logger)
