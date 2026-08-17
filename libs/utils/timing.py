"""Decorators that record how long a wrapped function took.

Attributing latency to a specific step is what separates optimising the slow
part from guessing at it. The measurement uses a monotonic clock, so a system
clock adjustment mid-call cannot produce a negative or wildly wrong duration.
"""

import functools
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def measure_time[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Log the duration of a synchronous function.

    Args:
        func: The function to time.

    Returns:
        The same function, wrapped so that each call logs its duration in
        milliseconds at INFO level. Return value and exceptions pass through
        unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("%s took %.2f ms", func.__qualname__, elapsed_ms)

    return wrapper


def measure_time_async[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Log the duration of an asynchronous function.

    Args:
        func: The coroutine function to time.

    Returns:
        The same function, wrapped so that each awaited call logs its duration
        in milliseconds at INFO level. Return value and exceptions pass
        through unchanged.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("%s took %.2f ms", func.__qualname__, elapsed_ms)

    return wrapper
