"""Retrying a failed call with growing, randomised delays.

Retrying immediately makes a struggling service worse. Doubling the wait after
each attempt gives it room to recover, and adding a random offset stops many
clients that failed together from returning together — a synchronised retry
storm that recreates the original overload.
"""

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def exponential_backoff[**P, R](
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
    retry_on: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Build a decorator that retries an async function on failure.

    The wait before attempt n is ``base_delay * 2 ** n``, capped at
    ``max_delay``, plus a random offset up to ``jitter`` seconds.

    Args:
        max_retries: How many times to retry after the first failure. The
            function is therefore called at most ``max_retries + 1`` times.
        base_delay: Seconds to wait before the first retry.
        max_delay: Ceiling on the doubling, before the random offset.
        jitter: Upper bound on the random offset added to every wait.
        retry_on: Exception type, or tuple of types, worth retrying. Anything
            else propagates immediately.

    Returns:
        A decorator that wraps an async function with this retry policy.

    Raises:
        ValueError: If max_retries is negative, or any delay is negative.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must not be negative, got {max_retries}")
    if base_delay < 0 or max_delay < 0 or jitter < 0:
        raise ValueError("delays and jitter must not be negative")

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:
                    if attempt == max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__qualname__,
                            attempt + 1,
                            type(exc).__name__,
                        )
                        raise
                    wait = min(base_delay * 2**attempt, max_delay)
                    wait += random.uniform(0, jitter)  # jitter
                    logger.warning(
                        "%s raised %s, retrying in %.2fs (attempt %d of %d)",
                        func.__qualname__,
                        type(exc).__name__,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
            raise AssertionError("unreachable: the loop either returns or raises")

        return wrapper

    return decorator
