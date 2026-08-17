"""Timing decorators: measure without altering what the caller receives."""

import logging

import pytest

from libs.utils.timing import measure_time, measure_time_async


def test_return_value_passes_through():
    """Timing is a side effect; the caller gets the function's own result."""

    @measure_time
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_arguments_pass_through_unchanged():
    """Positional and keyword arguments reach the wrapped function intact."""

    @measure_time
    def describe(a, b, sep=" "):
        return f"{a}{sep}{b}"

    assert describe("x", "y", sep="-") == "x-y"


def test_identity_is_preserved():
    """The wrapped function keeps its name, so log lines name it rather than
    the wrapper."""

    @measure_time
    def specific_name():
        return None

    assert specific_name.__name__ == "specific_name"


def test_duration_is_logged(caplog):
    """A completed call reports how long it took."""

    @measure_time
    def work():
        return None

    with caplog.at_level(logging.INFO):
        work()

    assert "work took" in caplog.text
    assert "ms" in caplog.text


def test_duration_is_logged_even_when_the_function_raises(caplog):
    """A failing call is the one whose duration matters most, so it is still
    reported, and the exception still propagates."""

    @measure_time
    def explode():
        raise ValueError("boom")

    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        explode()

    assert "explode took" in caplog.text


async def test_async_return_value_passes_through():
    """The asynchronous decorator awaits the call and returns its result."""

    @measure_time_async
    async def add(a, b):
        return a + b

    assert await add(2, 3) == 5


async def test_async_duration_is_logged(caplog):
    """An awaited call reports its duration."""

    @measure_time_async
    async def work():
        return None

    with caplog.at_level(logging.INFO):
        await work()

    assert "work took" in caplog.text


async def test_async_duration_is_logged_on_failure(caplog):
    """A failing awaited call still reports, and still raises."""

    @measure_time_async
    async def explode():
        raise ValueError("boom")

    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        await explode()

    assert "explode took" in caplog.text
