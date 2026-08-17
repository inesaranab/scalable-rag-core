"""Retry policy: how many attempts, how long between them, and what is retried."""

import pytest

from libs.retry import backoff
from libs.retry.backoff import exponential_backoff


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Replace the sleep with a recorder, so waits are asserted not endured.

    Args:
        monkeypatch: Pytest's attribute-patching fixture.

    Returns:
        A list that receives every requested wait, in order.
    """
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(backoff.asyncio, "sleep", fake_sleep)
    return waits


async def test_a_successful_call_is_not_retried(recorded_sleeps):
    """Nothing is waited on when the first attempt succeeds."""
    calls = 0

    @exponential_backoff(max_retries=3)
    async def works():
        nonlocal calls
        calls += 1
        return "fine"

    assert await works() == "fine"
    assert calls == 1
    assert recorded_sleeps == []


async def test_it_retries_until_it_succeeds(recorded_sleeps):
    """A call that recovers on a later attempt returns that attempt's result."""
    calls = 0

    @exponential_backoff(max_retries=3, jitter=0.0)
    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError
        return "recovered"

    assert await flaky() == "recovered"
    assert calls == 3


async def test_the_wait_doubles_between_attempts(recorded_sleeps):
    """Each wait is twice the previous, so a struggling service is given
    progressively more room."""
    calls = 0

    @exponential_backoff(max_retries=3, base_delay=1.0, jitter=0.0)
    async def always_fails():
        nonlocal calls
        calls += 1
        raise ConnectionError

    with pytest.raises(ConnectionError):
        await always_fails()

    assert recorded_sleeps == [1.0, 2.0, 4.0]


async def test_the_wait_is_capped(recorded_sleeps):
    """Doubling stops at the ceiling, so no attempt waits indefinitely."""

    @exponential_backoff(max_retries=4, base_delay=1.0, max_delay=3.0, jitter=0.0)
    async def always_fails():
        raise ConnectionError

    with pytest.raises(ConnectionError):
        await always_fails()

    assert recorded_sleeps == [1.0, 2.0, 3.0, 3.0]


async def test_jitter_is_added_within_its_bound(recorded_sleeps):
    """A random offset separates clients that failed together, and never
    exceeds the bound it was given."""

    @exponential_backoff(max_retries=3, base_delay=1.0, jitter=0.5)
    async def always_fails():
        raise ConnectionError

    with pytest.raises(ConnectionError):
        await always_fails()

    for wait, floor in zip(recorded_sleeps, [1.0, 2.0, 4.0]):
        assert floor <= wait <= floor + 0.5


async def test_the_original_exception_propagates(recorded_sleeps):
    """The caller sees the failure that actually occurred, not a wrapper."""

    @exponential_backoff(max_retries=1, jitter=0.0)
    async def always_fails():
        raise TimeoutError("upstream gone")

    with pytest.raises(TimeoutError, match="upstream gone"):
        await always_fails()


async def test_exceptions_outside_retry_on_are_not_retried(recorded_sleeps):
    """A programming error is not transient, so retrying it only hides it."""
    calls = 0

    @exponential_backoff(max_retries=3, retry_on=ConnectionError, jitter=0.0)
    async def bad_code():
        nonlocal calls
        calls += 1
        raise TypeError("not a transient failure")

    with pytest.raises(TypeError):
        await bad_code()

    assert calls == 1
    assert recorded_sleeps == []


async def test_zero_retries_calls_once(recorded_sleeps):
    """A policy of no retries still calls the function, exactly once."""
    calls = 0

    @exponential_backoff(max_retries=0)
    async def always_fails():
        nonlocal calls
        calls += 1
        raise ConnectionError

    with pytest.raises(ConnectionError):
        await always_fails()

    assert calls == 1


def test_negative_retries_are_rejected():
    """A nonsensical policy fails when it is declared, not when it is used."""
    with pytest.raises(ValueError, match="max_retries"):
        exponential_backoff(max_retries=-1)


def test_negative_delays_are_rejected():
    """A negative wait has no meaning and is refused at declaration."""
    with pytest.raises(ValueError, match="negative"):
        exponential_backoff(base_delay=-1.0)


async def test_identity_is_preserved():
    """The wrapped function keeps its name, so failures name it in the log."""

    @exponential_backoff(max_retries=0)
    async def specific_name():
        return None

    assert specific_name.__name__ == "specific_name"
