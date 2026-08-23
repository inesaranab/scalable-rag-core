"""The Redis gateway: cache hits, cache expiry, and the rate-limit counter."""

import fakeredis.aioredis
import pytest

from services.api.app.stores.redis_gateway import RedisGateway


@pytest.fixture
async def gateway():
    # A real Redis protocol implementation, in memory: same commands run.
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield RedisGateway(client)
    await client.aclose()


async def test_a_cached_answer_comes_back(gateway):
    await gateway.set_cached("question-hash", "the answer", ttl_s=60)

    assert await gateway.get_cached("question-hash") == "the answer"


async def test_a_cache_miss_is_none(gateway):
    assert await gateway.get_cached("never-stored") is None


async def test_cached_values_expire(gateway):
    await gateway.set_cached("brief", "gone soon", ttl_s=60)
    # Fast-forward: force-expire instead of sleeping in a test.
    await gateway.redis.expire("cache:brief", 0)

    assert await gateway.get_cached("brief") is None


async def test_requests_under_the_limit_pass(gateway):
    for _ in range(5):
        allowed = await gateway.check_rate_limit("ines", limit=5, window_s=60)

    assert allowed is True


async def test_the_request_over_the_limit_is_refused(gateway):
    for _ in range(5):
        await gateway.check_rate_limit("ines", limit=5, window_s=60)

    assert await gateway.check_rate_limit("ines", limit=5, window_s=60) is False


async def test_limits_are_per_subject(gateway):
    for _ in range(5):
        await gateway.check_rate_limit("ines", limit=5, window_s=60)

    assert await gateway.check_rate_limit("maarten", limit=5, window_s=60) is True
