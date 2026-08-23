"""The API's gateway to Redis: response cache and rate limiting.

Redis is an in-memory key-value store with per-key expiry, which is
exactly what both jobs need: cached answers that go stale on their own,
and request counters that reset when their time window ends.
"""

import redis.asyncio as redis

CACHE_PREFIX = "cache:"
RATE_PREFIX = "rate:"


class RedisGateway:
    """Cache and rate limiting over an injected Redis connection.

    Attributes:
        redis: The pooled async client, built by the client layer and
            handed in; this class owns operations, never the connection.
    """

    def __init__(self, client: redis.Redis) -> None:
        self.redis = client

    async def get_cached(self, key: str) -> str | None:
        """Return the cached value for a key, or None on miss or expiry.

        Args:
            key: The cache key (e.g. a hash of the question).

        Returns:
            The stored value, or None when absent.
        """
        return await self.redis.get(CACHE_PREFIX + key)

    async def set_cached(self, key: str, value: str, ttl_s: int) -> None:
        """Store a value that expires by itself.

        Args:
            key: The cache key.
            value: What to store.
            ttl_s: Seconds until Redis deletes the key on its own.
        """
        await self.redis.set(CACHE_PREFIX + key, value, ex=ttl_s)

    async def check_rate_limit(
        self, subject: str, limit: int, window_s: int
    ) -> bool:
        """Count one request for a subject; say whether it is allowed.

        The counter's key lives for one window and then vanishes, so the
        allowance refills automatically — no cleanup job needed.

        Args:
            subject: Who is asking (e.g. the JWT's ``sub`` claim).
            limit: Requests allowed per window.
            window_s: The window's length, in seconds.

        Returns:
            True while the subject is at or under the limit; False once
            this request would exceed it.
        """
        key = RATE_PREFIX + subject
        count = await self.redis.incr(key)
        if count == 1:
            # First request of a fresh window starts the countdown.
            await self.redis.expire(key, window_s)
        return count <= limit
