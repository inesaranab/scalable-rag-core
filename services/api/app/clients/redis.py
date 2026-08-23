"""Connection lifecycle for Redis: reach it, check it, release it.

The gateway (store layer) receives the connected client and owns the
cache and rate-limit operations; this class owns how Redis is reached.
"""

import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisClient:
    """Owns the Redis connection: connect, health, close.

    Attributes:
        url: The Redis server's address.
        client: The pooled client handing back ``str`` values; None
            until ``connect`` runs and again after ``close``.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.client: redis.Redis | None = None

    async def connect(self) -> None:
        """Build the pooled client. Called once, at app startup."""
        self.client = redis.from_url(
            self.url, encoding="utf-8", decode_responses=True
        )
        # The URL may embed a password, so it stays out of the log.
        logger.info("redis connected")

    async def close(self) -> None:
        """Close the pool. Called once, at app shutdown."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            logger.info("redis closed")

    async def health(self) -> bool:
        """Say whether the server answers.

        Returns:
            True when a PING round trip succeeds, False on any failure
            or before ``connect``.
        """
        if self.client is None:
            return False
        try:
            return await self.client.ping()
        except Exception:
            logger.warning("redis health check failed", exc_info=True)
            return False
