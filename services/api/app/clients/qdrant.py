"""Connection lifecycle for Qdrant: reach it, check it, release it.

The store layer receives the connected SDK object and owns the
operations; this class owns how the database is reached.
"""

import logging

from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)


class QdrantClient:
    """Owns the Qdrant connection: connect, health, close.

    Attributes:
        url: The Qdrant server's address.
        client: The connected SDK client; None until ``connect`` runs
            and again after ``close``.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """Build the pooled client. Called once, at app startup."""
        self.client = AsyncQdrantClient(url=self.url)
        logger.info("qdrant connected", extra={"url": self.url})

    async def close(self) -> None:
        """Close the pool. Called once, at app shutdown."""
        if self.client is not None:
            await self.client.close()
            self.client = None
            logger.info("qdrant closed")

    async def health(self) -> bool:
        """Say whether the database answers.

        Returns:
            True when a round trip succeeds, False on any failure or
            before ``connect``.
        """
        if self.client is None:
            return False
        try:
            await self.client.get_collections()
            return True
        except Exception:
            logger.warning("qdrant health check failed", exc_info=True)
            return False
