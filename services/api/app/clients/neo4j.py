"""Connection lifecycle for Neo4j: reach it, check it, release it.

The store layer receives the connected driver and owns the Cypher; this
class owns how the database is reached.
"""

import logging

from neo4j import AsyncDriver, AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Owns the Neo4j connection: connect, health, close.

    Attributes:
        uri: The database address (bolt:// scheme).
        user: Login username.
        password: Login password.
        driver: The connected driver; None until ``connect`` runs and
            again after ``close``.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Build the pooled driver. Called once, at app startup."""
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        # Credentials travel separately, so the URI is safe to log.
        logger.info("neo4j connected", extra={"uri": self.uri})

    async def close(self) -> None:
        """Close the pool. Called once, at app shutdown."""
        if self.driver is not None:
            await self.driver.close()
            self.driver = None
            logger.info("neo4j closed")

    async def health(self) -> bool:
        """Say whether the database answers.

        Returns:
            True when a round trip succeeds, False on any failure or
            before ``connect``.
        """
        if self.driver is None:
            return False
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            logger.warning("neo4j health check failed", exc_info=True)
            return False
