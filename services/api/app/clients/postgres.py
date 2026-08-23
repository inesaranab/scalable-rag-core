"""Connection lifecycle for Postgres: reach it, check it, release it.

The user store receives the connected engine and owns the queries; this
class owns how the database is reached, including the driver naming
SQLAlchemy requires in its URLs.
"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)


class PostgresClient:
    """Owns the Postgres connection: connect, health, close.

    Attributes:
        dsn: The connection string. A plain ``postgresql://`` scheme is
            rewritten to name the asyncpg driver; any other async DSN
            (e.g. sqlite+aiosqlite, in tests) passes through unchanged.
        engine: The pooled engine; None until ``connect`` runs and again
            after ``close``.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.engine: AsyncEngine | None = None

    async def connect(self) -> None:
        """Build the pooled engine. Called once, at app startup."""
        # SQLAlchemy async URLs name their driver explicitly.
        self.engine = create_async_engine(
            self.dsn.replace("postgresql://", "postgresql+asyncpg://")
        )
        # The DSN embeds the password, so it stays out of the log.
        logger.info("postgres connected")

    async def close(self) -> None:
        """Close the pool. Called once, at app shutdown."""
        if self.engine is not None:
            await self.engine.dispose()
            self.engine = None
            logger.info("postgres closed")

    async def health(self) -> bool:
        """Say whether the database answers.

        Returns:
            True when SELECT 1 succeeds, False on any failure or
            before ``connect``.
        """
        if self.engine is None:
            return False
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.warning("postgres health check failed", exc_info=True)
            return False
