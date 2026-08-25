"""The feedback table: user verdicts on answers, the seed of training data.

Rows are written but not yet read: the first consumer will be a quality
metric over scores.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from services.api.app.db_models import Feedback


class FeedbackStore:
    """Reads and writes Feedback rows.

    Attributes:
        engine: The async database engine (Postgres in production,
            SQLite in tests — same code path).
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

    async def ensure_table(self) -> None:
        """Create the schema if absent. Safe to call at every boot."""
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def add(
        self,
        session_id: str,
        user_id: str,
        score: int,
        comment: str | None = None,
        message_id: str | None = None,
    ) -> None:
        """Record one verdict.

        Args:
            session_id: The conversation the verdict belongs to.
            user_id: Who judged (the JWT subject).
            score: -1, 0 or 1.
            comment: Optional free-text explanation.
            message_id: Optional turn identifier.
        """
        async with self._session() as session:
            session.add(
                Feedback(
                    session_id=session_id,
                    user_id=user_id,
                    score=score,
                    comment=comment,
                    message_id=message_id,
                )
            )
            await session.commit()

    async def for_session(self, session_id: str) -> list[Feedback]:
        """Return a session's verdicts, oldest first.

        Args:
            session_id: The conversation to read.

        Returns:
            The Feedback rows in insertion order.
        """
        async with self._session() as session:
            result = await session.exec(
                select(Feedback)
                .where(Feedback.session_id == session_id)
                .order_by(Feedback.id)
            )
            return list(result.all())
