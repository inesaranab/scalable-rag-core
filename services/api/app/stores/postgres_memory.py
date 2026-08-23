"""Conversation memory: every turn of every session, in one table.

What the model recalls from five turns ago is whatever this store gives
back; the LLM itself remembers nothing between requests.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from services.api.app.db_models import ChatHistory


class ChatMemoryStore:
    """Reads and writes ChatHistory rows.

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
        # every class you declared with table=True registered itself
        # at import time, into a shared catalogue
        # SQLModel.metadata.create_all walks that catalogue
        # and emits
        # CREATE TABLE for each table that doesn't already exist
        async with self.engine.begin() as conn:  # opens a transcation
            await conn.run_sync(
                SQLModel.metadata.create_all
            )  # bridges the create_all (sync)

    async def add_message(
        self, session_id: str, role: str, content: str, user_id: str
    ) -> None:
        """Append one turn to a conversation.

        Args:
            session_id: The conversation the turn belongs to.
            role: Who spoke: "user" or "assistant".
            content: The text of the turn.
            user_id: Who owns the conversation.
        """
        async with self._session() as session:
            session.add(
                ChatHistory(
                    session_id=session_id,
                    role=role,
                    content=content,
                    user_id=user_id,
                )
            )
            await session.commit()

    async def get_history(self, session_id: str, limit: int = 10) -> list[ChatHistory]:
        """Return a session's most recent turns, oldest first.

        Args:
            session_id: The conversation to read.
            limit: At most this many turns, counted from the newest.

        Returns:
            The turns in chronological order, ready to prepend to a
            prompt.
        """
        async with self._session() as session:
            result = await session.exec(
                select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                # id breaks ties between turns stamped in the same instant.
                .order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc())
                .limit(limit)
            )
            newest_first = list(result.all())
        return newest_first[::-1]
