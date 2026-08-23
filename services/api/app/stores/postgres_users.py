"""The users table, accessed through SQLModel.

The class exposes the same port as before (get_password_hash, upsert_user);
only the machinery inside changed from raw SQL to the ORM — which is the
point of keeping stores behind their interfaces.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from services.api.app.db_models import User


class PostgresUserStore:
    """Reads and writes User rows.

    Attributes:
        engine: The async database engine (Postgres in production,
            SQLite in tests — same code path).
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._session = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,  # I'll trust my photo and re-SELECT when freshness matters
        )

    async def ensure_table(self) -> None:
        """Create the schema if absent. Safe to call at every boot."""
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def get_password_hash(self, username: str) -> str | None:
        """Return the stored bcrypt hash for a username, or None.

        Args:
            username: The username to look up.

        Returns:
            The bcrypt hash string, or None when the user does not exist.
        """
        async with self._session() as session:
            user = await session.get(User, username)
            return user.password_hash if user else None

    async def upsert_user(self, username: str, password_hash: str) -> None:
        """Create or update a user with an already-hashed password.

        Args:
            username: The username to create or update.
            password_hash: The bcrypt hash — never a plaintext password.
        """
        async with self._session() as session:
            await session.merge(User(username=username, password_hash=password_hash))
            await session.commit()
