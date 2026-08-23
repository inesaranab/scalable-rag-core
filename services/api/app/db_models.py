"""The database schema, as SQLModel classes: one class = one table."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """A user who may log in.

    Attributes:
        username: The login name; the table's primary key.
        password_hash: The bcrypt hash of the password. The plaintext is
            never stored anywhere.
    """

    __tablename__ = "users"

    username: str = Field(primary_key=True)
    password_hash: str


class ChatHistory(SQLModel, table=True):
    """One turn of one conversation, kept for context and audit.

    Attributes:
        id: Auto-incremented row id; also the tiebreaker for ordering
            turns created in the same instant.
        session_id: Which conversation the turn belongs to; indexed,
            because history is always fetched by session.
        user_id: Who owns the conversation; indexed for per-user audit.
        role: Who spoke: "user" or "assistant".
        content: The text of the turn.
        meta: Free-form extras (latency, token counts, model version).
        created_at: When the turn happened; timezone-aware UTC.
    """

    __tablename__ = "chat_history"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    user_id: str = Field(index=True)
    role: str
    content: str
    meta: dict = Field(
        default_factory=dict, sa_column=Column(JSON)
    )  # store this field as a JSON COLUMN
    # timezone=True: Postgres refuses an aware timestamp in a naive
    # column (SQLite silently accepts it, so only live runs catch this).
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True)),
    )
