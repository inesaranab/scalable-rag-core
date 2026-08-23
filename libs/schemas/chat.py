"""The chat contracts every service agrees on.

Strict types at the boundary: garbage is rejected by validation before it
reaches an agent, and the same models auto-generate the Swagger docs.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """One turn of a conversation.

    Attributes:
        role: Who spoke: the user, the assistant, or the system prompt.
        content: The text of the turn.
        timestamp: When the turn happened; timezone-aware, stamped on
            creation.
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatRequest(BaseModel):
    """The request body for a chat turn.

    Attributes:
        message: The user's message.
        session_id: Which conversation this belongs to; None starts a new
            one.
        stream: Whether the answer should arrive token by token.
        filters: Optional retrieval restrictions (e.g. only HR documents).
    """

    message: str = Field(min_length=1)
    session_id: str | None = None
    stream: bool = True
    filters: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """The response body for a non-streaming chat turn.

    Attributes:
        answer: The assistant's reply.
        session_id: The conversation the reply belongs to.
        citations: Where the answer came from, one source per entry —
            what lets a reader verify instead of trust.
        latency_ms: How long the turn took, wall-clock.
    """

    answer: str
    session_id: str
    citations: list[dict[str, str]] = Field(default_factory=list)
    latency_ms: float


class RetrievalResult(BaseModel):
    """One retrieved chunk, with its provenance.

    Attributes:
        content: The chunk's text.
        source: The document it came from.
        score: Similarity to the query, higher is closer.
        metadata: Anything else the store knows about the chunk.
    """

    content: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
