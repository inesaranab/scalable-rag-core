"""The chat contracts: strict types at the API boundary."""

import pytest
from pydantic import ValidationError

from libs.schemas.chat import ChatRequest, ChatResponse, Message, RetrievalResult


def test_a_message_knows_its_role_and_stamps_its_time():
    message = Message(role="user", content="hola")

    assert message.role == "user"
    assert message.timestamp.tzinfo is not None  # aware, never naive


def test_an_invented_role_is_rejected():
    with pytest.raises(ValidationError):
        Message(role="hacker", content="hola")


def test_chat_request_defaults():
    request = ChatRequest(message="what stores vectors?")

    assert request.session_id is None
    assert request.stream is True
    assert request.filters is None


def test_citation_lists_are_not_shared_between_responses():
    """A mutable default would make two responses share one list."""
    first = ChatResponse(answer="a", session_id="s1", latency_ms=1.0)
    first.citations.append({"source": "doc.pdf", "text": "..."})
    second = ChatResponse(answer="b", session_id="s2", latency_ms=2.0)

    assert second.citations == []


def test_retrieval_result_carries_provenance():
    result = RetrievalResult(
        content="chunk text",
        source="doc.pdf",
        score=0.87,
        metadata={"page": 3},
    )

    assert result.source == "doc.pdf"
    assert 0 <= result.score <= 1
