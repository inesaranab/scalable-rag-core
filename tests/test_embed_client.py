"""The API-side client for the embedding service: pooled, retrying, async."""

import httpx
import pytest

from libs.retry import backoff
from services.api.app.clients.ray_embed import RayEmbedClient


@pytest.fixture
def no_waits(monkeypatch):
    """Retries are asserted, not endured: replace the retry sleep."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(backoff.asyncio, "sleep", fake_sleep)
    return waits


def client_backed_by(handler) -> RayEmbedClient:
    """Build a started client whose transport is the given handler.

    Args:
        handler: Function from request to httpx.Response, standing in for
            the embedding service.

    Returns:
        A client wired to the stubbed transport, making no network call.
    """
    client = RayEmbedClient(endpoint="http://embed.test/embed")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_texts_are_posted_and_embeddings_returned():
    """The service's contract: text list in, embeddings list out."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = client_backed_by(handler)

    vectors = await client.embed(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert seen == [{"text": ["first", "second"], "task_type": "query"}]


async def test_document_embedding_declares_its_task_type():
    """Ingestion marks its texts as documents, search as queries."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.5, 0.6]]})

    client = client_backed_by(handler)

    await client.embed(["a chunk"], task_type="document")

    assert seen[0]["task_type"] == "document"


async def test_one_connection_pool_serves_many_calls():
    """The transport is created once and reused, not opened per call."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    client = client_backed_by(handler)
    pool_before = client.client

    await client.embed(["one"])
    await client.embed(["two"])

    assert client.client is pool_before


async def test_a_transient_failure_is_retried(no_waits):
    """One 500 does not fail the call; the retry succeeds."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"embeddings": [[0.9]]})

    client = client_backed_by(handler)

    vectors = await client.embed(["retry me"])

    assert vectors == [[0.9]]
    assert attempts == 2
    assert len(no_waits) == 1


async def test_calling_before_start_is_a_clear_error():
    """A never-started client fails loudly, not with a cryptic AttributeError."""
    client = RayEmbedClient(endpoint="http://embed.test/embed")

    with pytest.raises(RuntimeError):
        await client.embed(["too early"])
