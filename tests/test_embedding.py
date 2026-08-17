"""Embedding: batches reach the service and come back carrying vectors."""

import httpx
import pytest

from pipelines.ingestion.embedding.compute import BatchEmbedder


def embedder_returning(vectors, status: int = 200) -> BatchEmbedder:
    """Build an embedder whose service responds with fixed vectors.

    Args:
        vectors: What the service should return under ``embeddings``.
        status: HTTP status the service should respond with.

    Returns:
        An embedder wired to a stubbed transport, making no network call.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        handler.last_request = request
        return httpx.Response(status, json={"embeddings": vectors})

    handler.last_request = None
    embedder = BatchEmbedder(endpoint="http://embed.test/embed")
    embedder.client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder._handler = handler
    return embedder


def test_vectors_are_added_to_the_batch():
    """The batch comes back carrying one vector per chunk."""
    embedder = embedder_returning([[0.1, 0.2], [0.3, 0.4]])

    result = embedder({"text": ["first chunk", "second chunk"]})

    assert result["vector"] == [[0.1, 0.2], [0.3, 0.4]]


def test_the_original_batch_keys_survive():
    """Metadata travelling alongside the text is not lost."""
    embedder = embedder_returning([[0.1]])

    result = embedder({"text": ["chunk"], "metadata": [{"chunk_index": 0}]})

    assert result["metadata"] == [{"chunk_index": 0}]


def test_the_whole_batch_is_sent_in_one_request():
    """Chunks travel together, which is what keeps the GPU busy."""
    embedder = embedder_returning([[0.1], [0.2], [0.3]])

    embedder({"text": ["a", "b", "c"]})

    import json

    sent = json.loads(embedder._handler.last_request.content)
    assert sent["text"] == ["a", "b", "c"]


def test_the_task_type_marks_these_as_documents():
    """Some models embed a stored document differently from a query, so the
    role is stated rather than assumed."""
    embedder = embedder_returning([[0.1]])

    embedder({"text": ["chunk"]})

    import json

    sent = json.loads(embedder._handler.last_request.content)
    assert sent["task_type"] == "document"


def test_an_error_status_raises():
    """A failed batch is not silently returned without vectors."""
    embedder = embedder_returning([], status=503)

    with pytest.raises(httpx.HTTPStatusError):
        embedder({"text": ["chunk"]})


def test_an_empty_batch_is_sent_and_returns_no_vectors():
    """A batch with nothing in it is handled rather than special-cased."""
    embedder = embedder_returning([])

    result = embedder({"text": []})

    assert result["vector"] == []


def test_the_endpoint_is_configurable():
    """The address differs between a laptop and a cluster, so it is not fixed
    in the code."""
    embedder = BatchEmbedder(endpoint="http://elsewhere:9000/embed")

    assert embedder.endpoint == "http://elsewhere:9000/embed"
