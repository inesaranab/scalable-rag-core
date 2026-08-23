"""The embedding service answers the contract BatchEmbedder sends."""

import pytest

from services.api.app.models import embedding_engine


class FakeRequest:
    """A request carrying a fixed JSON body.

    Attributes:
        body: What ``json()`` resolves to.
    """

    def __init__(self, body: dict) -> None:
        self.body = body

    async def json(self) -> dict:
        return self.body


class FakeModel:
    """An encoder returning one fixed-size vector per input string."""

    def encode(self, texts, normalize_embeddings=False, batch_size=32):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def service(monkeypatch):
    """The deployment's underlying class, with the model swapped for a stub.

    Args:
        monkeypatch: Pytest's attribute-patching fixture.

    Returns:
        A service instance that loads no weights and touches no network.
    """
    monkeypatch.setattr(
        embedding_engine, "SentenceTransformer", lambda *a, **k: FakeModel()
    )
    return embedding_engine.EmbeddingDeployment.func_or_class()


async def test_each_text_comes_back_as_one_vector(service):
    """The response carries embeddings, one per input, in input order."""
    request = FakeRequest({"text": ["first chunk", "second chunk"]})

    response = await service(request)

    assert len(response["embeddings"]) == 2
    assert response["embeddings"][0] == [0.1, 0.2, 0.3]


async def test_an_empty_batch_embeds_to_an_empty_list(service):
    """No text in, no vectors out, no error."""
    response = await service(FakeRequest({"text": []}))

    assert response["embeddings"] == []


async def test_encoding_does_not_block_the_event_loop(service, monkeypatch):
    """Another coroutine gets turns while the model is busy encoding.

    encode is synchronous compute: called directly it holds the event loop's
    only thread until done, so nothing else in the replica runs. Off-loaded
    to a worker thread, the loop keeps serving between start and finish.
    """
    import asyncio
    import time

    def slow_encode(texts, normalize_embeddings=False, batch_size=32):
        time.sleep(0.2)
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(service.model, "encode", slow_encode)
    turns = 0

    async def count_turns():
        nonlocal turns
        while True:
            turns += 1
            await asyncio.sleep(0.01)

    counter = asyncio.get_event_loop().create_task(count_turns())
    await service(FakeRequest({"text": ["one chunk"]}))
    counter.cancel()

    assert turns >= 5


async def test_concurrent_requests_share_one_model_pass(service, monkeypatch):
    """Three simultaneous requests are answered from a single encode call.

    A GPU encodes 50 texts in one pass for nearly the cost of one, so
    concurrent requests are collected and encoded together. Each caller
    still receives exactly the vectors for its own texts, in its order.
    """
    import asyncio

    calls: list[list[str]] = []

    def counting_encode(texts, normalize_embeddings=False, batch_size=32):
        calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]

    monkeypatch.setattr(service.model, "encode", counting_encode)

    responses = await asyncio.gather(
        service(FakeRequest({"text": ["aa", "bbb"]})),
        service(FakeRequest({"text": ["cccc"]})),
        service(FakeRequest({"text": ["d", "ee"]})),
    )

    assert len(calls) == 1, f"expected one shared pass, got {len(calls)}"
    assert responses[0]["embeddings"] == [[2.0, 0.0], [3.0, 0.0]]
    assert responses[1]["embeddings"] == [[4.0, 0.0]]
    assert responses[2]["embeddings"] == [[1.0, 0.0], [2.0, 0.0]]


def test_the_shipped_bge_config_carries_the_engine_settings():
    """The yaml in models/embeddings/ validates and names the model."""
    from services.api.app.models.model_config import (
        EmbeddingModelConfig,
        load_model_config,
    )

    config = load_model_config("models/embeddings/bge-m3.yaml", EmbeddingModelConfig)

    assert config.model_id == "BAAI/bge-m3"
    assert config.normalize_embeddings is True
