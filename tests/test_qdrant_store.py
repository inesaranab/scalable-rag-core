"""The VectorStore port over a real Qdrant, exercised in-memory."""

import pytest
from qdrant_client import AsyncQdrantClient

from services.api.app.stores.qdrant_store import QdrantVectorStore


@pytest.fixture
async def store():
    """A store over Qdrant's in-memory mode: real engine, no server."""
    client = AsyncQdrantClient(":memory:")
    store = QdrantVectorStore(client, collection="test_chunks")
    await store.ensure_collection(dim=3)
    return store


async def test_added_chunks_come_back_nearest_first(store):
    """Search returns the chunk whose vector is closest to the query."""
    await store.add(
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
        chunks=["east", "north", "east-ish"],
    )

    found = await store.search([1.0, 0.0, 0.0], k=2)

    assert found == ["east", "east-ish"]


async def test_k_limits_how_many_chunks_return(store):
    await store.add(
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        chunks=["a", "b", "c"],
    )

    found = await store.search([1.0, 0.0, 0.0], k=1)

    assert len(found) == 1


async def test_ensure_collection_twice_is_harmless(store):
    """Booting the app twice must not wipe or duplicate the collection."""
    await store.add(vectors=[[1.0, 0.0, 0.0]], chunks=["kept"])

    await store.ensure_collection(dim=3)

    assert await store.search([1.0, 0.0, 0.0], k=1) == ["kept"]
