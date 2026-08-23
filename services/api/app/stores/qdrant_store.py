"""The VectorStore port implemented over Qdrant."""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantVectorStore:
    """Chunks and their vectors in a Qdrant collection.

    Attributes:
        client: The async Qdrant client (server URL, or ":memory:" in tests).
        collection: Name of the collection holding the chunks.
    """

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    async def ensure_collection(self, dim: int) -> None:
        """Create the collection if it does not exist. Safe to call on boot.

        Args:
            dim: Dimensionality of the vectors the collection will hold.
        """
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def add(self, vectors: list[list[float]], chunks: list[str]) -> None:
        """Store chunks with their vectors.

        Args:
            vectors: One vector per chunk, same order.
            chunks: The chunk texts, kept as payload for retrieval.
        """
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk},
            )
            for vector, chunk in zip(vectors, chunks)
        ]
        await self.client.upsert(collection_name=self.collection, points=points)

    async def search(self, vector: list[float], k: int) -> list[str]:
        """Return the k chunks nearest to the query vector, nearest first.

        Args:
            vector: The query vector.
            k: How many chunks to return.

        Returns:
            The chunk texts, most similar first.
        """
        result = await self.client.query_points(
            collection_name=self.collection, query=vector, limit=k
        )
        return [point.payload["text"] for point in result.points]
