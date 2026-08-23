"""The semantic cache: reuse an answer when a question MEANS the same.

Exact caching misses every paraphrase; this store embeds the incoming
question and searches previously answered ones by vector similarity, so
"what is Kubernetes?" and "explain k8s" can share one answer. A cache
must never break a request, so every failure here degrades to a miss.
"""

import logging
import uuid

from qdrant_client.http import models

logger = logging.getLogger(__name__)


class SemanticCacheStore:
    """Question-answer pairs, matched by meaning above a threshold.

    Attributes:
        embedder: Turns the question into a vector.
        qdrant: The connected Qdrant SDK client.
        collection: The collection holding cached pairs, separate from
            the document chunks.
        threshold: Minimum similarity score for a hit; below it, the
            question is treated as new.
    """

    def __init__(self, embedder, qdrant, collection: str, threshold: float) -> None:
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection = collection
        self.threshold = threshold

    async def ensure_collection(self, dim: int) -> None:
        """Create the cache collection if absent. Safe at every boot.

        Args:
            dim: Dimensionality of the embedding vectors.
        """
        if not await self.qdrant.collection_exists(self.collection):
            await self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )

    async def get(self, query: str) -> str | None:
        """Return a stored answer whose question means the same, or None.

        Args:
            query: The incoming question.

        Returns:
            The cached answer on a hit; None on a miss or any failure.
        """
        try:
            [vector] = await self.embedder.embed([query])
            response = await self.qdrant.query_points(
                collection_name=self.collection,
                query=vector,
                limit=1,
                score_threshold=self.threshold,
                with_payload=True,
            )
            if response.points:
                logger.info(
                    "semantic cache hit",
                    extra={"score": response.points[0].score},
                )
                return response.points[0].payload["answer"]
        except Exception:
            logger.warning("semantic cache lookup failed", exc_info=True)
        return None

    async def set(self, query: str, answer: str) -> None:
        """Store a question-answer pair for future similar questions.

        Args:
            query: The question as asked.
            answer: The answer the models produced for it.
        """
        try:
            [vector] = await self.embedder.embed([query])
            await self.qdrant.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"query": query, "answer": answer},
                    )
                ],
            )
        except Exception:
            logger.warning("semantic cache write failed", exc_info=True)
