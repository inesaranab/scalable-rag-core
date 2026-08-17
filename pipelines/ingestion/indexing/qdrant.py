"""Writing embedded chunks to the vector store.

Records are written in batches: one round trip for a thousand points rather
than a thousand round trips, which is where the throughput comes from.

Each point's identifier is derived from the chunk's content hash rather than
minted randomly, so re-running ingestion overwrites a chunk in place instead
of creating a second copy of it.
"""

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

# From the environment because the address is deployment configuration:
# cluster DNS deployed, localhost in development. Workers import this
# module fresh, so only the environment reaches them.
DEFAULT_HOST = os.environ.get("QDRANT_HOST", "qdrant-service")
DEFAULT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_collection")

# Qdrant identifiers must be a UUID or an unsigned integer, so a content hash
# is folded into a UUID rather than used directly.
_ID_NAMESPACE = uuid.NAMESPACE_URL


class QdrantIndexer:
    """Writes vectors and their payloads to a Qdrant collection.

    Attributes:
        client: Connection to the vector store, held open across batches.
        collection_name: Collection the points are written to.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        """Open a connection to the vector store.

        Args:
            host: Address of the store. Resolves to internal cluster DNS when
                deployed.
            port: Port the REST interface listens on.
            collection_name: Collection to write into.
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def write(self, batch: dict[str, Any]) -> int:
        """Write one batch of embedded chunks.

        Chunks carrying no vector are skipped, since a failed embedding must
        not be stored as a point without one.

        Args:
            batch: A batch carrying ``text``, ``vector`` and ``metadata``,
                each holding one entry per chunk.

        Returns:
            How many points were written.
        """
        points = [
            # structure qdrant uses to contain a record
            models.PointStruct(
                # uuid.uuid5(namespace, name)   # both arguments mandatory
                # with uuid4 you get two copies of the same chunk does not deduplicate
                # with uuid5 ingesting the same document twice gives you one copy because of the same hash
                # second write overwrites the first
                id=str(uuid.uuid5(_ID_NAMESPACE, metadata["chunk_hash"]))
                if "chunk_hash" in metadata
                else str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, **metadata},
            )
            for text, vector, metadata in zip(
                batch["text"], batch["vector"], batch["metadata"]
            )
            if vector is not None
        ]

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        return len(points)
