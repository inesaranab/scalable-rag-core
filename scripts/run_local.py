"""End-to-end proof: real files through real Ray into real Qdrant.

The embedding service does not exist yet, so a deterministic stand-in embeds
by hashing. Everything else is the production path.
"""
import hashlib

import ray
from qdrant_client import QdrantClient
from qdrant_client.http import models

from pipelines.ingestion.indexing.sinks import QdrantSink
from pipelines.ingestion.processing import process_batch

DIM = 64

def fake_embed(batch):
    """Deterministic stand-in for the embedding service."""
    vectors = []
    for text in batch["text"]:
        h = hashlib.sha256(text.encode()).digest()
        vectors.append([b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - min(DIM, len(h))))
    batch["vector"] = vectors
    return batch

# The collection must exist before anything upserts into it.
client = QdrantClient(host="localhost", port=6333)
client.recreate_collection(
    collection_name="rag_collection",
    vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
)

ray.init(num_cpus=4, include_dashboard=False, logging_level="ERROR")

docs = ray.data.read_binary_files("local://" + "/Users/inesarana/scalable-rag-core/data/text", include_paths=True).limit(20)
chunks = docs.map_batches(process_batch, batch_size=5).materialize()
vectors = chunks.map_batches(fake_embed, batch_size=100)

import pipelines.ingestion.indexing.sinks as sinks_mod
sink = QdrantSink()
# Point the worker-side indexer at localhost rather than cluster DNS.
import pipelines.ingestion.indexing.qdrant as qmod
qmod.DEFAULT_HOST = "localhost"

vectors.write_datasink(sink)

count = client.count("rag_collection").count
print(f"\nPOINTS IN QDRANT: {count}")
hits = client.query_points("rag_collection", query=fake_embed({"text": ["memory allocation"]})["vector"][0], limit=2)
for hit in hits.points:
    print(f"  score {hit.score:.3f}  {hit.payload['filename'][:50]}  chunk {hit.payload.get('chunk_index')}")
