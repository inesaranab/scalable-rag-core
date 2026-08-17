"""Indexing: batched writes, stable identifiers, and repeatable runs."""

from unittest.mock import MagicMock

from pipelines.ingestion.indexing.qdrant import QdrantIndexer


def indexer_with_fake_client() -> QdrantIndexer:
    """Build an indexer whose store records calls instead of receiving them.

    Returns:
        An indexer whose ``client`` is a recording double, connecting to
        nothing.
    """
    indexer = QdrantIndexer.__new__(QdrantIndexer)
    indexer.client = MagicMock()
    indexer.collection_name = "test_collection"
    return indexer


def batch(*chunks) -> dict:
    """Assemble a column-oriented batch from per-chunk tuples.

    Args:
        *chunks: One tuple per chunk, holding its text, vector and metadata.

    Returns:
        A batch in the shape the pipeline passes between stages.
    """
    return {
        "text": [text for text, _, _ in chunks],
        "vector": [vector for _, vector, _ in chunks],
        "metadata": [metadata for _, _, metadata in chunks],
    }


def test_the_whole_batch_is_written_in_one_call():
    """One round trip carries every point, which is where throughput comes
    from."""
    indexer = indexer_with_fake_client()

    written = indexer.write(
        batch(
            ("one", [0.1], {"chunk_hash": "a"}),
            ("two", [0.2], {"chunk_hash": "b"}),
            ("three", [0.3], {"chunk_hash": "c"}),
        )
    )

    assert written == 3
    assert indexer.client.upsert.call_count == 1


def test_the_text_is_stored_in_the_payload():
    """Retrieval must return readable content, not only a vector."""
    indexer = indexer_with_fake_client()

    indexer.write(batch(("the chunk text", [0.1], {"chunk_hash": "a"})))

    points = indexer.client.upsert.call_args.kwargs["points"]
    assert points[0].payload["text"] == "the chunk text"


def test_metadata_travels_into_the_payload():
    """Filename and position are needed to attribute an answer to a source."""
    indexer = indexer_with_fake_client()

    indexer.write(
        batch(("text", [0.1], {"chunk_hash": "a", "filename": "spec.pdf"}))
    )

    points = indexer.client.upsert.call_args.kwargs["points"]
    assert points[0].payload["filename"] == "spec.pdf"


def test_the_same_chunk_always_gets_the_same_identifier():
    """A re-run overwrites a chunk in place rather than storing it twice."""
    first = indexer_with_fake_client()
    second = indexer_with_fake_client()

    first.write(batch(("text", [0.1], {"chunk_hash": "same"})))
    second.write(batch(("text", [0.1], {"chunk_hash": "same"})))

    id_one = first.client.upsert.call_args.kwargs["points"][0].id
    id_two = second.client.upsert.call_args.kwargs["points"][0].id
    assert id_one == id_two


def test_different_chunks_get_different_identifiers():
    """Distinct chunks must not overwrite one another."""
    indexer = indexer_with_fake_client()

    indexer.write(
        batch(("one", [0.1], {"chunk_hash": "a"}), ("two", [0.2], {"chunk_hash": "b"}))
    )

    points = indexer.client.upsert.call_args.kwargs["points"]
    assert points[0].id != points[1].id


def test_chunks_without_a_vector_are_skipped():
    """A failed embedding must not be stored as a point without one."""
    indexer = indexer_with_fake_client()

    written = indexer.write(
        batch(("good", [0.1], {"chunk_hash": "a"}), ("failed", None, {"chunk_hash": "b"}))
    )

    assert written == 1
    points = indexer.client.upsert.call_args.kwargs["points"]
    assert points[0].payload["text"] == "good"


def test_an_empty_batch_writes_nothing():
    """No chunks means no call, rather than an empty upsert."""
    indexer = indexer_with_fake_client()

    written = indexer.write(batch())

    assert written == 0
    assert indexer.client.upsert.call_count == 0


def test_the_collection_name_is_used():
    """Points reach the collection the indexer was configured for."""
    indexer = indexer_with_fake_client()

    indexer.write(batch(("text", [0.1], {"chunk_hash": "a"})))

    assert (
        indexer.client.upsert.call_args.kwargs["collection_name"] == "test_collection"
    )
