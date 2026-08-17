"""Chunk metadata: stable hashes for deduplication, timestamps for freshness."""

from datetime import UTC, datetime

from pipelines.ingestion.chunking.metadata import enrich_metadata


def test_base_metadata_is_preserved():
    """Existing keys survive, so a chunk keeps its source and position."""
    enriched = enrich_metadata({"filename": "spec.pdf", "chunk_index": 3}, "text")

    assert enriched["filename"] == "spec.pdf"
    assert enriched["chunk_index"] == 3


def test_the_base_dictionary_is_not_modified():
    """A new dictionary is returned, so a caller reusing the base is safe."""
    base = {"filename": "spec.pdf"}

    enrich_metadata(base, "text")

    assert base == {"filename": "spec.pdf"}


def test_identical_content_yields_the_same_hash():
    """Repeated passages produce one hash, so they can be embedded once."""
    first = enrich_metadata({}, "shared boilerplate")
    second = enrich_metadata({}, "shared boilerplate")

    assert first["chunk_hash"] == second["chunk_hash"]


def test_the_hash_ignores_surrounding_metadata():
    """The hash covers the text alone, so the same passage from two documents
    is recognised as the same passage."""
    from_one = enrich_metadata({"filename": "a.pdf"}, "shared boilerplate")
    from_two = enrich_metadata({"filename": "b.pdf"}, "shared boilerplate")

    assert from_one["chunk_hash"] == from_two["chunk_hash"]


def test_different_content_yields_different_hashes():
    """Distinct passages must not be collapsed into one another."""
    first = enrich_metadata({}, "one passage")
    second = enrich_metadata({}, "one passage ")

    assert first["chunk_hash"] != second["chunk_hash"]


def test_the_hash_is_a_sha256_digest():
    """The digest is 64 hexadecimal characters, and nothing else."""
    chunk_hash = enrich_metadata({}, "text")["chunk_hash"]

    assert len(chunk_hash) == 64
    int(chunk_hash, 16)


def test_empty_content_still_hashes():
    """An empty chunk is identifiable rather than an error."""
    assert len(enrich_metadata({}, "")["chunk_hash"]) == 64


def test_the_timestamp_carries_a_timezone():
    """A naive timestamp compares wrongly against an aware one, so the offset
    is recorded."""
    ingested_at = datetime.fromisoformat(enrich_metadata({}, "text")["ingested_at"])

    assert ingested_at.tzinfo is not None


def test_the_timestamp_is_in_utc_and_current():
    """Ingestion time is recorded in one timezone everywhere, at the moment it
    happened."""
    before = datetime.now(UTC)
    ingested_at = datetime.fromisoformat(enrich_metadata({}, "text")["ingested_at"])
    after = datetime.now(UTC)

    assert before <= ingested_at <= after


def test_enrichment_adds_only_the_two_expected_keys():
    """Nothing unexpected reaches the vector store's payload."""
    enriched = enrich_metadata({"filename": "spec.pdf"}, "text")

    assert set(enriched) == {"filename", "chunk_hash", "ingested_at"}
