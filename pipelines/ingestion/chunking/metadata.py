"""Enriching a chunk's metadata before it is embedded.

Two facts are added. A hash of the chunk's own text identifies duplicates:
passages repeated across documents — boilerplate, shared sections — produce
the same hash, so they can be embedded once rather than once per occurrence.
A timestamp records when the chunk entered the index, which is what makes
staleness measurable and reindexing selective.
"""

import hashlib
from datetime import UTC, datetime
from typing import Any


def enrich_metadata(base_metadata: dict[str, Any], content: str) -> dict[str, Any]:
    """Add a content hash and an ingestion timestamp to a chunk's metadata.

    Args:
        base_metadata: Metadata already carried by the chunk, such as its
            source filename and index. Left unchanged; a new dictionary is
            returned.
        content: The chunk's text, which the hash is taken over.

    Returns:
        The base metadata with ``chunk_hash``, the SHA-256 digest of the text
        as hexadecimal, and ``ingested_at``, the current time in UTC as an
        ISO 8601 string.
    """
    return {
        **base_metadata,
        "chunk_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "ingested_at": datetime.now(UTC).isoformat(),
    }
