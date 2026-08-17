"""Turning raw files into chunks ready to embed.

This is the parse stage: pick a loader by file extension, extract text, split
it, and attach each chunk's hash and ingestion time. It runs on CPU only, so
it scales independently of the stages that follow.

It also transposes. A batch of files arrives column-oriented — one dictionary
whose values are lists — and chunking produces one dictionary per chunk, so
the results are gathered back into columns before being returned.

A file that cannot be parsed contributes no chunks rather than ending the run.
"""

import logging
from pathlib import Path
from typing import Any

from pipelines.ingestion.chunking.metadata import enrich_metadata
from pipelines.ingestion.chunking.splitter import split_text
from pipelines.ingestion.loaders.docx import parse_docx_bytes
from pipelines.ingestion.loaders.html import parse_html_bytes

logger = logging.getLogger(__name__)

LOADERS = {
    ".docx": parse_docx_bytes,
    ".html": parse_html_bytes,
    ".htm": parse_html_bytes,
}


def parse_file(file_bytes: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    """Extract text from a file, choosing the parser by extension.

    Args:
        file_bytes: The file's raw bytes.
        filename: Name the file arrived under, whose extension selects the
            parser.

    Returns:
        The extracted text, and metadata carrying the filename and type.

    Raises:
        ValueError: If no parser handles the extension.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        # Deferred: this import loads the whole recognition stack, which every
        # test run and every text-only worker would otherwise pay for.
        from pipelines.ingestion.loaders.pdf import parse_pdf_bytes

        return parse_pdf_bytes(file_bytes, filename)
    if suffix in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="replace"), {
            "filename": filename,
            "type": "text",
        }
    if suffix in LOADERS:
        return LOADERS[suffix](file_bytes, filename)

    raise ValueError(f"no loader for {suffix!r} ({filename})")


def process_batch(
    batch: dict[str, Any], chunk_size: int = 512, overlap: int = 50
) -> dict[str, Any]:
    """Parse, split and enrich a batch of files.

    Args:
        batch: A batch carrying ``bytes``, holding each file's contents, and
            ``path``, holding where each came from.
        chunk_size: Target size of each chunk, in characters.
        overlap: Characters repeated between neighbouring chunks.

    Returns:
        A batch carrying ``text``, one string per chunk, and ``metadata``,
        one dictionary per chunk holding its filename, position, content hash
        and ingestion time. Files that could not be parsed contribute nothing.
    """
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for file_bytes, path in zip(batch["bytes"], batch["path"]):
        filename = Path(path).name
        try:
            text, file_metadata = parse_file(file_bytes, filename)
        except Exception as exc:
            logger.warning("skipping %s: %s", filename, type(exc).__name__)
            continue

        for chunk in split_text(text, chunk_size=chunk_size, overlap=overlap):
            if not chunk["text"].strip():
                continue
            texts.append(chunk["text"])
            metadatas.append(
                enrich_metadata(
                    {**file_metadata, **chunk["metadata"]}, chunk["text"]
                )
            )

    return {"text": texts, "metadata": metadatas}
