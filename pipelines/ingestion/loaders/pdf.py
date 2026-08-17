"""Extracting text from PDF bytes.

A file arrives as bytes because ingestion reads from object storage rather
than from a path. It is written to a temporary file on disk before parsing, so
one oversized document cannot exhaust a worker's memory.
"""

import tempfile
from typing import Any

from unstructured.partition.pdf import partition_pdf


def parse_pdf_bytes(file_bytes: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    """Extract text from a PDF held in memory.

    The hi_res strategy runs layout analysis and optical character
    recognition, so pages carrying only scanned images are read as well as
    those with an embedded text layer.

    Args:
        file_bytes: The PDF's raw bytes.
        filename: Name recorded in the metadata. Not used to locate the file.

    Returns:
        The extracted text, and metadata carrying the filename and the type
        ``pdf``.
    """
    text_content = ""

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file.flush()

        # partition_pdf uses an object detector model the same family as YOLO
        elements = partition_pdf(filename=tmp_file.name, strategy="hi_res")

        for element in elements:
            text_content += str(element) + "\n"

    return text_content, {"filename": filename, "type": "pdf"}
