"""Extracting text from Word document bytes.

Word documents already carry their text structurally, so no recognition step
is needed and the bytes can be read straight from memory.
"""

import io
from typing import Any

import docx
from docx.document import Document


def parse_docx_bytes(file_bytes: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    """Extract text from a Word document held in memory.

    Empty paragraphs are dropped, and the remainder joined by blank lines so
    the paragraph boundaries survive into chunking.

    Args:
        file_bytes: The document's raw bytes.
        filename: Name recorded in the metadata.

    Returns:
        The extracted text, and metadata carrying the filename and the type
        ``docx``.
    """
    # BytesIO presents the bytes as an open file, which is what the parser
    # expects; the parsing itself happens in docx.Document.
    document: Document = docx.Document(io.BytesIO(file_bytes))

    full_text: list[str] = [
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    ]

    return "\n\n".join(full_text), {"filename": filename, "type": "docx"}
