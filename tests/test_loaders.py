"""Loaders: bytes of a given format become text plus metadata."""

import io

import docx
import pytest

from pipelines.ingestion.loaders.docx import parse_docx_bytes
from pipelines.ingestion.loaders.html import parse_html_bytes


def build_docx(*paragraphs: str) -> bytes:
    """Produce a real Word document in memory.

    Args:
        *paragraphs: Text of each paragraph, in order. Empty strings become
            empty paragraphs.

    Returns:
        The document's bytes, as an upload would deliver them.
    """
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_text_is_extracted():
    """Paragraph text survives the round trip through the parser."""
    text, _ = parse_docx_bytes(build_docx("Rated at 150 watts."), "spec.docx")

    assert "Rated at 150 watts." in text


def test_docx_empty_paragraphs_are_dropped():
    """Blank paragraphs carry no meaning and would become empty chunks."""
    text, _ = parse_docx_bytes(build_docx("first", "", "   ", "second"), "spec.docx")

    assert text == "first\n\nsecond"


def test_docx_metadata_records_filename_and_type():
    """Downstream stages route on the type and attribute chunks by filename."""
    _, metadata = parse_docx_bytes(build_docx("text"), "spec.docx")

    assert metadata == {"filename": "spec.docx", "type": "docx"}


def test_docx_with_no_paragraphs_yields_empty_text():
    """An empty document is a valid document, not an error."""
    text, _ = parse_docx_bytes(build_docx(), "empty.docx")

    assert text == ""


def test_docx_rejects_bytes_that_are_not_a_document():
    """Corrupt input fails at the loader rather than producing nonsense text."""
    with pytest.raises(Exception):
        parse_docx_bytes(b"not a docx at all", "broken.docx")


def test_html_scripts_and_styles_are_removed():
    """Code and styling would otherwise be embedded and compete with content."""
    html = (
        b"<html><head><style>body{color:red}</style></head>"
        b"<body><script>alert(1)</script><p>Rated at 150 watts.</p></body></html>"
    )

    text, _ = parse_html_bytes(html, "spec.html")

    assert "Rated at 150 watts." in text
    assert "alert" not in text
    assert "color:red" not in text


def test_html_meta_tags_are_removed():
    """Document metadata is not readable content."""
    html = b'<html><head><meta name="author" content="nobody"></head><body>real</body></html>'

    text, _ = parse_html_bytes(html, "page.html")

    assert "nobody" not in text
    assert "real" in text


def test_html_metadata_records_filename_and_type():
    """Downstream stages route on the type and attribute chunks by filename."""
    _, metadata = parse_html_bytes(b"<p>x</p>", "page.html")

    assert metadata == {"filename": "page.html", "type": "html"}


def test_html_without_markup_is_returned_as_text():
    """Plain text is valid HTML and passes through unchanged."""
    text, _ = parse_html_bytes(b"just words", "page.html")

    assert "just words" in text
