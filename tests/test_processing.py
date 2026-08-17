"""Parsing a batch of files into chunks ready to embed."""

import io

import docx
import pytest

from pipelines.ingestion.processing import parse_file, process_batch


def build_docx(*paragraphs: str) -> bytes:
    """Produce a real Word document in memory.

    Args:
        *paragraphs: Text of each paragraph, in order.

    Returns:
        The document's bytes.
    """
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_the_loader_is_chosen_by_extension():
    """A file is parsed according to what it is, not what it claims."""
    text, metadata = parse_file(b"<p>hello</p>", "page.html")

    assert "hello" in text
    assert metadata["type"] == "html"


def test_plain_text_needs_no_loader():
    """Text files are decoded rather than parsed."""
    text, metadata = parse_file(b"just words", "notes.txt")

    assert text == "just words"
    assert metadata["type"] == "text"


def test_the_extension_is_matched_case_insensitively():
    """An uppercase extension names the same format."""
    _, metadata = parse_file(b"<p>hello</p>", "PAGE.HTML")

    assert metadata["type"] == "html"


def test_an_unknown_extension_is_rejected():
    """A format with no loader fails where it is discovered."""
    with pytest.raises(ValueError, match="no loader"):
        parse_file(b"data", "archive.zip")


def test_a_batch_becomes_chunks_with_metadata():
    """Files in, chunks out, each carrying where it came from."""
    batch = {"bytes": [b"first document text"], "path": ["/blobs/one.txt"]}

    result = process_batch(batch)

    assert result["text"] == ["first document text"]
    assert result["metadata"][0]["filename"] == "one.txt"


def test_output_is_column_oriented():
    """The shape matches what the embedding stage expects."""
    batch = {"bytes": [b"one", b"two"], "path": ["/a.txt", "/b.txt"]}

    result = process_batch(batch)

    assert set(result) == {"text", "metadata"}
    assert len(result["text"]) == len(result["metadata"])


def test_every_chunk_is_enriched():
    """A hash and an ingestion time are attached before embedding."""
    batch = {"bytes": [b"some text"], "path": ["/one.txt"]}

    metadata = process_batch(batch)["metadata"][0]

    assert "chunk_hash" in metadata
    assert "ingested_at" in metadata
    assert "chunk_index" in metadata


def test_a_long_document_yields_several_chunks():
    """Splitting happens inside the stage, not after it."""
    batch = {"bytes": [b"word " * 500], "path": ["/long.txt"]}

    result = process_batch(batch, chunk_size=200, overlap=20)

    assert len(result["text"]) > 1


def test_several_files_are_flattened_into_one_batch():
    """Chunks from every file arrive together, not nested per file."""
    batch = {"bytes": [b"one", b"two", b"three"], "path": ["/a.txt", "/b.txt", "/c.txt"]}

    result = process_batch(batch)

    assert result["text"] == ["one", "two", "three"]


def test_an_unparseable_file_is_skipped():
    """One bad file must not discard the rest of the batch."""
    batch = {
        "bytes": [b"good text", b"data"],
        "path": ["/fine.txt", "/broken.zip"],
    }

    result = process_batch(batch)

    assert result["text"] == ["good text"]


def test_a_file_yielding_no_text_contributes_nothing():
    """An empty document produces no chunk rather than an empty one."""
    batch = {"bytes": [build_docx()], "path": ["/empty.docx"]}

    result = process_batch(batch)

    assert result["text"] == []


def test_an_empty_batch_returns_empty_columns():
    """No files means no chunks, in the expected shape."""
    result = process_batch({"bytes": [], "path": []})

    assert result == {"text": [], "metadata": []}
