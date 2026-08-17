"""Splitting a document into overlapping chunks.

A whole document is too large to embed as one vector and too coarse to
retrieve usefully, so it is cut into pieces. The separators are tried in
order, so a cut lands at a paragraph where one is available and falls back to
a line, a sentence, a word, and only then to an arbitrary character. Chunks
overlap so a sentence spanning a cut stays readable on both sides of it.

Chunks are returned as plain dictionaries rather than as the splitter's own
type, so the stages downstream depend on no particular library.
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tried in order: paragraph, line, sentence, word, character.
SEPARATORS = ["\n\n", "\n", ".", " ", ""]


def split_text(
    text: str, chunk_size: int = 512, overlap: int = 50
) -> list[dict[str, Any]]:
    """Cut text into overlapping chunks.

    Args:
        text: The document's full text.
        chunk_size: Target size of each chunk, in characters.
        overlap: Characters repeated from the end of one chunk at the start of
            the next, so context is not lost at a cut.

    Returns:
        One dictionary per chunk, each carrying the chunk's ``text`` and
        ``metadata`` holding its ``chunk_index``, counted from zero in reading
        order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
    )

    documents = splitter.create_documents([text])

    return [
        {"text": document.page_content, "metadata": {"chunk_index": index}}
        for index, document in enumerate(documents)
    ]
