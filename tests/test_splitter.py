"""Chunking: cuts land at the largest available boundary, and overlap holds."""

from pipelines.ingestion.chunking.splitter import split_text


def test_short_text_is_a_single_chunk():
    """Text below the target size is not cut at all."""
    chunks = split_text("A short sentence.", chunk_size=512)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "A short sentence."


def test_long_text_is_cut_into_several_chunks():
    """Text beyond the target size is divided."""
    chunks = split_text("word " * 500, chunk_size=200, overlap=20)

    assert len(chunks) > 1


def test_chunks_respect_the_target_size():
    """No chunk greatly exceeds the size it was asked for."""
    chunks = split_text("word " * 500, chunk_size=200, overlap=20)

    assert all(len(chunk["text"]) <= 200 for chunk in chunks)


def test_chunk_indices_count_from_zero_in_order():
    """The index records reading order, so a chunk can be placed back in its
    document."""
    chunks = split_text("word " * 500, chunk_size=200, overlap=20)

    assert [chunk["metadata"]["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


def test_every_chunk_carries_text_and_metadata():
    """The shape is fixed, so downstream stages need not check for absent keys."""
    chunks = split_text("word " * 500, chunk_size=200, overlap=20)

    for chunk in chunks:
        assert set(chunk) == {"text", "metadata"}
        assert isinstance(chunk["text"], str)
        assert "chunk_index" in chunk["metadata"]


def test_paragraph_boundaries_are_preferred_to_mid_sentence_cuts():
    """With a paragraph break available inside the window, the cut takes it
    rather than splitting a sentence."""
    text = "First paragraph here.\n\n" + "Second paragraph here."

    chunks = split_text(text, chunk_size=30, overlap=0)

    assert chunks[0]["text"] == "First paragraph here."


def test_overlap_repeats_content_between_neighbours():
    """Consecutive chunks share their boundary, so a sentence spanning a cut
    stays readable on both sides."""
    text = "".join(f"sentence number {i}. " for i in range(60))

    chunks = split_text(text, chunk_size=100, overlap=40)

    assert any(
        chunks[i]["text"][-20:] in chunks[i + 1]["text"]
        or chunks[i + 1]["text"][:20] in chunks[i]["text"]
        for i in range(len(chunks) - 1)
    )


def test_empty_text_yields_no_chunks():
    """Nothing to split produces nothing to embed, rather than an empty chunk."""
    assert split_text("") == []


def test_whitespace_only_text_yields_no_usable_chunks():
    """A document that extracted to whitespace contributes nothing."""
    chunks = split_text("   \n\n   \n  ")

    assert all(not chunk["text"].strip() for chunk in chunks)
