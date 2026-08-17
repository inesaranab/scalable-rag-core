"""Identifier generation: random where it must differ, stable where it must not."""

import uuid

from libs.utils.ids import generate_file_id, generate_session_id, generate_trace_id


def test_session_ids_differ_between_calls():
    """Two sessions must never collide, so nothing about them is derived."""
    assert generate_session_id() != generate_session_id()


def test_session_id_is_a_valid_uuid():
    """The identifier parses as a UUID, so anything expecting one accepts it."""
    uuid.UUID(generate_session_id())


def test_identical_content_yields_the_same_file_id():
    """The same bytes always produce the same identifier.

    This is what makes a repeated upload detectable before it is chunked and
    embedded a second time.
    """
    assert generate_file_id(b"a report") == generate_file_id(b"a report")


def test_different_content_yields_different_file_ids():
    """Distinct documents must not be mistaken for one another."""
    assert generate_file_id(b"a report") != generate_file_id(b"a report ")


def test_file_id_is_a_sha256_digest():
    """The identifier is 64 hexadecimal characters, and nothing else."""
    file_id = generate_file_id(b"anything")

    assert len(file_id) == 64
    int(file_id, 16)


def test_empty_content_still_yields_an_id():
    """An empty file is a file, and must be identifiable rather than an error."""
    assert len(generate_file_id(b"")) == 64


def test_trace_id_has_no_hyphens():
    """OpenTelemetry expects 32 hexadecimal characters without separators."""
    trace_id = generate_trace_id()

    assert len(trace_id) == 32
    assert "-" not in trace_id


def test_trace_ids_differ_between_calls():
    """Each traced operation must be distinguishable from every other."""
    assert generate_trace_id() != generate_trace_id()
