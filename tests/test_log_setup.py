"""Structured logging: every record becomes one machine-readable JSON line."""

import json
import logging

from services.api.app.log_setup import JSONFormatter


def format_record(**kwargs) -> dict:
    """Build a record, format it, parse it back.

    Args:
        kwargs: Overrides for the record's fields (e.g. ``exc_info``).

    Returns:
        The formatted line, parsed from JSON back into a dict.
    """
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname=__file__,
        lineno=42,
        msg="something %s happened",
        args=("odd",),
        exc_info=kwargs.get("exc_info"),
    )
    for key, value in kwargs.get("extra", {}).items():
        setattr(record, key, value)
    return json.loads(JSONFormatter().format(record))


def test_a_record_becomes_json_with_the_essential_fields():
    line = format_record()

    assert line["level"] == "WARNING"
    assert line["logger"] == "test.logger"
    assert line["message"] == "something odd happened"
    assert line["line"] == 42
    assert "T" in line["timestamp"]  # ISO format


def test_an_exception_travels_inside_the_line():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        line = format_record(exc_info=sys.exc_info())

    assert "ValueError: boom" in line["exception"]


def test_extra_fields_like_request_id_are_included():
    line = format_record(extra={"request_id": "req-123"})

    assert line["request_id"] == "req-123"
