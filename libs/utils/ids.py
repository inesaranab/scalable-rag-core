"""Identifiers that tie one user action to every record it produces.

A single request touches the gateway, the cache, the planner, both stores and
the model. Correlating those by identifier is what makes a failure traceable
after the fact, rather than a set of unrelated log lines.
"""

import hashlib
import uuid


def generate_session_id() -> str:
    """Mint an identifier for a chat session.

    Returns:
        A random UUID in its canonical hyphenated form.
    """
    return str(uuid.uuid4())


def generate_file_id(content: bytes) -> str:
    """Derive an identifier from a file's contents.

    Identical bytes yield an identical identifier, so re-uploading a document
    is detectable before it is chunked and embedded a second time.

    Args:
        content: The file's raw bytes.

    Returns:
        The SHA-256 digest of the contents, as hexadecimal.
    """
    return hashlib.sha256(content).hexdigest()


def generate_trace_id() -> str:
    """Mint an identifier for one traced operation.

    Returns:
        A random UUID as 32 hexadecimal characters without hyphens, the form
        OpenTelemetry expects.
    """
    return uuid.uuid4().hex
