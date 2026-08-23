"""Structured JSON logging: one JSON object per line, on stdout.

Machines index fields; humans grep prose. Production log collectors
(Azure Log Analytics, Datadog, Splunk) parse each line's fields directly,
so "all WARNINGs for request_id X" is a query instead of a regex.

Named log_setup (not logging) so it cannot shadow the stdlib module.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Fields carried by every LogRecord that we do NOT copy verbatim — anything
# outside this set was passed via `extra=` and gets included in the line.
_STANDARD_ATTRS = frozenset(
    vars(
        logging.LogRecord("", 0, "", 0, "", (), None)
    ).keys()
) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    """Formats each record as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        # Anything passed via `extra=` (request_id, user, latency_ms, ...)
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                line[key] = value
        return json.dumps(line, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Route the root logger to JSON-on-stdout. Called once, at app startup.

    Args:
        level: Minimum level emitted (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]  # replace defaults: exactly one JSON stream

    # Noisy libraries: uvicorn's access lines duplicate what we log
    # ourselves, and httpx logs every request at INFO.
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
