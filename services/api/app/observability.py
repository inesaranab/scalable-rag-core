"""Tracing: every request becomes a timed tree of spans.

A SPAN is one timed operation ("handle /ask", "call the LLM"); a TRACE is
the tree of spans one request produced, sharing a trace id — including
across services, because the id travels in an HTTP header. Where logs say
what happened, traces say where the time went.

Locally spans print to stdout; in production the exporter is swapped for
one that ships them to a collector (Azure Monitor, Jaeger) — config, not
code.
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)


def setup_observability(
    app: FastAPI, console: bool = False, service_name: str = "rag-api"
) -> None:
    """Attach OpenTelemetry tracing to the app. Called once, at startup.

    Spans are always created; whether they are EXPORTED anywhere is a
    deployment choice: ``console=True`` (Settings.otel_console) prints
    them to stdout for local demos; production adds a collector exporter
    instead. Tests pass neither, so nothing writes to streams pytest owns.

    Args:
        app: The FastAPI application to instrument.
        console: Whether spans print to stdout.
        service_name: How this service is labeled in every span it emits.
    """
    # factory: identity of the service and the pipeline every span travels
    # factory -> batch -> destination
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if console:
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )  # TODO: swap for Azure Monitor
    trace.set_tracer_provider(provider)

    # Auto-instrument: every route gets a span with method, path, status
    # and duration — no per-route code needed.
    FastAPIInstrumentor.instrument_app(app)
