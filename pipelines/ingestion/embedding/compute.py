"""Turning batches of chunk text into vectors.

The model is not loaded here. It lives in a serving process that holds its
weights in GPU memory permanently, and this sends text to it over HTTP.
Loading weights per task would pay the model's startup cost on every batch,
which for a large model is minutes rather than seconds.

Batching is what keeps the GPU busy: one request carrying many chunks is
processed in far less time than the same chunks sent one at a time.

A batch here is column-oriented — one dictionary whose values are lists —
because that is the shape a batch arrives in from a distributed data pipeline,
and because ``batch["text"]`` is then already the list the model expects.
Chunking produces the opposite shape, one dictionary per chunk::

    rows    [{"text": "one", "metadata": {...}}, {"text": "two", ...}]
    columns {"text": ["one", "two"], "metadata": [{...}, {...}]}

Nothing in this package converts between them yet. A pipeline framework does
it when batching rows; calling this directly means transposing first.
"""

from typing import Any

import httpx

DEFAULT_ENDPOINT = "http://ray-serve-embed:8000/embed"
DEFAULT_TIMEOUT_S = 30.0


class BatchEmbedder:
    """Sends batches of chunk text to the embedding service.

    One instance is constructed per worker and reused across batches, so the
    connection pool is established once rather than per request.

    Attributes:
        endpoint: Address of the embedding service. Resolves to internal
            cluster DNS when deployed.
        client: HTTP client held open across calls.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Prepare a reusable client for the embedding service.

        Args:
            endpoint: Address the service is reachable at.
            timeout_s: How long to wait for one batch before giving up.
        """
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=timeout_s)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Embed a batch of chunks.

        Args:
            batch: A batch carrying a ``text`` key, holding one string per
                chunk.

        Returns:
            The same batch with a ``vector`` key added, holding one embedding
            per chunk in the order the text was given.

        Raises:
            httpx.HTTPStatusError: If the service returns an error status.
            httpx.TimeoutException: If the batch is not embedded in time.
        """
        response = self.client.post(
            self.endpoint,
            json={"text": batch["text"], "task_type": "document"},
        )
        response.raise_for_status()

        batch["vector"] = response.json()["embeddings"]
        return batch
