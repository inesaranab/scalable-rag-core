"""The API's client for the embedding service.

One connection pool for the process, opened at app startup and closed at
shutdown. Opening a client per call would pay a new TCP and TLS handshake
every request — the pool keeps sockets alive and reuses them.
"""

import httpx

from libs.retry.backoff import exponential_backoff

DEFAULT_TIMEOUT_S = 60.0
# Bounds on the pool: enough parallelism to saturate the embedder without
# opening an unbounded number of sockets to it.
MAX_KEEPALIVE_CONNECTIONS = 20
MAX_CONNECTIONS = 50


class RayEmbedClient:
    """Async client for the embedding service, pooled and retrying.

    Attributes:
        endpoint: Address of the embedding service.
        timeout_s: How long to wait for one embedding call.
        client: The pooled HTTP client. Created by ``start`` at app startup,
            closed by ``close`` at shutdown; None before ``start``.
    """

    def __init__(
        self, endpoint: str, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Open the connection pool. Called once, at app startup."""
        self.client = httpx.AsyncClient(
            timeout=self.timeout_s,
            limits=httpx.Limits(
                max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                max_connections=MAX_CONNECTIONS,
            ),
        )

    async def close(self) -> None:
        """Close the pool and its sockets. Called once, at app shutdown."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    @exponential_backoff(max_retries=2, retry_on=httpx.HTTPError)
    async def embed(
        self, texts: list[str], task_type: str = "query"
    ) -> list[list[float]]:
        """Embed texts through the service.

        Args:
            texts: One string per item to embed.
            task_type: ``"query"`` for search-time embedding, ``"document"``
                for ingestion. Some models embed the two differently.

        Returns:
            One vector per input text, in input order.

        Raises:
            RuntimeError: If called before ``start``.
            httpx.HTTPError: If the service keeps failing after the retries.
        """
        if self.client is None:
            raise RuntimeError("client not started; call start() at app startup")

        response = await self.client.post(
            self.endpoint,
            json={"text": texts, "task_type": task_type},
        )
        response.raise_for_status()
        return response.json()["embeddings"]
