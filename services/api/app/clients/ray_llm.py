"""The API's client for the LLM service.

One connection pool for the process, opened at app startup and closed at
shutdown. LLM calls are long, so the timeout is generous and the retry
policy narrow: transient transport errors retry, everything else surfaces.
"""

import httpx

from libs.retry.backoff import exponential_backoff

# LLM generation is slow by nature; well under an ingress's request ceiling,
# but far above an embedding call.
DEFAULT_TIMEOUT_S = 120.0
MAX_KEEPALIVE_CONNECTIONS = 20
MAX_CONNECTIONS = 50

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024


class RayLLMClient:
    """Async client for the LLM service, pooled and retrying.

    Attributes:
        endpoint: Address of the LLM service.
        timeout_s: How long to wait for one generation.
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
    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Ask the LLM service to complete a conversation.

        Args:
            messages: The conversation, as role/content dicts.
            temperature: Sampling temperature the service should use.
            max_tokens: Cap on the generated length, in tokens.

        Returns:
            The assistant's reply text.

        Raises:
            RuntimeError: If called before ``start``.
            httpx.HTTPError: If the service keeps failing after the retries.
        """
        if self.client is None:
            raise RuntimeError("client not started; call start() at app startup")

        response = await self.client.post(
            self.endpoint,
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
