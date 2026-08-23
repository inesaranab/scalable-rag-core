"""The API-side client for the LLM service: pooled, retrying, async."""

import json

import httpx
import pytest

from libs.retry import backoff
from services.api.app.clients.ray_llm import RayLLMClient


@pytest.fixture
def no_waits(monkeypatch):
    """Retries are asserted, not endured: replace the retry sleep."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(backoff.asyncio, "sleep", fake_sleep)
    return waits


def client_backed_by(handler) -> RayLLMClient:
    """Build a started client whose transport is the given handler.

    Args:
        handler: Function from request to httpx.Response, standing in for
            the LLM service.

    Returns:
        A client wired to the stubbed transport, making no network call.
    """
    client = RayLLMClient(endpoint="http://llm.test/chat")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def service_reply(content: str) -> dict:
    """The LLM deployment's response shape (vllm_engine.py's __call__)."""
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


async def test_messages_are_posted_and_the_answer_extracted():
    """The service's contract: messages in, the assistant's text out."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=service_reply("Paris."))

    client = client_backed_by(handler)
    messages = [{"role": "user", "content": "Capital of France?"}]

    answer = await client.chat_completion(messages)

    assert answer == "Paris."
    assert seen[0]["messages"] == messages
    assert 0 <= seen[0]["temperature"] <= 1
    assert seen[0]["max_tokens"] > 0


async def test_one_connection_pool_serves_many_calls():
    """The transport is created once and reused, not opened per call."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=service_reply("ok"))

    client = client_backed_by(handler)
    pool_before = client.client

    await client.chat_completion([{"role": "user", "content": "one"}])
    await client.chat_completion([{"role": "user", "content": "two"}])

    assert client.client is pool_before


async def test_a_transient_failure_is_retried(no_waits):
    """One 500 does not fail the call; the retry succeeds."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(500)
        return httpx.Response(200, json=service_reply("recovered"))

    client = client_backed_by(handler)

    answer = await client.chat_completion([{"role": "user", "content": "hi"}])

    assert answer == "recovered"
    assert attempts == 2
    assert len(no_waits) == 1


async def test_calling_before_start_is_a_clear_error():
    """A never-started client fails loudly, not with a cryptic AttributeError."""
    client = RayLLMClient(endpoint="http://llm.test/chat")

    with pytest.raises(RuntimeError):
        await client.chat_completion([{"role": "user", "content": "too early"}])
