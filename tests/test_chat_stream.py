"""/chat/stream: node progress and the answer arrive as NDJSON lines."""

import json

import fakeredis.aioredis
import httpx
import pytest

from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import enforce_rate_limit
from services.api.app.routes.chat import get_agent_for_chat
from services.api.app.stores.redis_gateway import RedisGateway


def _authorized():
    return {"sub": "ines"}


class FakeStreamingAgent:
    """Yields planner and responder events like a compiled graph would."""

    async def astream(self, state):
        yield {"planner": {"route": "respond"}}
        yield {"responder": {"messages": [
            {"role": "assistant", "content": "streamed answer"}
        ]}}


class FakeSemanticCache:
    def __init__(self, hit=None):
        self.hit = hit
        self.stored = []

    async def get(self, query):
        return self.hit

    async def set(self, query, answer):
        self.stored.append((query, answer))


class FakeMemory:
    def __init__(self):
        self.added = []

    async def get_history(self, session_id, limit=10):
        return []

    async def add_message(self, session_id, role, content, user_id):
        self.added.append((session_id, role, content, user_id))


@pytest.fixture
async def client():
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app.state.redis = RedisGateway(fake_redis)
    app.state.settings = Settings()
    app.state.semantic_cache = FakeSemanticCache()
    app.state.memory = FakeMemory()
    app.dependency_overrides[get_agent_for_chat] = lambda: FakeStreamingAgent()
    app.dependency_overrides[enforce_rate_limit] = _authorized

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as http:
        yield http

    app.dependency_overrides.clear()
    await fake_redis.aclose()


async def _lines(client, payload: dict) -> list[dict]:
    async with client.stream("POST", "/chat/stream", json=payload) as response:
        assert response.status_code == 200
        return [json.loads(line) async for line in response.aiter_lines() if line]


async def test_statuses_stream_before_the_answer(client):
    events = await _lines(client, {"message": "hola", "session_id": "s1"})

    kinds = [e["type"] for e in events]
    assert kinds == ["status", "status", "answer"]
    assert events[0]["node"] == "planner"
    assert events[-1]["content"] == "streamed answer"


async def test_both_turns_land_in_memory_after_the_stream(client):
    memory = app.state.memory

    await _lines(client, {"message": "hola", "session_id": "s1"})

    assert ("s1", "user", "hola", "ines") in memory.added
    assert ("s1", "assistant", "streamed answer", "ines") in memory.added


async def test_a_semantic_hit_streams_the_cached_answer_only(client):
    app.state.semantic_cache = FakeSemanticCache(hit="cached by meaning")

    events = await _lines(client, {"message": "explain k8s"})

    assert events == [{"type": "answer", "content": "cached by meaning"}]
