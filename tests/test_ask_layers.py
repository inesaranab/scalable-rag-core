"""/ask's answer layers: exact cache, then semantic cache, then the models.

Also: with a session_id, both turns land in conversation memory.
"""

import fakeredis.aioredis
import httpx
import pytest

from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import enforce_rate_limit, get_retrieval_service
from services.api.app.stores.redis_gateway import RedisGateway


def _authorized():
    return {"sub": "ines"}


class FakeService:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def answer(self, question: str, history=None) -> str:
        self.asked.append(question)
        return "from the models"


class FakeSemanticCache:
    def __init__(self, hit: str | None = None) -> None:
        self.hit = hit
        self.stored: list[tuple[str, str]] = []

    async def get(self, query: str) -> str | None:
        return self.hit

    async def set(self, query: str, answer: str) -> None:
        self.stored.append((query, answer))


class FakeMemory:
    def __init__(self) -> None:
        self.added: list[tuple[str, str, str, str]] = []

    async def add_message(self, session_id, role, content, user_id) -> None:
        self.added.append((session_id, role, content, user_id))

    async def get_history(self, session_id, limit=10):
        return []


@pytest.fixture
async def client():
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app.state.redis = RedisGateway(fake_redis)
    app.state.settings = Settings()
    app.state.semantic_cache = FakeSemanticCache()
    app.state.memory = FakeMemory()

    service = FakeService()
    app.dependency_overrides[get_retrieval_service] = lambda: service
    app.dependency_overrides[enforce_rate_limit] = _authorized

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as http:
        http.service = service
        yield http

    app.dependency_overrides.clear()
    await fake_redis.aclose()


async def test_a_semantic_hit_skips_the_models(client):
    app.state.semantic_cache = FakeSemanticCache(hit="cached by meaning")

    response = await client.post("/ask", json={"question": "explain k8s"})

    assert response.json() == {"answer": "cached by meaning"}
    assert client.service.asked == []


async def test_an_exact_hit_beats_the_semantic_cache(client):
    semantic = FakeSemanticCache(hit="semantic answer")
    app.state.semantic_cache = semantic
    await app.state.redis.set_cached(
        __import__("hashlib").sha256(b"repeat").hexdigest(), "exact answer", 60
    )

    response = await client.post("/ask", json={"question": "repeat"})

    assert response.json() == {"answer": "exact answer"}


async def test_a_full_miss_runs_the_models_and_fills_both_caches(client):
    semantic = app.state.semantic_cache

    response = await client.post("/ask", json={"question": "fresh question"})

    assert response.json() == {"answer": "from the models"}
    assert client.service.asked == ["fresh question"]
    assert semantic.stored == [("fresh question", "from the models")]


async def test_a_session_records_both_turns(client):
    memory = app.state.memory

    await client.post(
        "/ask", json={"question": "hola", "session_id": "s1"}
    )

    assert ("s1", "user", "hola", "ines") in memory.added
    assert ("s1", "assistant", "from the models", "ines") in memory.added


async def test_no_session_records_nothing(client):
    memory = app.state.memory

    await client.post("/ask", json={"question": "hola"})

    assert memory.added == []
