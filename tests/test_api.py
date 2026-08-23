"""The API's front door: POST /ask runs the retrieval service."""

import httpx

import fakeredis.aioredis

from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import enforce_rate_limit, get_retrieval_service
from services.api.app.stores.redis_gateway import RedisGateway


def _fake_redis() -> RedisGateway:
    """A gateway over in-memory Redis; the cache path needs one on app.state."""
    return RedisGateway(fakeredis.aioredis.FakeRedis(decode_responses=True))


def _authorized():
    """Stand-in for token check and rate limit: these tests exercise routing."""
    return {"sub": "test"}


class FakeService:
    """Stands in for RetrievalService, recording the question."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def answer(self, question: str, history=None) -> str:
        self.asked.append(question)
        return "42"


class NeverHitsSemanticCache:
    """A semantic cache that always misses and swallows writes."""

    async def get(self, query):
        return None

    async def set(self, query, answer):
        pass


async def test_ask_returns_the_services_answer():
    """The route hands the question to the service and returns its answer."""
    fake = FakeService()
    app.dependency_overrides[get_retrieval_service] = lambda: fake
    app.dependency_overrides[enforce_rate_limit] = _authorized
    app.state.redis = _fake_redis()
    app.state.settings = Settings()
    app.state.semantic_cache = NeverHitsSemanticCache()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as client:
        response = await client.post(
            "/ask", json={"question": "what is the answer?"}
        )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"answer": "42"}
    assert fake.asked == ["what is the answer?"]


async def test_a_missing_question_is_rejected():
    """No question, no screening: validation fails before the service runs."""
    fake = FakeService()
    app.dependency_overrides[get_retrieval_service] = lambda: fake
    app.dependency_overrides[enforce_rate_limit] = _authorized
    app.state.redis = _fake_redis()
    app.state.settings = Settings()
    app.state.semantic_cache = NeverHitsSemanticCache()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as client:
        response = await client.post("/ask", json={})

    app.dependency_overrides.clear()
    assert response.status_code == 422
    assert fake.asked == []
