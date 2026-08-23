"""Redis on the /ask route: the rate limit refuses, the cache short-circuits."""

import fakeredis.aioredis
import httpx
import pytest

from services.api.app.auth import require_token
from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import get_retrieval_service
from services.api.app.stores.redis_gateway import RedisGateway


def _authorized():
    """Stand-in for a verified token; the sub is what the limiter counts."""
    return {"sub": "ines"}


class FakeService:
    """Stands in for RetrievalService, counting how often it is called."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def answer(self, question: str, history=None) -> str:
        self.asked.append(question)
        return f"answer #{len(self.asked)}"


class NeverHitsSemanticCache:
    """A semantic cache that always misses and swallows writes."""

    async def get(self, query):
        return None

    async def set(self, query, answer):
        pass


@pytest.fixture
async def client():
    """An HTTP client against the app, with a fake Redis behind the gateway."""
    fake_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app.state.redis = RedisGateway(fake_client)
    # The lifespan does not run under the test transport; the routes read
    # limits and TTLs from here.
    app.state.settings = Settings()
    app.state.semantic_cache = NeverHitsSemanticCache()

    fake = FakeService()
    app.dependency_overrides[get_retrieval_service] = lambda: fake
    app.dependency_overrides[require_token] = _authorized

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as http:
        http.fake_service = fake
        yield http

    app.dependency_overrides.clear()
    await fake_client.aclose()


async def test_the_request_over_the_limit_gets_429(client):
    for i in range(30):
        response = await client.post("/ask", json={"question": f"q{i}"})
        assert response.status_code == 200

    refused = await client.post("/ask", json={"question": "one too many"})

    assert refused.status_code == 429
    # The service never saw the refused question.
    assert "one too many" not in client.fake_service.asked


async def test_a_repeated_question_is_answered_from_the_cache(client):
    first = await client.post("/ask", json={"question": "what is qdrant?"})
    second = await client.post("/ask", json={"question": "what is qdrant?"})

    assert first.json() == second.json()
    # The expensive path ran once; the second answer came from Redis.
    assert client.fake_service.asked == ["what is qdrant?"]


async def test_different_questions_do_not_share_a_cache_entry(client):
    await client.post("/ask", json={"question": "what is qdrant?"})
    await client.post("/ask", json={"question": "what is neo4j?"})

    assert client.fake_service.asked == ["what is qdrant?", "what is neo4j?"]
