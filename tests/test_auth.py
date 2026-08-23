"""JWT auth: only requests bearing a valid signed token reach the service."""

from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import httpx
import pytest
from jose import jwt

from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import get_retrieval_service
from services.api.app.stores.redis_gateway import RedisGateway

SECRET = "test-secret-key-at-least-16-chars"


class FakeService:
    async def answer(self, question: str, history=None) -> str:
        return "42"


class NeverHitsSemanticCache:
    """A semantic cache that always misses and swallows writes."""

    async def get(self, query):
        return None

    async def set(self, query, answer):
        pass


def token(secret: str = SECRET, expires_in_s: int = 3600) -> str:
    claims = {
        "sub": "ines",
        "exp": datetime.now(UTC) + timedelta(seconds=expires_in_s),
    }
    return jwt.encode(claims, secret, algorithm="HS256")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    app.dependency_overrides[get_retrieval_service] = lambda: FakeService()
    # The real rate limiter and cache run here; the lifespan does not.
    app.state.redis = RedisGateway(
        fakeredis.aioredis.FakeRedis(decode_responses=True)
    )
    app.state.settings = Settings()
    app.state.semantic_cache = NeverHitsSemanticCache()
    transport = httpx.ASGITransport(app=app)
    yield httpx.AsyncClient(transport=transport, base_url="http://api.test")
    app.dependency_overrides.clear()


async def test_a_valid_token_is_admitted(client):
    async with client as c:
        response = await c.post(
            "/ask",
            json={"question": "hi"},
            headers={"Authorization": f"Bearer {token()}"},
        )
    assert response.status_code == 200


async def test_no_token_is_rejected(client):
    async with client as c:
        response = await c.post("/ask", json={"question": "hi"})
    assert response.status_code in (401, 403)


async def test_a_token_signed_with_the_wrong_secret_is_rejected(client):
    async with client as c:
        response = await c.post(
            "/ask",
            json={"question": "hi"},
            headers={"Authorization": f"Bearer {token(secret='wrong-secret-016chars')}"},
        )
    assert response.status_code == 401


async def test_an_expired_token_is_rejected(client):
    async with client as c:
        response = await c.post(
            "/ask",
            json={"question": "hi"},
            headers={"Authorization": f"Bearer {token(expires_in_s=-10)}"},
        )
    assert response.status_code == 401
