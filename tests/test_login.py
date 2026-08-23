"""The /token login: correct credentials mint a JWT that opens /ask."""

import fakeredis.aioredis
import httpx
import pytest

from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import get_retrieval_service
from services.api.app.routes.auth import get_user_store
from services.api.app.stores.redis_gateway import RedisGateway

SECRET = "test-secret-key-at-least-16-chars"
# bcrypt hash of "correct horse" (cost 4 for test speed)
import bcrypt

PASSWORD = "correct horse"
PASSWORD_HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4)).decode()


class FakeService:
    async def answer(self, question: str, history=None) -> str:
        return "42"


class NeverHitsSemanticCache:
    """A semantic cache that always misses and swallows writes."""

    async def get(self, query):
        return None

    async def set(self, query, answer):
        pass


class FakeUserStore:
    """One user in memory, hash-only, like the real table."""

    def __init__(self) -> None:
        self.users = {"ines": PASSWORD_HASH}

    async def get_password_hash(self, username: str) -> str | None:
        return self.users.get(username)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    app.dependency_overrides[get_retrieval_service] = lambda: FakeService()
    app.dependency_overrides[get_user_store] = lambda: FakeUserStore()
    # The real rate limiter and cache run here; the lifespan does not.
    app.state.redis = RedisGateway(
        fakeredis.aioredis.FakeRedis(decode_responses=True)
    )
    app.state.settings = Settings()
    app.state.semantic_cache = NeverHitsSemanticCache()
    transport = httpx.ASGITransport(app=app)
    yield httpx.AsyncClient(transport=transport, base_url="http://api.test")
    app.dependency_overrides.clear()


async def test_correct_credentials_yield_a_working_token(client):
    async with client as c:
        login = await c.post(
            "/token", data={"username": "ines", "password": PASSWORD}
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        assert login.json()["token_type"] == "bearer"

        response = await c.post(
            "/ask",
            json={"question": "hi"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200


async def test_a_wrong_password_is_rejected(client):
    async with client as c:
        login = await c.post(
            "/token", data={"username": "ines", "password": "wrong"}
        )
    assert login.status_code == 401


async def test_an_unknown_username_is_rejected(client):
    async with client as c:
        login = await c.post(
            "/token", data={"username": "mallory", "password": PASSWORD}
        )
    assert login.status_code == 401
