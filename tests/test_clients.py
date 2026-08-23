"""The database clients: connect builds the pool, health pings, close resets."""

import fakeredis.aioredis

from services.api.app.clients.neo4j import Neo4jClient
from services.api.app.clients.postgres import PostgresClient
from services.api.app.clients.qdrant import QdrantClient
from services.api.app.clients.redis import RedisClient


async def test_redis_client_lifecycle(monkeypatch):
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "services.api.app.clients.redis.redis.from_url", lambda *a, **k: fake
    )
    client = RedisClient(url="redis://unused.test")
    assert client.client is None

    await client.connect()
    assert await client.health() is True

    await client.close()
    assert client.client is None


async def test_postgres_client_lifecycle():
    # An in-memory SQLite database answers SELECT 1 like Postgres does.
    client = PostgresClient(dsn="sqlite+aiosqlite://")
    assert client.engine is None

    await client.connect()
    assert await client.health() is True

    await client.close()
    assert client.engine is None


async def test_qdrant_client_lifecycle(monkeypatch):
    class StubSDK:
        async def get_collections(self):
            return []

        async def close(self):
            pass

    monkeypatch.setattr(
        "services.api.app.clients.qdrant.AsyncQdrantClient",
        lambda *a, **k: StubSDK(),
    )
    client = QdrantClient(url="http://unused.test")
    assert client.client is None

    await client.connect()
    assert await client.health() is True

    await client.close()
    assert client.client is None


async def test_neo4j_client_lifecycle(monkeypatch):
    class StubDriver:
        async def verify_connectivity(self):
            return None

        async def close(self):
            pass

    monkeypatch.setattr(
        "services.api.app.clients.neo4j.AsyncGraphDatabase.driver",
        lambda *a, **k: StubDriver(),
    )
    client = Neo4jClient(uri="bolt://unused.test", user="u", password="p")
    assert client.driver is None

    await client.connect()
    assert await client.health() is True

    await client.close()
    assert client.driver is None


async def test_an_unreachable_service_reports_unhealthy(monkeypatch):
    class BrokenSDK:
        async def get_collections(self):
            raise ConnectionError("down")

        async def close(self):
            pass

    monkeypatch.setattr(
        "services.api.app.clients.qdrant.AsyncQdrantClient",
        lambda *a, **k: BrokenSDK(),
    )
    client = QdrantClient(url="http://unused.test")
    await client.connect()

    assert await client.health() is False
