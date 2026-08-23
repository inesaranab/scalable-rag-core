"""The users table: hashes in, hashes out, never a plaintext password.

These tests run against a REAL database engine (SQLite in memory), so the
SQL that SQLModel generates is actually executed, not faked.
"""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from services.api.app.stores.postgres_users import PostgresUserStore


@pytest.fixture
async def store():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    store = PostgresUserStore(engine)
    await store.ensure_table()
    yield store
    await engine.dispose()


async def test_an_upserted_user_can_be_looked_up(store):
    await store.upsert_user("ines", "$2b$04$somehash")

    assert await store.get_password_hash("ines") == "$2b$04$somehash"


async def test_an_unknown_user_yields_none(store):
    assert await store.get_password_hash("mallory") is None


async def test_upserting_again_replaces_the_hash(store):
    await store.upsert_user("ines", "$2b$04$oldhash")
    await store.upsert_user("ines", "$2b$04$newhash")

    assert await store.get_password_hash("ines") == "$2b$04$newhash"
