"""Conversation memory: turns are stored and come back oldest-first."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from services.api.app.stores.postgres_memory import ChatMemoryStore


@pytest.fixture
async def memory():
    engine = create_async_engine("sqlite+aiosqlite://")
    store = ChatMemoryStore(engine)
    await store.ensure_table()
    yield store
    await engine.dispose()


async def test_a_turn_is_stored_and_recalled(memory):
    await memory.add_message("s1", role="user", content="hola", user_id="ines")

    history = await memory.get_history("s1")

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "hola"
    assert history[0].user_id == "ines"


async def test_history_comes_back_in_chronological_order(memory):
    await memory.add_message("s1", role="user", content="first", user_id="i")
    await memory.add_message("s1", role="assistant", content="second", user_id="i")
    await memory.add_message("s1", role="user", content="third", user_id="i")

    history = await memory.get_history("s1")

    assert [m.content for m in history] == ["first", "second", "third"]


async def test_limit_keeps_the_most_recent_turns(memory):
    for n in range(5):
        await memory.add_message("s1", role="user", content=f"t{n}", user_id="i")

    history = await memory.get_history("s1", limit=2)

    assert [m.content for m in history] == ["t3", "t4"]


async def test_sessions_do_not_leak_into_each_other(memory):
    await memory.add_message("s1", role="user", content="mine", user_id="i")
    await memory.add_message("s2", role="user", content="theirs", user_id="j")

    history = await memory.get_history("s1")

    assert [m.content for m in history] == ["mine"]
