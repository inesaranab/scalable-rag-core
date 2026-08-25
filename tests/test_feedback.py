"""Feedback: a scored verdict on an answer, recorded for later training."""

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from services.api.app.main import app
from services.api.app.routes.ask import enforce_rate_limit
from services.api.app.stores.postgres_feedback import FeedbackStore


def _authorized():
    return {"sub": "ines"}


@pytest.fixture
async def store():
    engine = create_async_engine("sqlite+aiosqlite://")
    store = FeedbackStore(engine)
    await store.ensure_table()
    yield store
    await engine.dispose()


async def test_feedback_is_stored_with_its_author(store):
    await store.add(
        session_id="s1", user_id="ines", score=1, comment="grounded answer"
    )

    [row] = await store.for_session("s1")
    assert row.user_id == "ines"
    assert row.score == 1
    assert row.comment == "grounded answer"


async def test_the_route_records_the_callers_feedback(store):
    app.state.feedback = store
    app.dependency_overrides[enforce_rate_limit] = _authorized

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as client:
        response = await client.post(
            "/feedback",
            json={"session_id": "s1", "score": -1, "comment": "hallucinated"},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    [row] = await store.for_session("s1")
    assert (row.user_id, row.score) == ("ines", -1)


async def test_a_score_outside_the_range_is_rejected(store):
    app.state.feedback = store
    app.dependency_overrides[enforce_rate_limit] = _authorized

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as client:
        response = await client.post(
            "/feedback", json={"session_id": "s1", "score": 7}
        )

    app.dependency_overrides.clear()
    assert response.status_code == 422
