"""/agent/ask with a session: history flows in, both turns are recorded."""

import httpx
import pytest

from services.api.app.main import app
from services.api.app.routes.agent import get_agent
from services.api.app.routes.ask import enforce_rate_limit


def _authorized():
    return {"sub": "ines"}


class FakeAgent:
    """Records the initial state; answers with a fixed assistant turn."""

    def __init__(self) -> None:
        self.invoked_with: list[dict] = []

    async def ainvoke(self, state: dict) -> dict:
        self.invoked_with.append(state)
        return {"messages": state["messages"]
                + [{"role": "assistant", "content": "the answer"}]}


class FakeMemory:
    def __init__(self, turns=()) -> None:
        self.turns = list(turns)
        self.added: list[tuple[str, str, str, str]] = []

    async def get_history(self, session_id, limit=10):
        return self.turns

    async def add_message(self, session_id, role, content, user_id) -> None:
        self.added.append((session_id, role, content, user_id))


class Turn:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


@pytest.fixture
async def client():
    from services.api.app.config import Settings

    app.state.settings = Settings()
    fake_agent = FakeAgent()
    app.state.memory = FakeMemory()
    app.dependency_overrides[get_agent] = lambda: fake_agent
    app.dependency_overrides[enforce_rate_limit] = _authorized

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as http:
        http.agent = fake_agent
        yield http

    app.dependency_overrides.clear()


async def test_a_session_feeds_history_into_the_graph(client):
    app.state.memory = FakeMemory([
        Turn("user", "tell me about Kubernetes"),
        Turn("assistant", "Kubernetes runs clusters."),
    ])

    response = await client.post(
        "/agent/ask", json={"question": "how much does it cost?",
                            "session_id": "s1"}
    )

    assert response.json() == {"answer": "the answer"}
    [state] = client.agent.invoked_with
    assert [m["content"] for m in state["messages"]] == [
        "tell me about Kubernetes",
        "Kubernetes runs clusters.",
        "how much does it cost?",
    ]


async def test_a_session_records_both_turns(client):
    memory = FakeMemory()
    app.state.memory = memory

    await client.post(
        "/agent/ask", json={"question": "hola", "session_id": "s1"}
    )

    assert ("s1", "user", "hola", "ines") in memory.added
    assert ("s1", "assistant", "the answer", "ines") in memory.added


async def test_no_session_stays_single_turn_and_records_nothing(client):
    memory = FakeMemory([Turn("user", "old irrelevant turn")])
    app.state.memory = memory

    await client.post("/agent/ask", json={"question": "hola"})

    [state] = client.agent.invoked_with
    assert [m["content"] for m in state["messages"]] == ["hola"]
    assert memory.added == []
