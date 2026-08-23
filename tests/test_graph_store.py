"""The graph lookup: entities and their connections, as facts for the agent."""

import pytest

from services.api.app.stores.neo4j_store import Neo4jGraphStore


class FakeResult:
    """Stands in for a neo4j result: iterable of record-like dicts."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    async def data(self) -> list[dict]:
        return self._rows


class FakeSession:
    """Records the Cypher and parameters; answers with canned rows."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict]] = []

    async def run(self, query: str, **params) -> FakeResult:
        self.queries.append((query, params))
        return FakeResult(self.rows)

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class FakeDriver:
    """Hands out the prepared session."""

    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def session(self) -> FakeSession:
        return self._session


@pytest.fixture
def wired():
    session = FakeSession(
        rows=[
            {"source": "Qdrant", "relation": "STORES", "target": "vectors"},
            {"source": "Qdrant", "relation": "QUERIED_BY", "target": "RetrievalService"},
        ]
    )
    store = Neo4jGraphStore(FakeDriver(session))
    return store, session


async def test_lookup_returns_readable_facts(wired):
    """Each graph row becomes one 'source -RELATION-> target' fact string."""
    store, _ = wired

    facts = await store.lookup("Qdrant")

    assert facts == [
        "Qdrant -[STORES]-> vectors",
        "Qdrant -[QUERIED_BY]-> RetrievalService",
    ]


async def test_the_entity_is_passed_as_a_parameter_not_pasted(wired):
    """The entity travels as a bound parameter — no Cypher injection."""
    store, session = wired

    await store.lookup("Qdrant'; MATCH (n) DETACH DELETE n //")

    query, params = session.queries[0]
    assert "$name" in query
    assert params["name"] == "Qdrant'; MATCH (n) DETACH DELETE n //"
