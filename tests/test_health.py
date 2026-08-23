"""GET /health: one line per database, 503 the moment any stops answering."""

import httpx

from services.api.app.main import app


class FakeClient:
    """Stands in for any database client; health is fixed at creation."""

    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    async def health(self) -> bool:
        return self.healthy


def _install(**clients: FakeClient) -> None:
    app.state.clients = clients


async def _get_health() -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as client:
        return await client.get("/health")


async def test_all_healthy_answers_200():
    _install(qdrant=FakeClient(True), redis=FakeClient(True))

    response = await _get_health()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dependencies": {"qdrant": True, "redis": True},
    }


async def test_one_sick_dependency_answers_503_and_names_it():
    _install(qdrant=FakeClient(True), redis=FakeClient(False))

    response = await _get_health()

    assert response.status_code == 503
    assert response.json()["dependencies"] == {"qdrant": True, "redis": False}
    assert response.json()["status"] == "degraded"


async def test_health_needs_no_token():
    """Probes cannot log in; the endpoint must answer without auth."""
    _install(qdrant=FakeClient(True))

    response = await _get_health()

    assert response.status_code != 401
