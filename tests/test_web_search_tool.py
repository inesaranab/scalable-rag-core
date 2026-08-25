"""The web search tool: formatted results, and honest silence without a key."""

import httpx

from services.api.app.tools.web_search import make_web_search


def _tool(handler, api_key="test-key"):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return make_web_search(client, api_key=api_key)


async def test_results_come_back_titled():
    def handler(request):
        return httpx.Response(200, json={"results": [
            {"title": "K8s docs", "content": "Kubernetes schedules containers."},
        ]})

    search = _tool(handler)

    assert await search("kubernetes") == [
        "K8s docs: Kubernetes schedules containers."
    ]


async def test_no_api_key_means_disabled_not_broken():
    search = _tool(lambda request: httpx.Response(500), api_key="")

    [result] = await search("anything")
    assert "disabled" in result.lower()


async def test_a_dead_provider_degrades_to_a_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    search = _tool(handler)

    [result] = await search("anything")
    assert "error" in result.lower()
