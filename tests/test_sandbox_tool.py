"""The sandbox client tool: happy path, error path, unreachable sandbox."""

import httpx

from services.api.app.agents.sandbox_tool import make_sandbox_tool


def _tool_with(handler) -> object:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return make_sandbox_tool(client, endpoint="http://sandbox.test/execute")


async def test_successful_output_is_returned_as_evidence():
    def handler(request):
        return httpx.Response(
            200, json={"status": "success", "output": "42\n"}
        )

    run = _tool_with(handler)

    assert await run("print(21*2)") == ["Code output:\n42\n"]


async def test_an_execution_error_is_reported_not_raised():
    def handler(request):
        return httpx.Response(
            200, json={"status": "error", "output": "NameError: x"}
        )

    run = _tool_with(handler)

    [result] = await run("print(x)")
    assert "NameError" in result


async def test_an_unreachable_sandbox_degrades_to_a_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    run = _tool_with(handler)

    [result] = await run("print(1)")
    assert "unavailable" in result.lower()
