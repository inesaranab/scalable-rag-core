"""The agent's door to the code sandbox.

The API never executes model-written code itself; it mails the code to
the sandbox service and relays the result. A dead sandbox degrades to a
message — losing a tool must never kill the request.
"""

import logging

import httpx

logger = logging.getLogger(__name__)


# we inject the client as an httpx.AsyncClient holds a connection pool (open sockets it reuses)
# if the function created its own client inside each call, every sandbox request would pay a fresh
# TCP + TLS handshake
def make_sandbox_tool(client: httpx.AsyncClient, endpoint: str):
    """Build the tool around an injected HTTP client.

    Args:
        client: The pooled async client the calls go through.
        endpoint: The sandbox service's /execute address.

    Returns:
        ``async (code) -> list[str]``: the code's output as one evidence
        string; execution errors and an unreachable sandbox come back as
        messages, never exceptions.
    """

    async def run_python_code(code: str) -> list[str]:
        """Run Python code in an isolated sandbox and return its output."""
        try:
            # network out + sandbox starting child process + full 5s child
            # kill and respond + network back
            # outer timeout = inner timeout + overhead around
            response = await client.post(
                endpoint, json={"code": code, "timeout": 5}, timeout=10.0
            )
            body = response.json()
            if body.get("status") == "success":
                return [f"Code output:\n{body['output']}"]
            return [f"Code execution error:\n{body.get('output', response.text)}"]
        except Exception:
            logger.warning("sandbox unreachable", exc_info=True)
            return ["Code execution unavailable: the sandbox did not answer."]

    return run_python_code
