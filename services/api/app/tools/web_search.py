"""Web search via Tavily, a search API built for agents.

The corpus knows only what was ingested; this tool reaches the live
internet. Without an API key it announces itself disabled instead of
erroring, so the agent learns the truth either way.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def make_web_search(client: httpx.AsyncClient, api_key: str):
    """Build the tool around an injected HTTP client and key.

    Args:
        client: The pooled async client the calls go through.
        api_key: The Tavily key; empty string disables the tool.

    Returns:
        ``async (query) -> list[str]``: "title: content" lines; missing
        key or provider failure comes back as one message, never an
        exception.
    """

    async def web_search(query: str) -> list[str]:
        """Search the live internet for information beyond the corpus."""
        if not api_key:
            return ["Web search is disabled: no API key configured."]
        try:
            response = await client.post(
                TAVILY_URL,
                json={"api_key": api_key, "query": query, "max_results": 3},
                timeout=15.0,
            )
            results = response.json().get("results", [])
            return [f"{r['title']}: {r['content']}" for r in results]
        except Exception:
            logger.warning("web search failed", exc_info=True)
            return ["Web search error: the provider did not answer."]

    return web_search
