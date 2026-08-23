"""The agent's tools: the actions the model may choose between.

Each tool composes adapters the composition root hands in; the tool
docstrings are shown verbatim to the model in its decide prompt, so they
are written for the model, not only for the reader.
"""

from typing import Awaitable, Callable

from services.api.app.ports import Embedder, VectorStore

Tool = Callable[[str], Awaitable[list[str]]]


def build_tools(
    embedder: Embedder,
    vector_store: VectorStore,
    graph_store,
    top_k: int,
) -> dict[str, Tool]:
    """Assemble the agent's tools around the connected adapters.

    Args:
        embedder: Turns text into vectors.
        vector_store: Searches chunks by vector similarity.
        graph_store: Looks up entity relationships in the graph.
        top_k: How many chunks a vector search returns.

    Returns:
        Tool name to async callable; every callable takes one string and
        returns a list of evidence strings.
    """

    async def vector_search(query: str) -> list[str]:
        """Find text chunks whose meaning is closest to the query."""
        [vector] = await embedder.embed([query])
        return await vector_store.search(vector, top_k)

    async def graph_lookup(entity: str) -> list[str]:
        """Find an entity's direct relationships in the knowledge graph."""
        return await graph_store.lookup(entity)

    return {"vector_search": vector_search, "graph_lookup": graph_lookup}
