"""The retriever node: hybrid search — vectors and graph, in parallel.

The query is first rewritten to stand alone (pronouns resolved from the
conversation), then HyDE turns it into a hypothetical answer whose
embedding lands nearer to real documents than the question's would.
Vector and graph search then run concurrently; a dead graph store costs
nothing but its results.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


def make_retriever(embedder, vector_store, graph_store, top_k, rewriter, hyde):
    """Build the retriever node around the injected search machinery.

    Args:
        embedder: Turns text into vectors.
        vector_store: Searches chunks by vector similarity.
        graph_store: Looks up entity relationships.
        top_k: How many chunks the vector search returns.
        rewriter: ``async (question, history) -> str`` standalone query.
        hyde: ``async (question) -> str`` hypothetical document.

    Returns:
        An async node: state -> {"documents": merged deduped evidence}.
    """

    async def retriever(state: dict) -> dict:
        question = state.get("current_query") or state["messages"][-1]["content"]
        history = state["messages"][:-1]

        standalone = await rewriter(question, history)
        hypothetical = await hyde(standalone)
        [vector] = await embedder.embed([hypothetical])

        # raises: crash loudly if the primary source dissapears
        async def vector_search() -> list[str]:
            return await vector_store.search(vector, top_k)

        # second source for enrichment: graceful degradation
        async def graph_search() -> list[str]:
            try:
                return await graph_store.lookup(standalone)
            except Exception:
                logger.warning("graph search failed", exc_info=True)
                return []

        # data dependency: each early step consumes the previous setp's output
        # so they form a chain that not concurrency can shorten.
        vector_docs, graph_docs = await asyncio.gather(vector_search(), graph_search())
        # dict.fromkeys deduplicates while keeping first-seen order.
        merged = list(dict.fromkeys(vector_docs + graph_docs))
        # Per-source counts: a graph that silently degrades to [] is
        # invisible inside a single total.
        logger.info(
            "retrieved",
            extra={
                "documents": len(merged),
                "vector": len(vector_docs),
                "graph": len(graph_docs),
            },
        )
        return {"documents": merged}

    return retriever
