"""Graph search that survives natural language.

A question is not an entity name, so exact lookup on the raw question
finds nothing. Here the LLM's only job is to NAME the entities the
question mentions; the lookup itself stays parametrized Cypher owned by
the store — the model never writes a query, so it cannot inject one.
"""

import json
import logging

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """Extract the core entities (proper names of systems,
people, organizations, concepts) from the question, for a knowledge-graph
lookup.

Question: {question}

Output ONLY JSON: {{"entities": ["name", "name"]}}"""


def make_entity_graph_search(llm, graph_store):
    """Build the tool around an injected LLM and graph store.

    Args:
        llm: Anything with ``async answer(prompt) -> str``; names the
            entities, nothing else.
        graph_store: Looks up one entity's relationships.

    Returns:
        ``async (question) -> list[str]``: relationship strings for every
        extracted entity; extraction failure degrades to no results.
    """

    async def search(question: str) -> list[str]:
        """Find graph facts about the entities a question mentions."""
        try:
            reply = await llm.answer(_EXTRACT_PROMPT.format(question=question))
            start, end = reply.index("{"), reply.rindex("}") + 1
            entities = json.loads(reply[start:end]).get("entities", [])
        except Exception:
            logger.warning("entity extraction failed", exc_info=True)
            return []

        results: list[str] = []
        for entity in entities:
            try:
                results.extend(await graph_store.lookup(str(entity)))
            except Exception:
                logger.warning("graph lookup failed", exc_info=True)
        return results

    return search
