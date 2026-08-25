"""Entity-extraction graph search: the LLM names entities, Cypher stays ours."""

import json

from services.api.app.tools.graph_search import make_entity_graph_search


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class FakeGraphStore:
    def __init__(self) -> None:
        self.looked_up: list[str] = []

    async def lookup(self, entity):
        self.looked_up.append(entity)
        if entity == "Qdrant":
            return ["Qdrant -[STORES]-> vectors"]
        return []


async def test_extracted_entities_are_looked_up_not_the_question():
    llm = ScriptedLLM(json.dumps({"entities": ["Qdrant", "Kubernetes"]}))
    store = FakeGraphStore()
    search = make_entity_graph_search(llm, store)

    results = await search("what does Qdrant store and where does it run?")

    # The store received entity NAMES, never the raw question.
    assert store.looked_up == ["Qdrant", "Kubernetes"]
    assert results == ["Qdrant -[STORES]-> vectors"]


async def test_a_confused_extractor_degrades_to_no_results():
    llm = ScriptedLLM("I am not JSON")
    search = make_entity_graph_search(llm, FakeGraphStore())

    assert await search("anything") == []


async def test_a_dead_llm_degrades_to_no_results():
    class BrokenLLM:
        async def answer(self, prompt):
            raise ConnectionError("down")

    search = make_entity_graph_search(BrokenLLM(), FakeGraphStore())

    assert await search("anything") == []
