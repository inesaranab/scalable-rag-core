"""The agent's tools: each one drives the right adapters, nothing more."""

from services.api.app.agents.tools import build_tools


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.searched: list[tuple[list[float], int]] = []

    async def search(self, vector, top_k):
        self.searched.append((vector, top_k))
        return ["chunk about batching"]


class FakeGraphStore:
    def __init__(self):
        self.looked_up: list[str] = []

    async def lookup(self, entity):
        self.looked_up.append(entity)
        return ["Qdrant -[STORES]-> vectors"]


async def test_vector_search_embeds_then_searches():
    store = FakeVectorStore()
    tools = build_tools(FakeEmbedder(), store, FakeGraphStore(), top_k=4)

    result = await tools["vector_search"]("what is batching?")

    assert result == ["chunk about batching"]
    assert store.searched == [([0.1, 0.2, 0.3], 4)]


async def test_graph_lookup_reaches_the_graph_store():
    graph = FakeGraphStore()
    tools = build_tools(FakeEmbedder(), FakeVectorStore(), graph, top_k=4)

    result = await tools["graph_lookup"]("Qdrant")

    assert result == ["Qdrant -[STORES]-> vectors"]
    assert graph.looked_up == ["Qdrant"]


def test_every_tool_carries_a_docstring_for_the_agents_prompt():
    tools = build_tools(FakeEmbedder(), FakeVectorStore(), FakeGraphStore(), 4)

    for name, tool in tools.items():
        assert tool.__doc__, f"{name} has no docstring for the decide prompt"
