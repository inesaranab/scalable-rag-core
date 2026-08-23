"""The retriever node: rewrite, HyDE, parallel hybrid search, dedupe."""

from services.api.app.agents.nodes.retriever import make_retriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed(self, texts):
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    async def search(self, vector, top_k):
        return ["shared chunk", "vector-only chunk"]


class FakeGraphStore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def lookup(self, entity):
        if self.fail:
            raise ConnectionError("neo4j down")
        return ["shared chunk", "graph-only fact"]


async def fake_rewriter(question, history):
    return f"rewritten: {question}"


async def fake_hyde(question):
    return f"hypothetical doc for {question}"


def _make(graph_store=None, embedder=None):
    return make_retriever(
        embedder=embedder or FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=graph_store or FakeGraphStore(),
        top_k=4,
        rewriter=fake_rewriter,
        hyde=fake_hyde,
    )


def _state(query: str) -> dict:
    return {"messages": [{"role": "user", "content": query}],
            "documents": [], "current_query": query}


async def test_the_hyde_document_is_what_gets_embedded():
    embedder = FakeEmbedder()
    retriever = _make(embedder=embedder)

    await retriever(_state("what is k8s?"))

    # rewrite ran first, then HyDE, and the HyDE text was embedded.
    assert embedder.embedded == ["hypothetical doc for rewritten: what is k8s?"]


async def test_vector_and_graph_results_merge_without_duplicates():
    retriever = _make()

    update = await retriever(_state("what is k8s?"))

    assert update["documents"] == [
        "shared chunk", "vector-only chunk", "graph-only fact",
    ]


async def test_a_dead_graph_store_degrades_to_vector_only():
    retriever = _make(graph_store=FakeGraphStore(fail=True))

    update = await retriever(_state("what is k8s?"))

    assert update["documents"] == ["shared chunk", "vector-only chunk"]
