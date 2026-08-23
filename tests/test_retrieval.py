"""The read path: a question is embedded, matched, and answered in context."""

import pytest

from services.api.app.retrieval import RetrievalService


class FakeEmbedder:
    """Returns a fixed vector per text, recording what it was asked."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    """Returns canned chunks, recording the query vector and k."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.queried_with: list[float] | None = None
        self.k: int | None = None

    async def search(self, vector: list[float], k: int) -> list[str]:
        self.queried_with = vector
        self.k = k
        return self.chunks[:k]


class FakeLLM:
    """Echoes a canned answer, recording the prompt it received."""

    def __init__(self) -> None:
        self.prompt: str | None = None

    async def answer(self, prompt: str) -> str:
        self.prompt = prompt
        return "a vector database stores embeddings"


@pytest.fixture
def wired():
    embedder = FakeEmbedder()
    store = FakeVectorStore(["chunk one", "chunk two", "chunk three"])
    llm = FakeLLM()
    service = RetrievalService(embedder=embedder, store=store, llm=llm, top_k=2)
    return service, embedder, store, llm


async def test_the_question_travels_embedded_to_the_store(wired):
    """The store is searched with the question's vector, not its text."""
    service, embedder, store, _ = wired

    await service.answer("what is a vector database?")

    assert embedder.seen == ["what is a vector database?"]
    assert store.queried_with == [0.1, 0.2, 0.3]
    assert store.k == 2


async def test_retrieved_chunks_reach_the_llm_with_the_question(wired):
    """The prompt carries both the evidence and the question."""
    service, _, _, llm = wired

    await service.answer("what is a vector database?")

    assert "chunk one" in llm.prompt
    assert "chunk two" in llm.prompt
    assert "what is a vector database?" in llm.prompt


async def test_the_llms_answer_is_returned_unchanged(wired):
    service, _, _, _ = wired

    result = await service.answer("what is a vector database?")

    assert result == "a vector database stores embeddings"
