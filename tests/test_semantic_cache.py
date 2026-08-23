"""The semantic cache: matches by meaning, and never kills a request."""

from services.api.app.stores.semantic_cache import SemanticCacheStore


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakePoint:
    def __init__(self, score: float, payload: dict) -> None:
        self.score = score
        self.payload = payload


class FakeResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    """Answers with whatever points it was seeded with; records upserts."""

    def __init__(self, points=()) -> None:
        self.points = list(points)
        self.upserted = []

    async def collection_exists(self, name):
        return True

    async def query_points(self, collection_name, query, limit, score_threshold,
                           with_payload):
        hits = [p for p in self.points if p.score >= score_threshold]
        return FakeResponse(hits[:limit])

    async def upsert(self, collection_name, points):
        self.upserted.extend(points)


def _cache(qdrant) -> SemanticCacheStore:
    return SemanticCacheStore(
        FakeEmbedder(), qdrant, collection="semantic_cache", threshold=0.95
    )


async def test_a_similar_question_hits():
    """Similar means: similarity score at or above the threshold."""
    qdrant = FakeQdrant([FakePoint(0.97, {"answer": "K8s runs clusters"})])

    assert await _cache(qdrant).get("explain k8s") == "K8s runs clusters"


async def test_a_dissimilar_question_misses():
    qdrant = FakeQdrant([FakePoint(0.80, {"answer": "wrong topic"})])

    assert await _cache(qdrant).get("what is bcrypt?") is None


async def test_answers_are_stored_with_their_question():
    qdrant = FakeQdrant()
    await _cache(qdrant).set("what is qdrant?", "a vector database")

    [point] = qdrant.upserted
    assert point.payload == {
        "query": "what is qdrant?",
        "answer": "a vector database",
    }


async def test_a_broken_cache_returns_a_miss_not_an_error():
    class BrokenQdrant(FakeQdrant):
        async def query_points(self, *a, **k):
            raise ConnectionError("cache down")

    assert await _cache(BrokenQdrant()).get("anything") is None
