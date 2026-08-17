"""Graph extraction: structure per chunk, and failures that do not spread."""

import json

import httpx

from pipelines.ingestion.graph.extractor import GraphExtractor


def extractor_returning(*responses: httpx.Response) -> GraphExtractor:
    """Build an extractor whose service returns the given responses in order.

    Args:
        *responses: One response per chunk the extractor will process. The
            last is repeated if more chunks arrive than responses given.

    Returns:
        An extractor wired to a stubbed transport, making no network call.
    """
    remaining = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        handler.requests.append(request)
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    handler.requests = []
    extractor = GraphExtractor(llm_endpoint="http://llm.test/chat")
    extractor.client = httpx.Client(transport=httpx.MockTransport(handler))
    extractor._handler = handler
    return extractor


def graph_response(nodes, edges) -> httpx.Response:
    """Build a service response carrying a graph as the model would return it.

    Args:
        nodes: Entities the model found.
        edges: Relationships the model found.

    Returns:
        A 200 response whose message content is the graph as JSON text.
    """
    content = json.dumps({"nodes": nodes, "edges": edges})
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_nodes_and_edges_are_added_to_the_batch():
    """Each chunk's graph is attached alongside its text."""
    extractor = extractor_returning(
        graph_response([{"label": "Person", "name": "Ada"}], [])
    )

    result = extractor({"text": ["Ada works here."]})

    assert result["graph_nodes"] == [[{"label": "Person", "name": "Ada"}]]
    assert result["graph_edges"] == [[]]


def test_one_result_per_chunk_in_order():
    """Positions line up with the text, so a graph can be traced to its chunk."""
    extractor = extractor_returning(
        graph_response([{"name": "first"}], []),
        graph_response([{"name": "second"}], []),
    )

    result = extractor({"text": ["one", "two"]})

    assert [nodes[0]["name"] for nodes in result["graph_nodes"]] == ["first", "second"]


def test_the_schema_is_sent_with_every_chunk():
    """The permitted vocabulary constrains each call, not just the first."""
    extractor = extractor_returning(graph_response([], []))

    extractor({"text": ["one", "two"]})

    for request in extractor._handler.requests:
        prompt = json.loads(request.content)["messages"][0]["content"]
        assert "Allowed Labels" in prompt
        assert "WORKS_FOR" in prompt


def test_the_chunk_text_reaches_the_model():
    """The text under extraction is included in the prompt."""
    extractor = extractor_returning(graph_response([], []))

    extractor({"text": ["Ada works at Acme."]})

    prompt = json.loads(extractor._handler.requests[0].content)["messages"][0]["content"]
    assert "Ada works at Acme." in prompt


def test_extraction_is_deterministic():
    """The same chunk must yield the same graph, or reindexing would rewrite
    relationships that never changed."""
    extractor = extractor_returning(graph_response([], []))

    extractor({"text": ["text"]})

    sent = json.loads(extractor._handler.requests[0].content)
    assert sent["temperature"] == 0.0


def test_a_failing_chunk_yields_an_empty_graph():
    """One bad response must not discard the work already done."""
    extractor = extractor_returning(httpx.Response(503))

    result = extractor({"text": ["text"]})

    assert result["graph_nodes"] == [[]]
    assert result["graph_edges"] == [[]]


def test_unparseable_output_yields_an_empty_graph():
    """A model returning prose instead of JSON is a failure, not a crash."""
    extractor = extractor_returning(
        httpx.Response(200, json={"choices": [{"message": {"content": "sorry!"}}]})
    )

    result = extractor({"text": ["text"]})

    assert result["graph_nodes"] == [[]]


def test_a_failing_chunk_does_not_affect_its_neighbours():
    """Positions are preserved, so a failure blanks one chunk and no other."""
    extractor = extractor_returning(
        httpx.Response(503),
        graph_response([{"name": "survived"}], []),
    )

    result = extractor({"text": ["fails", "works"]})

    assert result["graph_nodes"][0] == []
    assert result["graph_nodes"][1] == [{"name": "survived"}]


def test_a_graph_missing_keys_yields_empty_lists():
    """A response without nodes or edges is treated as an empty graph."""
    extractor = extractor_returning(
        httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )

    result = extractor({"text": ["text"]})

    assert result["graph_nodes"] == [[]]
    assert result["graph_edges"] == [[]]


def test_the_original_batch_keys_survive():
    """Metadata travelling alongside the text is not lost."""
    extractor = extractor_returning(graph_response([], []))

    result = extractor({"text": ["text"], "metadata": [{"chunk_index": 0}]})

    assert result["metadata"] == [{"chunk_index": 0}]


def test_an_empty_batch_calls_nothing():
    """No chunks means no model calls, and no cost."""
    extractor = extractor_returning(graph_response([], []))

    result = extractor({"text": []})

    assert result["graph_nodes"] == []
    assert extractor._handler.requests == []
