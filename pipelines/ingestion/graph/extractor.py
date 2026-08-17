"""Extracting entities and relationships from chunk text.

This reads structure where embedding reads meaning: who is mentioned, and how
they are connected. It is one model call per chunk, which is why the graph
stage is the expensive half of ingestion and why its concurrency is bounded.

A chunk whose extraction fails contributes an empty graph rather than ending
the run. Ingestion processes thousands of chunks, and one malformed response
should not discard the work already done.
"""

import json
from typing import Any

import httpx

from pipelines.ingestion.graph.schema import GraphSchema

DEFAULT_ENDPOINT = "http://ray-serve-llm:8000/llm/chat"

# Extraction reasons over a full chunk, so it is slower than a single answer.
DEFAULT_TIMEOUT_S = 60.0

# The same chunk must always yield the same graph, or reindexing would rewrite
# relationships that never changed.
TEMPERATURE = 0.0

MAX_TOKENS = 1024


class GraphExtractor:
    """Calls the model service to extract a graph from each chunk in a batch.

    One instance is constructed per worker and reused across batches, so the
    connection pool is established once rather than per request.

    Attributes:
        llm_endpoint: Address of the model service. Resolves to internal
            cluster DNS when deployed.
        client: HTTP client held open across calls.
    """

    def __init__(
        self,
        llm_endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Prepare a reusable client for the model service.

        Args:
            llm_endpoint: Address the service is reachable at.
            timeout_s: How long to wait for one chunk's extraction.
        """
        self.llm_endpoint = llm_endpoint
        self.client = httpx.Client(timeout=timeout_s)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Extract a graph for every chunk in a batch.

        Args:
            batch: A batch carrying a ``text`` key, holding one string per
                chunk.

        Returns:
            The same batch with ``graph_nodes`` and ``graph_edges`` added,
            each holding one list per chunk in the order the text was given.
            A chunk whose extraction failed contributes empty lists.
        """
        nodes_list: list[list[Any]] = []
        edges_list: list[list[Any]] = []

        for text in batch["text"]:
            try:
                prompt = f"{GraphSchema.get_system_prompt()}\n\nInput Text:\n{text}"

                response = self.client.post(
                    self.llm_endpoint,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                    },
                )
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"]
                graph_data = json.loads(content)

                nodes_list.append(graph_data.get("nodes", []))
                edges_list.append(graph_data.get("edges", []))

            except Exception as exc:
                print(f"graph extraction failed for a chunk: {type(exc).__name__}")
                nodes_list.append([])
                edges_list.append([])

        batch["graph_nodes"] = nodes_list
        batch["graph_edges"] = edges_list
        return batch
