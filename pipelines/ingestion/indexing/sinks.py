"""Adapters that let the indexers terminate a distributed pipeline.

The pipeline's write API hands each task a block — a pyarrow Table — and
serialises the sink object to every worker. Two consequences shape these
classes. Blocks are converted back to the column-oriented dictionaries the
indexers speak. And no connection is opened until the first write, because a
live client cannot be serialised; each worker opens its own on first use.

Datasink contract:
    https://docs.ray.io/en/latest/data/api/doc/ray.data.Datasink.html
"""

from collections.abc import Iterable
from typing import Any

from ray.data import Datasink
from ray.data._internal.execution.interfaces import TaskContext

from pipelines.ingestion.indexing.neo4j import Neo4jIndexer
from pipelines.ingestion.indexing.qdrant import QdrantIndexer


def _to_columns(block: Any) -> dict[str, list[Any]]:
    """Convert one block to the column-oriented shape the indexers expect.

    Args:
        block: A pyarrow Table, or a pandas DataFrame on older codepaths.

    Returns:
        One dictionary whose values are lists, one entry per row.
    """
    if hasattr(block, "to_pydict"):
        return block.to_pydict()
    return {name: list(values) for name, values in block.items()}


class QdrantSink(Datasink[None]):
    """Terminates the embedding branch by writing vectors to Qdrant."""

    def __init__(self) -> None:
        """Prepare a sink with no connection, which cannot be serialised."""
        self._indexer: QdrantIndexer | None = None

    def write(self, blocks: Iterable[Any], ctx: TaskContext) -> None:
        """Write every block this task received.

        Args:
            blocks: The task's share of the dataset.
            ctx: Execution details, unused.
        """
        if self._indexer is None:
            self._indexer = QdrantIndexer()
        for block in blocks:
            self._indexer.write(_to_columns(block))


class Neo4jSink(Datasink[None]):
    """Terminates the graph branch by writing entities and relationships."""

    def __init__(self) -> None:
        """Prepare a sink with no connection, which cannot be serialised."""
        self._indexer: Neo4jIndexer | None = None

    def write(self, blocks: Iterable[Any], ctx: TaskContext) -> None:
        """Write every block this task received.

        Args:
            blocks: The task's share of the dataset.
            ctx: Execution details, unused.
        """
        if self._indexer is None:
            self._indexer = Neo4jIndexer()
        for block in blocks:
            self._indexer.write(_to_columns(block))
