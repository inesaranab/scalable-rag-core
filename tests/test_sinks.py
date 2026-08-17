"""Sinks: pipeline blocks reach the indexers in the shape they expect."""

from unittest.mock import MagicMock

import pyarrow as pa

from pipelines.ingestion.indexing.sinks import QdrantSink, _to_columns


def test_a_pyarrow_block_becomes_columns():
    """The pipeline hands over tables; the indexers speak dictionaries."""
    block = pa.table({"text": ["one", "two"], "vector": [[0.1], [0.2]]})

    columns = _to_columns(block)

    assert columns == {"text": ["one", "two"], "vector": [[0.1], [0.2]]}


def test_no_connection_is_opened_at_construction():
    """The sink is serialised to workers, and a live client cannot be."""
    sink = QdrantSink()

    assert sink._indexer is None


def test_every_block_is_written():
    """A task may receive several blocks, and each must land."""
    sink = QdrantSink()
    sink._indexer = MagicMock()
    blocks = [
        pa.table({"text": ["a"], "vector": [[0.1]], "metadata": [{"chunk_hash": "x"}]}),
        pa.table({"text": ["b"], "vector": [[0.2]], "metadata": [{"chunk_hash": "y"}]}),
    ]

    sink.write(blocks, ctx=None)

    assert sink._indexer.write.call_count == 2


def test_the_written_shape_is_column_oriented():
    """What reaches the indexer is the dictionary shape it was tested with."""
    sink = QdrantSink()
    sink._indexer = MagicMock()

    sink.write([pa.table({"text": ["a"], "vector": [[0.1]]})], ctx=None)

    written = sink._indexer.write.call_args.args[0]
    assert written["text"] == ["a"]
    assert written["vector"] == [[0.1]]
