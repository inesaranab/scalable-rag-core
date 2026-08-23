"""The ingestion pipeline: read, parse, then embed and extract in parallel.

Stages are declared rather than executed. Nothing reads a file until a write
is requested, at which point the whole graph runs — so the framework can size
each stage independently and start a later stage on early results while
earlier files are still being parsed.

Parsing forks into two branches that share its output. Embedding needs a GPU;
graph extraction needs outbound model calls. Pairing them would tie the scale
of each to the other.
"""

import argparse
import logging

import ray

from pipelines.ingestion.embedding.compute import BatchEmbedder
from pipelines.ingestion.graph.extractor import GraphExtractor
from pipelines.ingestion.indexing.sinks import Neo4jSink, QdrantSink
from pipelines.ingestion.processing import process_batch

logger = logging.getLogger(__name__)

# Files per parsing task. Small, because a task holds every file it is given
# in memory at once.
PARSE_BATCH_SIZE = 10

# Chunks per embedding request. Large, because the cost is dominated by the
# round trip rather than by the text.
EMBED_BATCH_SIZE = 100

# Chunks per extraction task. Small, because each chunk is a separate model
# call and a large batch would hold the task open for minutes.
EXTRACT_BATCH_SIZE = 5

EMBED_CONCURRENCY = 5
EXTRACT_CONCURRENCY = 10

# A fraction, because the model itself is served elsewhere. This reserves a
# share of a device for the client rather than a whole one.
EMBED_GPU_FRACTION = 0.2


def main(container: str, prefix: str = "") -> None:
    """Run ingestion over every document under a prefix.

    The storage account is read from the environment, being fixed for a
    deployment rather than chosen per run.

    Args:
        container: Container holding the documents.
        prefix: Restricts ingestion to paths beginning with this, so a run can
            cover one upload rather than the whole corpus.
    """
    source = f"az://{container}/{prefix}"
    logger.info("reading from %s", source)

    # A Dataset is not generic, so the row shape each stage produces is
    # recorded here instead. One row per file:
    #   {bytes: bytes, path: str}
    documents: ray.data.Dataset = ray.data.read_binary_files(source, include_paths=True)

    # One row per chunk: {text: str, metadata: dict} — metadata carries
    # filename, type, chunk_index, chunk_hash and ingested_at.
    chunks: ray.data.Dataset = documents.map_batches(
        process_batch,
        batch_size=PARSE_BATCH_SIZE,
        num_cpus=1,
    )

    # checkpoint the chunks into memory (cluster RAM);
    # why: two branches read that copy
    chunks: ray.data.Dataset = chunks.materialize()

    # Chunk rows plus {vector: list[float]}.
    vectors: ray.data.Dataset = chunks.map_batches(
        BatchEmbedder,
        concurrency=EMBED_CONCURRENCY,
        num_gpus=EMBED_GPU_FRACTION,
        batch_size=EMBED_BATCH_SIZE,
    )

    # Chunk rows plus {graph_nodes: list, graph_edges: list}, one list each
    # per chunk.
    graphs: ray.data.Dataset = chunks.map_batches(
        GraphExtractor,
        concurrency=EXTRACT_CONCURRENCY,
        batch_size=EXTRACT_BATCH_SIZE,
    )

    # Sink: where a datapipeline ends
    # 1. QdrantSink() → __init__ runs, on your laptop. Ordinary object.
    # 2. write_datasink(sink) → now Ray pickles it.
    # 3. Workers unpickle their copies.
    # 4. write() runs on each copy — connection opens here.
    graphs.write_datasink(Neo4jSink())
    vectors.write_datasink(QdrantSink())

    logger.info("ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest documents into the index.")
    parser.add_argument("--container", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    main(args.container, args.prefix)
