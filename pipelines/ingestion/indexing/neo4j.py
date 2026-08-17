"""Writing extracted entities and relationships to the graph store.

Every write is a MERGE rather than a CREATE: it matches an existing node or
relationship and creates one only when none is found. Ingestion can therefore
run twice over the same documents without producing a second copy of the
graph, which matters because a re-run is the normal response to a partial
failure.
"""

import os
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction

# From the environment, for the same reason as the vector store address.
DEFAULT_URI = os.environ.get("NEO4J_URI", "bolt://neo4j-cluster:7687")
DEFAULT_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASSWORD", "password"),
)

# Node labels and relationship types cannot be parameterised in Cypher, so
# they are interpolated — and must therefore come from the declared schema
# rather than from model output.
_MERGE_NODE = "MERGE (n:{label} {{name: $name}})"

_MERGE_EDGE = (
    "MATCH (a {{name: $from_name}}), (b {{name: $to_name}}) MERGE (a)-[:{type}]->(b)"
)


class Neo4jIndexer:
    """Writes graph data with idempotent MERGE statements.

    Attributes:
        driver: Connection pool to the graph store, held open across batches.
    """

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        auth: tuple[str, str] = DEFAULT_AUTH,
    ) -> None:
        """Open a connection pool to the graph store.

        Args:
            uri: Bolt address of the store. Resolves to internal cluster DNS
                when deployed.
            auth: Username and password.
        """
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def write(self, batch: dict[str, Any]) -> None:
        """Write one batch of extracted graphs.

        The whole batch is written in a single transaction, so a batch either
        lands completely or not at all.

        Args:
            batch: A batch carrying ``graph_nodes`` and ``graph_edges``, each
                holding one list per chunk.
        """
        nodes = [node for chunk in batch["graph_nodes"] for node in chunk]
        edges = [edge for chunk in batch["graph_edges"] for edge in chunk]

        if not nodes and not edges:
            return

        with self.driver.session() as session:
            # rollback if the transaction fails in the middle
            # opens a transaction, runs the whole function and commits
            # at the end
            session.execute_write(self._merge_graph_data, nodes, edges)

    @staticmethod
    def _merge_graph_data(
        tx: ManagedTransaction,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        """Merge nodes and then relationships within one transaction.

        Nodes are written first, because a relationship can only be created
        between endpoints that already exist.

        Args:
            tx: The transaction the statements run in.
            nodes: Entities, each carrying a ``label`` and a ``name``.
            edges: Relationships, each carrying a ``type``, a ``from`` and a
                ``to``.
        """
        # merge is like upsert, but leaves the first one vs qdrant
        for node in nodes:
            tx.run(
                _MERGE_NODE.format(label=node["label"]),
                name=node["name"],
            )

        for edge in edges:
            tx.run(
                _MERGE_EDGE.format(type=edge["type"]),
                from_name=edge["from"],
                to_name=edge["to"],
            )

    def close(self) -> None:
        """Release the connection pool."""
        self.driver.close()
