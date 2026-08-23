"""Graph lookups over Neo4j: an entity's connections, as fact strings."""


class Neo4jGraphStore:
    """Reads an entity's relationships from the knowledge graph.

    Attributes:
        driver: The async Neo4j driver (or a stand-in with the same shape).
    """

    def __init__(self, driver) -> None:
        self.driver = driver

    async def lookup(self, entity: str, limit: int = 10) -> list[str]:
        """Return the entity's connections as readable facts.

        Args:
            entity: Name of the node to look up.
            limit: Maximum number of facts returned.

        Returns:
            Strings of the form ``source -[RELATION]-> target``, one per
            relationship touching the entity. Empty if the entity is
            unknown to the graph.
        """
        # The entity travels as a bound parameter ($name): never pasted
        # into the query text, so it cannot rewrite the Cypher.
        query = (
            "MATCH (n {name: $name})-[r]->(m) "
            "RETURN n.name AS source, type(r) AS relation, m.name AS target "
            "LIMIT $limit"
        )
        async with self.driver.session() as session:
            result = await session.run(query, name=entity, limit=limit)
            rows = await result.data()
        return [
            f"{row['source']} -[{row['relation']}]-> {row['target']}"
            for row in rows
        ]
