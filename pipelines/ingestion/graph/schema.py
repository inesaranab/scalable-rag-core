"""The vocabulary a knowledge graph is allowed to use.

Asked to extract relationships without constraint, a model invents its own
labels — ``WORKS_FOR``, ``works_at`` and ``EMPLOYED_BY`` for one idea. The
graph then cannot be queried, because a query must name the relationship it
is looking for and no name is reliable.
"""

from typing import Literal

VALID_NODE_LABELS = Literal["Person", "Organization", "Location", "Concept", "Product"]

VALID_RELATION_TYPES = Literal[
    "WORKS_FOR", "LOCATED_IN", "RELATES_TO", "PART_OF", "MENTIONS"
]


class GraphSchema:
    """The declared vocabulary, and the instruction that communicates it."""

    @staticmethod
    def get_system_prompt() -> str:
        """Build the instruction constraining extraction to this vocabulary.

        Returns:
            An instruction naming every permitted node label and relationship
            type.
        """
        return (
            f"Extract nodes/edges. "
            f"Allowed Labels: {VALID_NODE_LABELS.__args__}. "
            f"Allowed Relations: {VALID_RELATION_TYPES.__args__}."
        )
