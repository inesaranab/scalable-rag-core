"""The API's ports: the shapes its adapters must fit.

A port declares what the service NEEDS; adapters conform to it, never the
reverse. Protocols are structural: a class satisfies one by having the
methods, with no inheritance or registration.
"""

from typing import Protocol


class Embedder(Protocol):
    """Turns texts into vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """Returns the chunks nearest to a query vector."""

    async def search(self, vector: list[float], k: int) -> list[str]: ...


class LLM(Protocol):
    """Answers a prompt."""

    async def answer(self, prompt: str) -> str: ...
