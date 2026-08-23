"""The read path: a question is embedded, matched against the store, and
answered by the LLM with the retrieved chunks as context."""

from services.api.app.ports import LLM, Embedder, VectorStore


class RetrievalService:
    """Answers questions from the indexed corpus.

    Attributes:
        embedder: Turns the question into a query vector.
        store: Finds the chunks nearest to that vector.
        llm: Composes the answer from question plus retrieved chunks.
        top_k: How many chunks are retrieved as context.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k

    async def answer(
        self, question: str, history: list[str] | None = None
    ) -> str:
        """Answer a question grounded in retrieved chunks.

        Args:
            question: The user's question, in natural language.
            history: Earlier turns of the conversation, oldest first,
                each already formatted as "role: content". None or empty
                means a fresh conversation.

        Returns:
            The LLM's answer, composed from the retrieved context.
        """
        [vector] = await self.embedder.embed([question])
        chunks = await self.store.search(vector, self.top_k)
        prompt = self._compose(question, chunks, history or [])
        return await self.llm.answer(prompt)

    @staticmethod
    def _compose(
        question: str, chunks: list[str], history: list[str]
    ) -> str:
        """Build the prompt: conversation, then evidence, then question."""
        context = "\n\n".join(chunks)
        conversation = (
            "Conversation so far:\n" + "\n".join(history) + "\n\n"
            if history
            else ""
        )
        return (
            f"{conversation}"
            "Answer the question using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
