"""HyDE — Hypothetical Document Embeddings.

Questions and documents are shaped differently, so their embeddings sit
apart. HyDE asks the model to write a FAKE answer paragraph and embeds
that instead: the fake answer shares vocabulary and shape with real
documents, so cosine similarity finds them. The fake text is never shown
to anyone — only its embedding is used.
"""

import logging

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """Write one short hypothetical paragraph that answers the
question below. It does not need to be factually correct, but it must use
the vocabulary a relevant document would use. Output only the paragraph.

Question: {question}"""


def make_hyde(llm):
    """Build the HyDE transform around an injected LLM.

    Args:
        llm: Anything with ``async answer(prompt) -> str``.

    Returns:
        ``async (question) -> str``: the hypothetical document, or the
        original question when generation fails.
    """

    async def hyde(question: str) -> str:
        try:
            return await llm.answer(_HYDE_PROMPT.format(question=question))
        except Exception:
            logger.warning("hyde fell back to the question", exc_info=True)
            return question

    return hyde
