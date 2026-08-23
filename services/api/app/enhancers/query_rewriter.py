"""The query rewriter: makes a follow-up question stand alone.

"How much does it cost?" is unsearchable — "it" lives in the previous
turn. The rewriter resolves such references from the conversation so the
retriever gets a complete query.
"""

import logging

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """Rewrite the latest question as a standalone search
query, resolving references (it, he, she, they, that) from the history.
Output ONLY the rewritten question; if none is needed, output the
question unchanged.

History:
{history}

Latest question: {question}"""


def make_rewriter(llm):
    """Build the rewriter around an injected LLM.

    Args:
        llm: Anything with ``async answer(prompt) -> str``.

    Returns:
        ``async (question, history) -> str``: the standalone query. With
        no history there is nothing to resolve, so the question returns
        untouched and no model is called; failures also fall back to the
        original question.
    """

    async def rewrite(question: str, history: list[dict]) -> str:
        if not history:
            return question
        lines = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        try:
            rewritten = await llm.answer(
                _REWRITE_PROMPT.format(history=lines, question=question)
            )
            return rewritten.strip()
        except Exception:
            logger.warning("rewriter fell back to the question", exc_info=True)
            return question

    return rewrite
