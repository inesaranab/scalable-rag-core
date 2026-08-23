"""The responder node: turns gathered evidence into the final answer."""

_ANSWER_PROMPT = """You are a helpful assistant. Use the context below to
answer the question.

Context:
{context}

Question:
{question}

Instructions:
1. Cite sources using [Source: ...] when the context provides them.
2. If the answer is not in the context and the question needs one, say
   "I don't have that information."
3. Be concise."""


def make_responder(llm):
    """Build the responder node around an injected LLM.

    Args:
        llm: Anything with ``async answer(prompt) -> str``.

    Returns:
        An async node: state -> {"messages": [assistant turn]} — appended
        to the conversation by the state's reducer.
    """

    async def responder(state: dict) -> dict:
        question = state.get("current_query") or state["messages"][-1]["content"]
        context = "\n\n".join(state.get("documents", [])) or "(no context)"
        answer = await llm.answer(
            _ANSWER_PROMPT.format(context=context, question=question)
        )
        return {"messages": [{"role": "assistant", "content": answer}]}

    return responder
