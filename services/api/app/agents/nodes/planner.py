"""The planner node: one cheap LLM call that decides the path.

This is the gate that stops a greeting from paying for a vector search.
The model answers with a small JSON decision; anything unparseable
degrades into "retrieve", because searching too much is recoverable and
answering without evidence is not.
"""

import json
import logging

logger = logging.getLogger(__name__)

_PLAN_PROMPT = """You are the planner of a retrieval agent. Read the user's
question and decide ONE action. Reply with ONLY a JSON object:

  {{"action": "retrieve", "refined_query": "<standalone search query>",
    "reasoning": "<one line>"}}
      when the question needs documents from the knowledge base.

  {{"action": "tool", "tool_choice": "calculator",
    "tool_input": "<expression>", "reasoning": "<one line>"}}
      when the question is arithmetic or needs a tool.

  {{"action": "respond", "reasoning": "<one line>"}}
      for greetings, thanks, or questions needing no evidence.

Question: {question}"""


# factory function: builds and returns a configured function
def make_planner(llm):
    """Build the planner node around an injected LLM.

    Args:
        llm: Anything with ``async answer(prompt) -> str``.

    Returns:
        An async node: state -> partial state with route, current_query,
        plan, and tool fields when applicable.
    """

    async def planner(state: dict) -> dict:
        question = state["messages"][-1]["content"]
        try:
            reply = await llm.answer(_PLAN_PROMPT.format(question=question))
            start, end = reply.index("{"), reply.rindex("}") + 1
            decision = json.loads(reply[start:end])
            action = decision["action"]
        except Exception:
            # A confused or dead planner defaults to searching.
            logger.warning("planner fell back to retrieve", exc_info=True)
            return {
                "route": "retrieve",  # assymetric failure; fail toward cheap, visible, error
                # answer without evidence: lie the user may not detect, searching for a greeting: visible failure
                "current_query": question,
                "plan": ["planner failed; defaulting to retrieval"],
            }

        if action == "tool" and not decision.get("tool_choice"):
            # A tool call naming no tool cannot be dispatched; searching
            # is the recoverable interpretation.
            action = "retrieve"
        update = {
            "route": action
            if action in ("retrieve", "tool", "respond")
            else "retrieve",
            "current_query": decision.get("refined_query", question),
            "plan": [str(decision.get("reasoning", ""))],
        }
        if action == "tool":
            update["tool_choice"] = str(decision["tool_choice"])
            update["tool_input"] = str(decision.get("tool_input", ""))
        logger.info("plan decided", extra={"route": update["route"]})
        return update

    return planner
