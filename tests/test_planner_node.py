"""The planner: reads the question, decides the route, survives garbage."""

import json

from services.api.app.agents.nodes.planner import make_planner


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _state(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}],
            "documents": []}


async def test_the_prompt_lists_the_tools_it_was_given():
    """The tools' docstrings are the model's only description of them."""

    async def price_lookup(query: str) -> list[str]:
        """Look up the current price of a component."""
        return []

    llm = ScriptedLLM(json.dumps({"action": "respond", "reasoning": "hi"}))
    planner = make_planner(llm, tools={"price_lookup": price_lookup})

    await planner(_state("hola"))

    [prompt] = llm.prompts
    assert "price_lookup" in prompt
    assert "Look up the current price of a component." in prompt


async def test_a_planner_without_tools_offers_no_tool_action():
    llm = ScriptedLLM(json.dumps({"action": "respond", "reasoning": "hi"}))
    planner = make_planner(llm, tools={})

    await planner(_state("hola"))

    [prompt] = llm.prompts
    assert '"action": "tool"' not in prompt


async def test_a_search_question_routes_to_retrieve():
    llm = ScriptedLLM(json.dumps({
        "action": "retrieve",
        "refined_query": "what is dynamic batching",
        "reasoning": "needs documents",
    }))
    planner = make_planner(llm)

    update = await planner(_state("what's that batching thing?"))

    assert update["route"] == "retrieve"
    assert update["current_query"] == "what is dynamic batching"
    assert update["plan"] == ["needs documents"]


async def test_a_math_question_routes_to_tool():
    llm = ScriptedLLM(json.dumps({
        "action": "tool",
        "tool_choice": "calculator",
        "tool_input": "21*2",
        "reasoning": "arithmetic",
    }))
    planner = make_planner(llm)

    update = await planner(_state("what is 21 times 2?"))

    assert update["route"] == "tool"
    assert update["tool_choice"] == "calculator"
    assert update["tool_input"] == "21*2"


async def test_a_greeting_routes_straight_to_respond():
    llm = ScriptedLLM(json.dumps({
        "action": "respond", "reasoning": "no retrieval needed",
    }))
    planner = make_planner(llm)

    update = await planner(_state("hola!"))

    assert update["route"] == "respond"


async def test_a_tool_action_without_a_tool_falls_back_to_retrieve():
    """A tool call naming no tool is a doomed dispatch; searching beats it."""
    llm = ScriptedLLM(json.dumps({
        "action": "tool", "reasoning": "wants a tool, forgot which",
    }))
    planner = make_planner(llm)

    update = await planner(_state("what is 21 times 2?"))

    assert update["route"] == "retrieve"
    assert "tool_choice" not in update


async def test_garbage_output_falls_back_to_retrieve():
    """A confused model must degrade into a search, never an exception."""
    planner = make_planner(ScriptedLLM("I am not JSON at all"))

    update = await planner(_state("what is qdrant?"))

    assert update["route"] == "retrieve"
    assert update["current_query"] == "what is qdrant?"  # unchanged


async def test_an_llm_crash_falls_back_to_retrieve():
    class BrokenLLM:
        async def answer(self, prompt: str) -> str:
            raise ConnectionError("model down")

    planner = make_planner(BrokenLLM())

    update = await planner(_state("what is qdrant?"))

    assert update["route"] == "retrieve"
