"""The compiled agent graph: the planner's decision drives the whole path."""

import json

from services.api.app.agents.graph import build_agent_graph, final_answer


class ScriptedLLM:
    """Returns queued replies in order: first the plan, then the answer."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    async def search(self, vector, top_k):
        return ["batching groups requests into one GPU pass"]


class FakeGraphStore:
    async def lookup(self, entity):
        return []


async def identity_rewriter(question, history):
    return question


async def identity_hyde(question):
    return question


def _build(llm) -> object:
    return build_agent_graph(
        llm=llm,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=FakeGraphStore(),
        top_k=4,
        rewriter=identity_rewriter,
        hyde=identity_hyde,
        tools={},
    )


def _start(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}],
            "documents": []}


async def test_a_doc_question_travels_planner_retriever_responder():
    llm = ScriptedLLM([
        json.dumps({"action": "retrieve",
                    "refined_query": "dynamic batching",
                    "reasoning": "needs context"}),
        "Batching groups requests. [Source: corpus]",
    ])
    agent = _build(llm)

    final = await agent.ainvoke(_start("what is batching?"))

    assert final_answer(final) == "Batching groups requests. [Source: corpus]"
    # The retrieved chunk reached the responder's prompt.
    assert "one GPU pass" in llm.prompts[-1]


async def test_a_math_question_travels_planner_tool_responder():
    llm = ScriptedLLM([
        json.dumps({"action": "tool", "tool_choice": "calculator",
                    "tool_input": "21*2", "reasoning": "arithmetic"}),
        "The answer is 42.",
    ])
    agent = _build(llm)

    final = await agent.ainvoke(_start("what is 21 times 2?"))

    assert final_answer(final) == "The answer is 42."
    # The calculator's output reached the responder's prompt.
    assert "42" in llm.prompts[-1]


async def test_a_greeting_skips_retrieval_entirely():
    llm = ScriptedLLM([
        json.dumps({"action": "respond", "reasoning": "just a greeting"}),
        "Hola! Ask me about the corpus.",
    ])
    agent = _build(llm)

    final = await agent.ainvoke(_start("hola!"))

    assert final_answer(final) == "Hola! Ask me about the corpus."
    assert final["documents"] == []  # no search happened


async def test_a_confused_planner_still_produces_an_answer():
    """Garbage planning degrades to retrieval, and the run completes."""
    llm = ScriptedLLM([
        "not json at all",
        "Best effort answer from retrieved context.",
    ])
    agent = _build(llm)

    final = await agent.ainvoke(_start("what is batching?"))

    assert final_answer(final) == "Best effort answer from retrieved context."
