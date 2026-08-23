"""The agent's state: messages accumulate across nodes, documents replace."""

from langgraph.graph import END, START, StateGraph

from services.api.app.agents.state import AgentState


async def test_messages_append_and_documents_replace():
    async def first(state: AgentState) -> dict:
        return {"messages": [{"role": "user", "content": "a"}],
                "documents": ["old"]}

    async def second(state: AgentState) -> dict:
        return {"messages": [{"role": "assistant", "content": "b"}],
                "documents": ["new"]}

    g = StateGraph(AgentState)
    g.add_node("first", first)
    g.add_node("second", second)
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)

    final = await g.compile().ainvoke({"messages": [], "documents": []})

    # Two nodes each returned one message: the reducer appended both.
    assert [m["content"] for m in final["messages"]] == ["a", "b"]
    # documents has no reducer: the last write wins.
    assert final["documents"] == ["new"]
