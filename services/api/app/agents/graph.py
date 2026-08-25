"""The agent itself: a LangGraph state machine wired from injected parts.

The shape: Start -> planner -> (retriever | tool | responder) ->
responder -> End. The planner's one cheap decision keeps greetings from
paying for retrieval, and arithmetic away from the LLM.
"""

from langgraph.graph import END, START, StateGraph

from services.api.app.agents.nodes.planner import make_planner
from services.api.app.agents.nodes.responder import make_responder
from services.api.app.agents.nodes.retriever import make_retriever
from services.api.app.agents.nodes.tool import calculate, make_tool_node
from services.api.app.agents.state import AgentState


def build_agent_graph(llm, embedder, vector_store, graph_store, top_k,
                      rewriter, hyde, tools):
    """Assemble and compile the agent around its injected dependencies.

    Args:
        llm: The model behind the planner and the responder.
        embedder: Turns text into vectors, for the retriever.
        vector_store: Searches chunks by vector similarity.
        graph_store: Looks up entity relationships.
        top_k: How many chunks a retrieval returns.
        rewriter: ``async (question, history) -> str`` standalone query.
        hyde: ``async (question) -> str`` hypothetical document.
        tools: Extra tools for the tool node, name -> async callable.

    Returns:
        The compiled graph; run it with ``await agent.ainvoke(state)``.
    """
    graph = StateGraph(AgentState)
    # The planner is shown the tools the tool node can run, so its menu
    # cannot drift from what exists — minus vector_search, which the
    # "retrieve" route already covers with query rewriting and HyDE in
    # front of it. Offering both would let the model pick the weaker one.
    planner_tools = {
        name: fn for name, fn in tools.items() if name != "vector_search"
    }
    planner_tools["calculator"] = calculate
    graph.add_node("planner", make_planner(llm, tools=planner_tools))
    graph.add_node("retriever", make_retriever(
        embedder, vector_store, graph_store, top_k, rewriter, hyde
    ))
    graph.add_node("tool", make_tool_node(tools))
    graph.add_node("responder", make_responder(llm))

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: state.get("route", "retrieve"),
        {"retrieve": "retriever", "tool": "tool", "respond": "responder"},
    )
    graph.add_edge("retriever", "responder")
    graph.add_edge("tool", "responder")
    graph.add_edge("responder", END)
    return graph.compile()


def final_answer(state: dict) -> str:
    """Extract the assistant's answer from a finished run's state.

    Args:
        state: The dict ``ainvoke`` returned.

    Returns:
        The last message's text — the responder always speaks last.
    """
    return state["messages"][-1]["content"]
