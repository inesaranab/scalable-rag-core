"""The agent's state: the one dict every node reads and partially updates.

LangGraph merges each node's returned dict into this state. Fields
annotated with a reducer accumulate across nodes; plain fields are
replaced by the last node that writes them.
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """What the graph knows at any point of a run.

    Attributes:
        messages: The conversation, appended to by every node that
            speaks (operator.add reducer — never overwritten).
        documents: Evidence for the responder; replaced per retrieval.
        current_query: The question being worked on, possibly refined
            by the planner or rewriter.
        plan: The planner's reasoning trail, appended per decision.
        route: The planner's chosen path: "retrieve", "tool" or
            "respond".
        tool_choice: Which tool the tool node should run.
        tool_input: The argument for that tool.
    """

    messages: Annotated[list[dict], operator.add]  # operator.add(a,b) == a+b -> reducer
    documents: list[
        str
    ]  # if two nodes run in parallel to update this field langgraph raises
    current_query: str
    plan: Annotated[list[str], operator.add]
    route: str
    tool_choice: str
    tool_input: str
