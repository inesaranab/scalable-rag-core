"""The tool node: the calculator computes safely, unknown tools degrade."""

import pathlib

from services.api.app.agents.nodes.tool import calculate, make_tool_node


async def fake_graph_lookup(entity: str) -> list[str]:
    return [f"{entity} -[RUNS_ON]-> Kubernetes"]


def test_the_calculator_does_arithmetic():
    assert calculate("21 * 2") == "42"
    assert calculate("(1 + 2) ** 3 / 9") == "3.0"


def test_an_oversized_expression_is_refused_before_parsing():
    """Length is capped before the parser ever runs."""
    result = calculate("1+" * 200 + "1")

    assert "error" in result.lower() and "long" in result.lower()


def test_the_calculator_refuses_anything_but_arithmetic():
    """Code injection through the expression must come back as an error."""
    result = calculate("__import__('os').system('rm -rf /')")

    assert "error" in result.lower()


def test_the_module_never_uses_eval():
    source = pathlib.Path(
        "services/api/app/agents/nodes/tool.py"
    ).read_text()

    assert "eval(" not in source


async def test_the_tool_node_runs_the_chosen_tool():
    node = make_tool_node({"graph_lookup": fake_graph_lookup})

    update = await node({
        "tool_choice": "graph_lookup", "tool_input": "Qdrant",
        "messages": [], "documents": [],
    })

    assert update["documents"] == [
        "Tool graph_lookup: Qdrant -[RUNS_ON]-> Kubernetes"
    ]


async def test_the_calculator_is_available_without_registration():
    node = make_tool_node({})

    update = await node({
        "tool_choice": "calculator", "tool_input": "6*7",
        "messages": [], "documents": [],
    })

    assert update["documents"] == ["Tool calculator: 42"]


async def test_an_unknown_tool_degrades_into_a_note():
    node = make_tool_node({})

    update = await node({
        "tool_choice": "time_machine", "tool_input": "1985",
        "messages": [], "documents": [],
    })

    [note] = update["documents"]
    assert "time_machine" in note and "unknown" in note.lower()


def test_an_exponent_bomb_is_refused_not_computed():
    """9**9**9 is seven characters and would block the event loop."""
    import time

    start = time.monotonic()
    result = calculate("9**9**9")
    elapsed = time.monotonic() - start

    assert "error" in result.lower()
    assert elapsed < 1.0


def test_ordinary_powers_still_work():
    assert calculate("2**10") == "1024"
