"""The tool node: runs the tool the planner chose, safely.

LLMs are bad at arithmetic, so a calculator does it instead — built on
the ast module, which parses the expression into typed nodes and refuses
anything that is not pure arithmetic. The dangerous builtin that runs
arbitrary strings as code is deliberately absent: a model-supplied
expression must never be executable.
"""

import ast
import logging
import operator as op

logger = logging.getLogger(__name__)

_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg, ast.UAdd: op.pos,
}

# Exponentiation is the one operator whose cost explodes with tiny input:
# 9**9**9 is seven characters and computes a number with 370 million
# digits, blocking the worker that runs it. Length caps cannot catch it,
# so the exponent and the base are bounded directly.
MAX_EXPONENT = 1_000
MAX_POWER_BASE = 1_000_000


def calculate(expression: str) -> str:
    """Evaluate a pure-arithmetic expression.

    Args:
        expression: Numbers and + - * / ** % with parentheses, nothing
            else.

    Returns:
        The result as a string, or an error message for anything that is
        not arithmetic — including code smuggled into the expression.

    How the tree looks for "2 + 3 * 4" (grammar gives * precedence, so
    it becomes the deeper node; _walk computes bottom-up -> 14)::

                BinOp(+)
               /        \\
         Constant(2)   BinOp(*)
                      /        \\
                Constant(3)  Constant(4)
    """

    # Cap length first: parsing itself can be abused (deep nesting,
    # memory) and a legitimate expression is never this long.
    if len(expression) > 100:
        return "calculator error: expression too long"

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = _walk(node.left), _walk(node.right)
            if isinstance(node.op, ast.Pow) and (
                abs(right) > MAX_EXPONENT or abs(left) > MAX_POWER_BASE
            ):
                raise ValueError(
                    f"power too large: bases are capped at {MAX_POWER_BASE} "
                    f"and exponents at {MAX_EXPONENT}"
                )
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_walk(node.operand))
        raise ValueError(f"not arithmetic: {ast.dump(node)[:50]}")

    try:
        result = _walk(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as error:
        return f"calculator error: {error}"


def make_tool_node(tools: dict):
    """Build the tool node around the injected tool registry.

    Args:
        tools: Tool name -> ``async (input) -> list[str]``. The
            calculator is always available on top of these.

    Returns:
        An async node: state -> {"documents": [tool output]} — the
        responder reads tool results as evidence like any other.
    """

    async def tool_node(state: dict) -> dict:
        name = state.get("tool_choice", "")
        argument = state.get("tool_input", "")

        if name == "calculator":
            result = calculate(argument)
        elif name in tools:
            result = " | ".join(await tools[name](argument))
        else:
            logger.warning("unknown tool requested", extra={"tool": name})
            result = "unknown tool"

        return {"documents": [f"Tool {name}: {result}"]}

    return tool_node
