"""The agent: POST /agent/ask, answered by the compiled LangGraph."""

from fastapi import APIRouter, Depends, Request

from services.api.app.agents.graph import final_answer
from services.api.app.routes.ask import AskRequest, enforce_rate_limit

router = APIRouter()


def get_agent(request: Request):
    """Dependency: the compiled agent graph built at startup."""
    return request.app.state.agent


@router.post("/agent/ask")
async def agent_ask(
    body: AskRequest,
    request: Request,
    agent=Depends(get_agent),
    claims: dict = Depends(enforce_rate_limit),
) -> dict:
    """Answer a question by letting the model plan its own retrieval.

    With a session_id, the conversation's past turns are loaded into the
    graph's messages (so the rewriter can resolve references) and both
    new turns are recorded afterwards.
    """
    state = request.app.state
    messages: list[dict] = []
    if body.session_id is not None:
        turns = await state.memory.get_history(
            body.session_id, limit=state.settings.memory_turns
        )
        messages = [{"role": t.role, "content": t.content} for t in turns]
    messages.append({"role": "user", "content": body.question})

    final = await agent.ainvoke({"messages": messages, "documents": []})
    answer = final_answer(final)

    if body.session_id is not None:
        await state.memory.add_message(
            body.session_id, "user", body.question, user_id=claims["sub"]
        )
        await state.memory.add_message(
            body.session_id, "assistant", answer, user_id=claims["sub"]
        )
    return {"answer": answer}
