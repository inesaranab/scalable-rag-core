"""Streaming chat: the agent's progress arrives while it thinks.

NDJSON (one JSON object per line) instead of one final body: the client
renders "planning… retrieving…" live, then the answer. Memory and the
semantic cache are written after the stream completes, so a broken
connection mid-stream records nothing half-done.
"""

import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.api.app.routes.ask import enforce_rate_limit

router = APIRouter()


class ChatStreamRequest(BaseModel):
    """The request body for one streamed chat turn.

    Attributes:
        message: The user's message.
        session_id: Which conversation this belongs to; None starts a
            new one (a fresh id is created and used for recording).
    """

    message: str = Field(min_length=1)
    session_id: str | None = None


def get_agent_for_chat(request: Request):
    """Dependency: the compiled agent graph built at startup."""
    return request.app.state.agent


@router.post("/chat/stream")
async def chat_stream(
    body: ChatStreamRequest,
    request: Request,
    agent=Depends(get_agent_for_chat),
    claims: dict = Depends(enforce_rate_limit),
) -> StreamingResponse:
    """Answer a chat turn, streaming node progress as NDJSON lines."""
    state = request.app.state
    session_id = body.session_id or str(uuid.uuid4())

    cached = await state.semantic_cache.get(body.message)
    if cached is not None:
        async def stream_cached() -> AsyncGenerator[str, None]:
            yield json.dumps({"type": "answer", "content": cached}) + "\n"

        return StreamingResponse(
            stream_cached(), media_type="application/x-ndjson"
        )

    turns = await state.memory.get_history(
        session_id, limit=state.settings.memory_turns
    )
    messages = [{"role": t.role, "content": t.content} for t in turns]
    messages.append({"role": "user", "content": body.message})

    async def events() -> AsyncGenerator[str, None]:
        answer = ""
        async for event in agent.astream(
            {"messages": messages, "documents": []}
        ):
            node_name = next(iter(event))
            yield json.dumps({"type": "status", "node": node_name}) + "\n"
            node_update = event[node_name]
            if node_name == "responder" and "messages" in node_update:
                answer = node_update["messages"][-1]["content"]
                yield json.dumps({"type": "answer", "content": answer}) + "\n"

        # After the stream: record only completed turns.
        if answer:
            await state.memory.add_message(
                session_id, "user", body.message, user_id=claims["sub"]
            )
            await state.memory.add_message(
                session_id, "assistant", answer, user_id=claims["sub"]
            )
            await state.semantic_cache.set(body.message, answer)

    return StreamingResponse(events(), media_type="application/x-ndjson")
