"""The fixed retrieval pipeline: POST /ask, cached, rate limited, remembered."""

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from services.api.app.auth import require_token
from services.api.app.retrieval import RetrievalService

router = APIRouter()


class AskRequest(BaseModel):
    """The request body for a question.

    Attributes:
        question: The user's question, in natural language.
        session_id: Which conversation this belongs to. None means a
            one-off question: no memory read, no memory written.
    """

    question: str = Field(min_length=1)
    session_id: str | None = None


def get_retrieval_service(request: Request) -> RetrievalService:
    """Dependency: the service built at startup. Tests override this."""
    return request.app.state.retrieval_service


async def enforce_rate_limit(
    request: Request, claims: dict = Depends(require_token)
) -> dict:
    """Dependency: count this request against the token's subject.

    Raises:
        HTTPException: 429 once the subject exceeds the per-minute limit.
    """
    settings = request.app.state.settings
    allowed = await request.app.state.redis.check_rate_limit(
        subject=claims["sub"],
        limit=settings.rate_limit_per_minute,
        window_s=60,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return claims


@router.post("/ask")
async def ask(
    body: AskRequest,
    request: Request,
    service: RetrievalService = Depends(get_retrieval_service),
    claims: dict = Depends(enforce_rate_limit),
) -> dict:
    """Answer a question from the indexed corpus. Requires a Bearer JWT.

    Three layers, cheapest first: the exact cache (identical question),
    the semantic cache (same meaning), then the models. With a
    session_id, past turns are shown to the model and both new turns
    are recorded.
    """
    state = request.app.state
    # Hash the question so the key is fixed-length and safe for Redis.
    key = hashlib.sha256(body.question.encode()).hexdigest()
    answer = await state.redis.get_cached(key)
    if answer is None:
        answer = await state.semantic_cache.get(body.question)
    if answer is None:
        history = None
        if body.session_id is not None:
            turns = await state.memory.get_history(
                body.session_id, limit=state.settings.memory_turns
            )
            history = [f"{t.role}: {t.content}" for t in turns]
        answer = await service.answer(body.question, history=history)
        await state.redis.set_cached(
            key, answer, ttl_s=state.settings.cache_ttl_s
        )
        await state.semantic_cache.set(body.question, answer)
    if body.session_id is not None:
        await state.memory.add_message(
            body.session_id, "user", body.question, user_id=claims["sub"]
        )
        await state.memory.add_message(
            body.session_id, "assistant", answer, user_id=claims["sub"]
        )
    return {"answer": answer}
