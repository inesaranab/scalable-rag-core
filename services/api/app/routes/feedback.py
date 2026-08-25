"""Feedback: POST /feedback records a user's verdict on an answer."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from services.api.app.routes.ask import enforce_rate_limit

router = APIRouter()


class FeedbackRequest(BaseModel):
    """The request body for one verdict.

    Attributes:
        session_id: The conversation being judged.
        score: -1 (bad), 0 (neutral) or 1 (good).
        comment: Optional free-text explanation.
        message_id: Optional turn identifier, when the client tracks
            turns.
    """

    session_id: str
    score: int = Field(ge=-1, le=1)
    comment: str | None = None
    message_id: str | None = None


@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
    claims: dict = Depends(enforce_rate_limit),
) -> dict:
    """Record the caller's verdict. The author comes from the JWT."""
    await request.app.state.feedback.add(
        session_id=body.session_id,
        user_id=claims["sub"],
        score=body.score,
        comment=body.comment,
        message_id=body.message_id,
    )
    return {"status": "recorded"}
