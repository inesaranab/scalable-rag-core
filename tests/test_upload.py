"""Upload: the API hands out a signed URL; bytes never pass through it."""

import httpx

from services.api.app.config import Settings
from services.api.app.main import app
from services.api.app.routes.ask import enforce_rate_limit


def _authorized():
    return {"sub": "ines"}


async def _post(payload: dict) -> httpx.Response:
    app.state.settings = Settings()
    app.dependency_overrides[enforce_rate_limit] = _authorized
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as client:
        response = await client.post("/upload/presigned-url", json=payload)
    app.dependency_overrides.clear()
    return response


async def test_a_signed_url_is_scoped_to_the_user():
    response = await _post(
        {"filename": "report.txt", "content_type": "text/plain"}
    )

    assert response.status_code == 200
    body = response.json()
    # The blob path embeds the uploader, so users cannot collide.
    assert body["blob_name"].startswith("uploads/ines/")
    # The URL carries a SAS signature and points at the blob.
    assert "sig=" in body["upload_url"]
    assert body["blob_name"].split("/")[-1] in body["upload_url"]


async def test_two_uploads_of_the_same_filename_do_not_collide():
    first = await _post({"filename": "a.txt", "content_type": "text/plain"})
    second = await _post({"filename": "a.txt", "content_type": "text/plain"})

    assert first.json()["blob_name"] != second.json()["blob_name"]


async def test_a_filename_with_reserved_characters_is_encoded():
    """Unencoded ? or # would truncate the URL relative to the signed blob."""
    response = await _post(
        {"filename": "report #2 (final)?.txt", "content_type": "text/plain"}
    )

    body = response.json()
    url = body["upload_url"]
    # The blob part of the URL must not contain raw reserved characters.
    blob_part = url.split("?")[0]
    assert "#" not in blob_part
    assert " " not in blob_part
    assert body["blob_name"].endswith("report #2 (final)?.txt")
