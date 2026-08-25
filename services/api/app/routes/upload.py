"""Upload: hand out a signed URL so files go straight to Blob Storage.

Large files never travel through the API server: the client receives a
SAS URL (Shared Access Signature — a link carrying its own time-limited
permission) and PUTs the bytes directly to Azure. The API only does the
cheap part: naming the blob and signing the link.
"""

import uuid
from datetime import UTC, datetime, timedelta

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from services.api.app.routes.ask import enforce_rate_limit

router = APIRouter()


# signed in adavance = signature (cryptographic) with a key
# before the client acts, embeds the signature in the URL
class PresignRequest(BaseModel):
    """The request body for one upload slot.

    Attributes:
        filename: The client's name for the file; kept as a suffix for
            humans, never trusted for uniqueness.
        content_type: The MIME type the client will send.
    """

    filename: str
    content_type: str


@router.post("/upload/presigned-url")
async def presigned_url(
    body: PresignRequest,
    request: Request,
    claims: dict = Depends(enforce_rate_limit),
) -> dict:
    """Return a one-hour signed URL for a direct upload.

    The blob path embeds the uploader and a fresh UUID, so users cannot
    collide with each other or with themselves.
    """
    settings = request.app.state.settings
    service = BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )
    # uuid5 = re-processing the same thing should replace the old record
    # uuid4 = when each event is genuinely new
    blob_name = f"uploads/{claims['sub']}/{uuid.uuid4()}-{body.filename}"
    # shared access signature
    sas = generate_blob_sas(
        account_name=service.account_name,
        container_name=settings.azure_storage_container,
        blob_name=blob_name,
        account_key=service.credential.account_key,
        permission=BlobSasPermissions(write=True, create=True),
        expiry=datetime.now(UTC) + timedelta(hours=1),
    )
    upload_url = f"{service.url}{settings.azure_storage_container}/{blob_name}?{sas}"
    return {"upload_url": upload_url, "blob_name": blob_name}
