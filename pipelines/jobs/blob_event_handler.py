"""Submitting an ingestion job when a document is uploaded.

Ingestion is not triggered by hand. Storage raises an event on upload, and
this submits a job covering only what arrived, so the cluster does the work
and the handler returns immediately.

The handler must stay small: it holds no state, parses no documents, and its
only job is to hand a path to the cluster.
"""

import logging
import os
from typing import Any

from ray.job_submission import JobSubmissionClient

logger = logging.getLogger(__name__)

RAY_DASHBOARD_URL = os.environ.get(
    "RAY_DASHBOARD_URL", "http://rag-ray-cluster-head-svc:8265"
)


def handle_blob_event(event: dict[str, Any]) -> str:
    """Submit an ingestion job for a newly uploaded document.

    Args:
        event: The storage event, whose ``subject`` names the container and
            path of the uploaded blob.

    Returns:
        The submitted job's identifier, so the run can be followed.

    Raises:
        ValueError: If the event names no blob path.
    """
    subject = event.get("subject", "")
    container, _, blob_path = _parse_subject(subject)

    client = JobSubmissionClient(RAY_DASHBOARD_URL)

    job_id = client.submit_job(
        entrypoint=(
            f"python pipelines/ingestion/main.py "
            f"--container {container} --prefix {blob_path}"
        ),
        runtime_env={"working_dir": "./"},
    )

    logger.info("submitted job %s for %s/%s", job_id, container, blob_path)
    return job_id


def _parse_subject(subject: str) -> tuple[str, str, str]:
    """Split a storage event subject into its container and blob path.

    Args:
        subject: A subject of the form
            ``/blobServices/default/containers/<container>/blobs/<path>``.

    Returns:
        The container, the literal separator, and the blob path.

    Raises:
        ValueError: If the subject does not name a container and a blob.
    """
    try:
        _, remainder = subject.split("/containers/", 1)
        container, blob_path = remainder.split("/blobs/", 1)
    except ValueError as exc:
        raise ValueError(f"event names no blob: {subject!r}") from exc

    return container, "/blobs/", blob_path
