"""Uploading a directory of documents to blob storage.

Uploads are network-bound rather than processor-bound, so many run at once:
each thread spends its time waiting, and waiting in parallel costs nothing.

Usage:

    uv run python scripts/bulk_upload.py data/text --container documents
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 10


def upload_directory(
    directory: Path,
    account_url: str,
    container: str,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Upload every file in a directory, overwriting what is already there.

    Args:
        directory: Directory whose files are uploaded. Not searched
            recursively.
        account_url: Blob storage endpoint to upload to.
        container: Container the files are placed in.
        workers: How many uploads run at once.

    Returns:
        How many files were uploaded.

    Raises:
        ValueError: If the directory does not exist.
    """
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")

    service = BlobServiceClient(account_url, credential=DefaultAzureCredential())
    container_client = service.get_container_client(container)

    files = [path for path in directory.iterdir() if path.is_file()]

    def upload(path: Path) -> None:
        with path.open("rb") as handle:
            container_client.upload_blob(name=path.name, data=handle, overwrite=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(upload, files))

    logger.info("uploaded %d files to %s", len(files), container)
    return len(files)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Upload documents to blob storage.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--account-url", required=True)
    parser.add_argument("--container", default="documents")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    try:
        upload_directory(
            args.directory, args.account_url, args.container, args.workers
        )
    except ValueError as exc:
        sys.exit(str(exc))
