"""Extracting text from HTML bytes.

Scripts, styles and metadata carry no meaning for retrieval, and embedding
them puts CSS and JavaScript into the vector store where they compete with
real content. They are removed before the text is taken.
"""

from typing import Any

from bs4 import BeautifulSoup
from bs4.element import ResultSet, Tag

# Elements whose contents are markup or code rather than readable text.
_NOISE_TAGS = ["script", "style", "meta"]


def parse_html_bytes(file_bytes: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    """Extract readable text from an HTML document held in memory.

    Args:
        file_bytes: The document's raw bytes.
        filename: Name recorded in the metadata.

    Returns:
        The text of the remaining elements separated by newlines, and metadata
        carrying the filename and the type ``html``.
    """

    # Navigable html tree
    soup: BeautifulSoup = BeautifulSoup(file_bytes, "html.parser")

    # finding all tags that contain text but are not useful
    noise: ResultSet[Tag] = soup(_NOISE_TAGS)

    # eliminate those tags
    element: Tag
    for element in noise:
        element.decompose()

    # get the text from the navigable tree
    text: str = soup.get_text(separator="\n")

    return text, {"filename": filename, "type": "html"}
