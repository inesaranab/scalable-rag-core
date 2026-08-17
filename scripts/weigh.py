"""Report what a dependency would cost in disk, without installing it.

Resolving a requirement names its packages but never their size, so a package
that quietly brings a machine-learning framework with it looks the same as one
that brings nothing. This resolves the requirement, asks the index how large
each resulting file is, and reports the total before anything is downloaded.

The figure is the compressed download. Installed size is typically two to
three times larger.

Usage:

    uv run --with requests python scripts/weigh.py "unstructured[pdf]"
"""

import concurrent.futures
import subprocess
import sys

import requests

INDEX = "https://pypi.org/pypi"
TIMEOUT_S = 10
WORKERS = 20
SHOW_LARGEST = 10


def resolve(requirement: str) -> list[tuple[str, str]]:
    """Resolve a requirement to the exact packages it would install.

    Args:
        requirement: A requirement specifier, such as ``unstructured[pdf]``.

    Returns:
        Every resolved package as a name and version pair.

    Raises:
        SystemExit: If the requirement cannot be resolved.
    """
    result = subprocess.run(
        ["uv", "pip", "compile", "-", "--quiet"],
        input=requirement,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"could not resolve {requirement!r}:\n{result.stderr}")

    packages = []
    for line in result.stdout.splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            packages.append((name.strip(), version.strip()))
    return packages


def download_size(package: tuple[str, str]) -> tuple[str, int]:
    """Ask the index how large a package's download is.

    Args:
        package: The package's name and version.

    Returns:
        The name, and the size in bytes of its wheel where one exists,
        otherwise of its source archive. Zero when the index has no answer.
    """
    name, version = package
    try:
        response = requests.get(f"{INDEX}/{name}/{version}/json", timeout=TIMEOUT_S)
        files = response.json().get("urls", [])
    except Exception:
        return name, 0

    wheels = [f for f in files if f["packagetype"] == "bdist_wheel"]
    chosen = wheels[0] if wheels else (files[0] if files else None)
    return name, chosen["size"] if chosen else 0


def main() -> None:
    """Print the download total for the requirement named on the command line.

    Raises:
        SystemExit: If no requirement was given.
    """
    if len(sys.argv) < 2:
        sys.exit('usage: weigh.py "package[extra]"')

    requirement = sys.argv[1]
    packages = resolve(requirement)

    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        sizes = list(pool.map(download_size, packages))

    sizes.sort(key=lambda item: -item[1])
    total = sum(size for _, size in sizes)
    unknown = sum(1 for _, size in sizes if size == 0)

    print(f"\n{requirement}")
    print(f"  {len(sizes)} packages, {total / 1e9:.2f} GB to download")
    print(f"  roughly {total * 2.5 / 1e9:.1f} GB once installed")
    if unknown:
        print(f"  ({unknown} packages had no size reported)")

    print("\n  largest:")
    for name, size in sizes[:SHOW_LARGEST]:
        if size:
            print(f"    {size / 1e6:8.1f} MB  {name}")


if __name__ == "__main__":
    main()
