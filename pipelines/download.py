"""Fetch and verify the raw datasets.

Hillstrom is the primary dataset. If it is missing from ``data/raw/`` this script
downloads it from the canonical MineThatData URL and verifies its SHA-256 against
the pinned digest. The digest is the only thing that distinguishes *the* canonical
file from *a* file with the same shape, so it is checked on every run -- not just
after a download.

Usage::

    uv run python pipelines/download.py              # fetch if missing, then verify
    uv run python pipelines/download.py --verify-only  # never touch the network
    uv run python pipelines/download.py --force      # re-download and verify
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.datasets.hillstrom import (  # noqa: E402
    RAW_PATH,
    RAW_SHA256,
    RAW_SOURCE_URL,
    raw_sha256,
)


def download(dest: Path, url: str = RAW_SOURCE_URL) -> None:
    """Download ``url`` to ``dest``, raising ``RuntimeError`` on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            f"Fetch it manually into {dest}:\n  {url}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="verify an existing file, never download"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file is present"
    )
    args = parser.parse_args(argv)

    downloaded = False
    if args.force or not RAW_PATH.exists():
        if args.verify_only and not RAW_PATH.exists():
            print(f"Hillstrom CSV not found at {RAW_PATH}")
            print(f"Fetch it manually (or drop --verify-only):\n  {RAW_SOURCE_URL}")
            return 1
        if not args.verify_only:
            download(RAW_PATH)
            downloaded = True

    digest = raw_sha256(RAW_PATH)
    ok = digest == RAW_SHA256
    print(f"file:     {RAW_PATH.name}")
    print(f"sha256:   {digest}")
    print(f"expected: {RAW_SHA256}")
    print(f"status:   {'OK' if ok else 'MISMATCH'}")

    if not ok:
        if downloaded:
            print("A freshly downloaded file must match the canonical digest; this one does not.")
            print("Refusing to continue -- upstream changed or the download was corrupted.")
        else:
            print("The local file differs from the canonical copy. Results may not be comparable.")
            print(f"Replace it with the canonical file:\n  {RAW_SOURCE_URL}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
