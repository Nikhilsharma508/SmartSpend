"""Fetch and verify the raw datasets.

Hillstrom is the primary dataset and is expected to already be present in
``data/raw/``. This script verifies its SHA-256 against the pinned digest and,
if the file is absent, tells you where to get it (the original MineThatData
host is no longer reliable, so there is no automatic fetch by default).

Usage::

    uv run python pipelines/download.py            # verify
    uv run python pipelines/download.py --fetch     # try to download if missing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.datasets.hillstrom import RAW_FILENAME, RAW_PATH, RAW_SHA256  # noqa: E402

#: Community mirrors for the Hillstrom CSV. Verify the checksum after fetching.
HILLSTROM_MIRRORS = [
    "https://raw.githubusercontent.com/rileypredum/hillstrom-data/main/"
    + RAW_FILENAME,
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def try_fetch(dest: Path) -> bool:
    import urllib.request

    for url in HILLSTROM_MIRRORS:
        try:
            print(f"  trying {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"  downloaded -> {dest}")
            return True
        except Exception as exc:  # noqa: BLE001 - report and try the next mirror
            print(f"  failed: {exc}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download if missing")
    args = parser.parse_args()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        print(f"Hillstrom CSV not found at {RAW_PATH}")
        if not args.fetch:
            print("Re-run with --fetch to try the mirrors, or place the file manually.")
            return 1
        if not try_fetch(RAW_PATH):
            print("All mirrors failed. Download the file manually into data/raw/.")
            return 1

    digest = sha256(RAW_PATH)
    ok = digest == RAW_SHA256
    print(f"file:     {RAW_PATH.name}")
    print(f"sha256:   {digest}")
    print(f"expected: {RAW_SHA256}")
    print("status:   OK" if ok else "status:   MISMATCH - data differs from the pinned copy")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
