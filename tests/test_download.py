"""Fetch/verify helpers for the raw data. No network access."""

from __future__ import annotations

import pytest

from packages.datasets.hillstrom import (
    RAW_FILENAME,
    RAW_PATH,
    RAW_SHA256,
    RAW_SOURCE_URL,
    raw_sha256,
    verify_raw,
)

requires_data = pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason=f"raw Hillstrom CSV missing at {RAW_PATH}; run pipelines/download.py",
)


def test_source_url_points_at_the_canonical_file():
    assert RAW_SOURCE_URL.endswith(RAW_FILENAME)
    assert RAW_SOURCE_URL.startswith("http://")


def test_raw_sha256_raises_when_the_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        raw_sha256(tmp_path / "missing.csv")


@requires_data
def test_local_file_matches_the_pinned_digest():
    assert raw_sha256() == RAW_SHA256
    assert verify_raw()
