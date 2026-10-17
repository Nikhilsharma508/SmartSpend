"""Smoke tests for the Hillstrom loader and Day-1 diagnostics."""

from __future__ import annotations

import pytest

from packages.causal.diagnostics import ate_table, smd_table
from packages.datasets.hillstrom import (
    ARM_CONTROL,
    FEATURES,
    OUTCOMES,
    RAW_PATH,
    arm_summary,
    fetch_hillstrom,
    load_hillstrom_raw,
)

# The raw CSV is git-ignored, so CI and fresh clones must skip rather than fail.
pytestmark = pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason=f"raw Hillstrom CSV missing at {RAW_PATH}; run pipelines/download.py",
)


@pytest.fixture(scope="module")
def raw():
    return load_hillstrom_raw()

@pytest.fixture(scope="module")
def binary():
    return fetch_hillstrom(arm="mens")


def test_raw_shape_and_columns(raw):
    assert raw.shape == (64_000, 12)
    assert list(raw.columns)[:3] == ["recency", "history_segment", "history"]
    assert raw[["visit", "conversion", "spend"]].notna().all().all()


def test_raw_has_three_balanced_arms(raw):
    counts = raw["segment"].value_counts()
    assert set(counts.index) == {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
    # ~1/3 each, within a 1% tolerance
    assert (counts / len(raw)).between(0.32, 0.35).all()


def test_spend_is_zero_inflated_and_tied_to_conversion(raw):
    assert int((raw["spend"] > 0).sum()) == int(raw["conversion"].sum()) == 578
    assert (raw.loc[raw["conversion"] == 0, "spend"] == 0).all()


def test_binary_frame_is_clean_and_disjoint(binary):
    assert set(binary["segment"].unique()) == {"Mens E-Mail", ARM_CONTROL}
    assert len(binary) == 42_613
    assert binary["treatment"].isin([0, 1]).all()
    assert binary["treatment"].mean() == pytest.approx(0.5, abs=0.01)
    # invariants
    assert binary["spend"].ge(0).all()
    assert binary.loc[binary["conversion"] == 1, "visit"].eq(1).all()


def test_zip_typo_is_normalized_by_default(binary):
    assert "Surburban" not in set(binary["zip_code"].cat.categories)
    assert "Suburban" in set(binary["zip_code"].cat.categories)


def test_arm_summary_covers_all_arms(raw):
    summary = arm_summary(raw)
    assert len(summary) == 3
    assert summary["n"].sum() == 64_000


def test_smd_table_is_small_and_mostly_balanced(binary):
    smd = smd_table(binary, FEATURES)
    assert {"feature", "smd", "abs_smd", "flagged"} <= set(smd.columns)
    assert smd["abs_smd"].max() < 0.1  # randomized data should be balanced
    assert smd["flagged"].sum() == 0


def test_ate_table_reports_positive_visit_lift(binary):
    ate = ate_table(binary, OUTCOMES)
    visit = ate.set_index("outcome").loc["visit"]
    # e-mail lifts visits by roughly 7 points with a CI excluding zero
    assert visit["ate"] == pytest.approx(0.07, abs=0.02)
    assert visit["ci_low"] > 0
    assert visit["significant"]
    assert {"outcome", "ate", "ci_low", "ci_high"} <= set(ate.columns)
