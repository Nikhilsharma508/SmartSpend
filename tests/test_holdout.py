"""Tests for packages/causal/holdout.py — the model-agnostic split.

The holdout contract underpins every per-model context: one split, aligned
money outcomes and zip codes, deterministic per seed. Requires the raw CSV
(git-ignored): CI skips rather than fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.datasets.hillstrom import RAW_PATH

pytestmark = pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason=f"raw Hillstrom CSV missing at {RAW_PATH}; run pipelines/download.py",
)

from packages.causal.holdout import build_holdout  # noqa: E402


def test_default_split_shapes_and_alignment():
    arrays = build_holdout()
    n_tr = len(arrays.y_tr)
    n_te = len(arrays.y_visit_te)
    assert n_te == arrays.n_test > 0
    # every per-row array on the test side has the same length
    assert len(arrays.y_money_te) == n_te
    assert len(arrays.t_te) == n_te
    assert len(arrays.zip_code_te) == n_te
    assert len(arrays.feature_names) == arrays.X_te.shape[1] == arrays.X_tr.shape[1]
    assert arrays.X_tr.shape[0] == n_tr
    # binary treatment, visit outcomes in [0, 1]
    assert set(np.unique(arrays.t_te)) <= {0, 1}
    assert set(np.unique(arrays.y_visit_te)) <= {0.0, 1.0}


def test_zip_codes_alignment_is_raw_area_names():
    """zip_te is the mapped area column, aligned to X_te rows by .loc index."""
    arrays = build_holdout()
    assert set(arrays.zip_code_te) <= {"Rural", "Suburban", "Urban"}
    assert len(arrays.zip_code_te) == len(arrays.X_te)


def test_money_outcome_matches_conversion_target():
    """y_money_te must come from the conversion column, not visit spend math."""
    arrays = build_holdout()
    # conversions are rare vs visits and n is fixed by the split recipe
    assert float(arrays.y_money_te.mean()) < 0.2


def test_seed_changes_the_split():
    a = build_holdout(seed=42)
    b = build_holdout(seed=43)
    assert not np.array_equal(a.X_te, b.X_te)


def test_feature_names_are_the_encoded_schema():
    arrays = build_holdout()
    assert arrays.feature_names[0].startswith("onehot__zip_code")
    assert "num__history" in arrays.feature_names
    assert len(arrays.feature_names) == 11  # 3 zip + 3 channel + 5 numeric
