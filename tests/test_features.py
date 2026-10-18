"""Tests for the Day-2 feature pipeline in ``packages/causal/features.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.causal.features import build_features, make_splits, one_hot_design
from packages.datasets.hillstrom import FEATURES, fetch_hillstrom


@pytest.fixture(scope="module")
def hillstrom_binary() -> pd.DataFrame:
    return fetch_hillstrom("mens")


def test_build_features_excludes_leaky_columns(hillstrom_binary):
    X, y, t = build_features(hillstrom_binary, target="visit")

    assert list(X.columns) == FEATURES
    for leaky in ("segment", "treatment", "visit", "conversion", "spend"):
        assert leaky not in X.columns
    assert y.name == "visit"
    assert set(t.unique()) == {0, 1}


def test_build_features_rejects_missing_target(hillstrom_binary):
    with pytest.raises(ValueError, match="target column"):
        build_features(hillstrom_binary, target="open_rate")


def test_build_features_rejects_multi_arm():
    df = fetch_hillstrom("multi")
    with pytest.raises(ValueError, match="treatment"):
        build_features(df, target="visit")


def test_make_splits_stratifies_on_treatment_x_outcome(hillstrom_binary):
    X, y, t = build_features(hillstrom_binary, target="visit")
    splits = make_splits(X, y, t, random_state=0)

    assert set(splits) == {"train", "val", "test"}
    total = len(X)
    assert len(splits["test"][0]) == pytest.approx(0.2 * total, abs=5)
    # 80% remainder, 20% of it to val -> 16% of total
    assert len(splits["val"][0]) == pytest.approx(0.16 * total, abs=5)

    # The joint-label rate must be near-identical in every fold.
    rates = []
    for name in ("train", "val", "test"):
        _, y_i, t_i = splits[name]
        sub = pd.DataFrame({"y": y_i, "t": t_i})
        rates.append((sub["y"] == 1).mean())
    assert max(rates) - min(rates) < 0.005


def test_make_splits_no_row_overlap(hillstrom_binary):
    X, y, t = build_features(hillstrom_binary, target="visit")
    splits = make_splits(X, y, t, random_state=0)

    idx = {name: set(splits[name][0].index) for name in splits}
    assert idx["train"].isdisjoint(idx["val"])
    assert idx["train"].isdisjoint(idx["test"])
    assert idx["val"].isdisjoint(idx["test"])


def test_make_splits_zero_val_size(hillstrom_binary):
    X, y, t = build_features(hillstrom_binary, target="visit")
    splits = make_splits(X, y, t, val_size=0)
    assert set(splits) == {"train", "test"}


def test_make_splits_rejects_bad_sizes(hillstrom_binary):
    X, y, t = build_features(hillstrom_binary, target="visit")
    with pytest.raises(ValueError, match="test_size"):
        make_splits(X, y, t, test_size=1.5)
    with pytest.raises(ValueError, match="val_size"):
        make_splits(X, y, t, val_size=1.0)


def test_one_hot_design_expands_categoricals(hillstrom_binary):
    X, _, _ = build_features(hillstrom_binary, target="visit")
    transformer, names = one_hot_design(X)

    out = transformer.fit_transform(X)
    # zip_code has 3 levels, channel 3 -> 6 one-hot cols + 5 numeric = 11
    assert out.shape[1] == 11
    assert len(names) == len(FEATURES)
    assert not out.isna().any().any()
    assert np.isfinite(out.to_numpy(dtype=float)).all()
