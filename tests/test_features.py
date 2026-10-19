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


def test_make_splits_handles_continuous_outcome():
    """A continuous target (spend) has hundreds of unique values, so treating
    each as its own stratum would leave most strata with one member and crash
    train_test_split. It must be binned into quantile strata instead, keeping
    the treatment rate stable across folds. Regression test for that bug."""
    df = fetch_hillstrom("mens")
    X, y, t = build_features(df, target="spend")

    assert y.nunique() > 20  # genuinely continuous, unlike visit/conversion
    splits = make_splits(X, y, t, random_state=0)

    assert set(splits) == {"train", "val", "test"}
    total = len(X)
    assert len(splits["test"][0]) == pytest.approx(0.2 * total, abs=5)
    assert len(splits["val"][0]) == pytest.approx(0.16 * total, abs=5)

    # Treatment rate must be near-identical in every fold (stratified).
    rates = [splits[name][2].mean() for name in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.01

    # And the spend distribution must be preserved, not distorted by binning.
    train_mean = splits["train"][1].mean()
    assert train_mean == pytest.approx(y.mean(), rel=0.1)


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


def test_make_splits_default_matches_explicit_none(hillstrom_binary):
    """``stratify_extra=None`` must be byte-for-byte the original protocol:
    the Day-3 ledger numbers were measured on those exact splits."""
    X, y, t = build_features(hillstrom_binary, target="visit")

    base = make_splits(X, y, t, random_state=7)
    explicit = make_splits(X, y, t, random_state=7, stratify_extra=None)

    for name in base:
        assert base[name][0].index.equals(explicit[name][0].index)


def test_stratify_extra_balances_converter_counts(hillstrom_binary):
    """Ranking on visit but getting paid on conversion means the *money*
    outcome must survive the split: with ``stratify_extra=conversion`` the
    converter share is identical (up to rounding) in every fold, in both arms."""
    X, y, t = build_features(hillstrom_binary, target="visit")
    conversion = hillstrom_binary["conversion"]
    splits = make_splits(X, y, t, random_state=0, stratify_extra=conversion)

    total_conv_t = int(conversion[t == 1].sum())
    total_conv_c = int(conversion[t == 0].sum())
    for name, frac in (("train", 0.64), ("val", 0.16), ("test", 0.20)):
        _, y_i, t_i = splits[name]
        # y_i here is visit; converters align by index through the split.
        conv_i = conversion.loc[y_i.index]
        for arm, total in ((1, total_conv_t), (0, total_conv_c)):
            share_of_arm = conv_i[t_i == arm].sum() / total
            assert share_of_arm == pytest.approx(frac, abs=0.01), (name, arm)


def test_stratify_extra_rejects_misaligned_input(hillstrom_binary):
    X, y, t = build_features(hillstrom_binary, target="visit")
    with pytest.raises(ValueError, match="stratify_extra"):
        make_splits(X, y, t, stratify_extra=np.zeros(len(X) - 1))


def test_split_balance_report_shape_and_math(hillstrom_binary):
    from packages.causal.features import split_balance_report

    X, y, t = build_features(hillstrom_binary, target="visit")
    splits = make_splits(X, y, t, random_state=0)
    report = split_balance_report(splits)

    assert list(report.index) == ["train", "val", "test"]
    assert report["n"].sum() == len(X)
    for name in ("train", "val", "test"):
        _, y_i, t_i = splits[name]
        y_arr, t_arr = np.asarray(y_i), np.asarray(t_i)
        row = report.loc[name]
        assert row["n"] == len(y_i)
        assert row["events_treated"] == int(y_arr[t_arr == 1].sum())
        assert row["events_control"] == int(y_arr[t_arr == 0].sum())
        assert row["ate"] == pytest.approx(row["rate_treated"] - row["rate_control"], abs=1e-3)
        assert np.isfinite(row["smd"])


def test_split_balance_report_sees_extra_stratification(hillstrom_binary):
    """With the money outcome folded into the label, each fold holds the same
    converter share of each arm — the balance the report should surface."""
    from packages.causal.features import split_balance_report

    X, y, t = build_features(hillstrom_binary, target="visit")
    splits = make_splits(
        X, y, t, random_state=0, stratify_extra=hillstrom_binary["conversion"]
    )
    report = split_balance_report(splits)

    for events_col in ("events_treated", "events_control"):
        total = report[events_col].sum()
        for name, frac in (("train", 0.64), ("val", 0.16), ("test", 0.20)):
            assert report.loc[name, events_col] / total == pytest.approx(frac, abs=0.01)
