"""Day-2 feature pipeline: one design matrix, and honest train/val/test splits.

Two rules drive everything here.

**Features never see the outcome or the assignment.** ``build_features`` takes its
column list from :data:`packages.datasets.hillstrom.FEATURES`, so the treatment
(``segment``/``treatment``) and the three post-treatment outcomes can never leak
in by accident. ``history_segment`` is already excluded there (collinear with
``history``).

**The split is stratified on treatment x outcome.** Uplift evaluation compares
treated against control within each score bucket, so a split that happens to
leave one arm short of converters makes the Qini curve noisy for reasons that
have nothing to do with the model. Stratifying on the joint ``treatment x visit``
label keeps the treated:control converter ratio stable across all three splits.
Callers paid on a rarer *money* outcome can fold that outcome in too
(``stratify_extra``), and :func:`split_balance_report` makes the resulting
balance visible instead of assumed.

The validation split exists for one specific job: fitting recalibration and
choosing thresholds *without* touching the test set (Day 3). The test split is
looked at once, at the end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from packages.datasets.hillstrom import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

#: Default split sizes. Sum to 1.0.
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.2  # taken out of the 80% train remainder -> 64/16/20

#: Above this many unique values an outcome is treated as continuous and binned
#: into quantile strata for the split label. Binary outcomes (visit,
#: conversion) sit far below it; spend (578 distinct values) sits far above.
MAX_STRATA_FROM_OUTCOME = 20

#: Number of quantile bins used to stratify a continuous outcome.
N_STRATA_BINS = 10


def build_design(
    df: pd.DataFrame,
    target: str = "visit",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a cleaned Hillstrom frame into ``X``, ``y``, ``t``.

    Parameters
    ----------
    df:
        Output of :func:`fetch_hillstrom` with a binary arm (must carry a
        ``treatment`` column).
    target:
        Outcome column to model — ``"visit"`` (Day 2, the primary ranking
        target), or ``"conversion"`` / ``"spend"`` (Day-3 challengers).
    features:
        Override the feature list. Defaults to
        :data:`packages.datasets.hillstrom.FEATURES`.

    Returns
    -------
    (X, y, t):
        ``X`` keeps the original feature columns and dtypes (``zip_code`` and
        ``channel`` stay categorical so LightGBM can encode them natively);
        ``y`` is the chosen outcome; ``t`` is the 0/1 treatment indicator.
    """
    return build_features(df, target=target, features=features)


def build_features(
    df: pd.DataFrame,
    target: str = "visit",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Like :func:`build_design`, but the caller names the outcome column.

    Parameters
    ----------
    target:
        One of ``"visit"``, ``"conversion"`` or ``"spend"``.
    """
    features = features if features is not None else FEATURES
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in df")
    if "treatment" not in df.columns:
        raise ValueError("df has no 'treatment' column — load a binary arm, not 'multi'.")

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"df is missing feature columns: {sorted(missing)}")

    X = df[features].copy()
    return X, df[target].copy(), df["treatment"].astype(int).copy()


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = 42,
    stratify_extra: pd.Series | np.ndarray | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """Stratified train/val/test split on the ``treatment x outcome`` label.

    Parameters
    ----------
    X, y, t:
        As returned by :func:`build_features`.
    test_size:
        Fraction held out for the final evaluation.
    val_size:
        Fraction of the *remaining* data held out for calibration and threshold
        choices. ``val_size=0`` returns only ``train`` and ``test``.
    stratify_extra:
        Optional second outcome column (aligned positionally with ``X``) to
        fold into the stratification label, e.g. ``df["conversion"]`` when
        ranking on ``visit``. Real campaigns are paid on the *money* outcome,
        which is far rarer than the ranking target (1% converters vs 15%
        visitors on Hillstrom) — balancing only the ranking outcome can still
        leave folds with visibly different converter counts, and profit is
        exactly the quantity that wobbles. With the extra column the label
        becomes ``treatment x y x extra``. ``None`` (the default) preserves the
        original protocol byte-for-byte, so the Day-3 ledger numbers reproduce.

    Returns
    -------
    dict
        ``{"train": (X, y, t), "val": (...), "test": (...)}`` — ``"val"`` is
        absent when ``val_size=0``.

    Notes
    -----
    Stratifying on the joint label rather than on ``y`` alone matters: with a
    1-in-15 visit rate, an unstratified split can easily leave a test fold with
    a very different treated:control converter ratio, which moves the Qini
    curve independent of model quality. Use
    :func:`split_balance_report` to *see* that balance instead of assuming it.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")

    # Joint stratification label. For a discrete outcome (visit, conversion)
    # each unique (treatment, outcome) pair is its own stratum, which keeps the
    # treated:control converter ratio stable across folds. A *continuous*
    # outcome (spend) has ~hundreds of unique values, so treating each as a
    # stratum leaves most with a single member and crashes train_test_split —
    # so it is binned into quantile buckets for the label only. The binning
    # touches just the stratification string; the y values used for training
    # stay continuous and unmodified.
    y_arr = np.asarray(y)
    n_unique = len(np.unique(y_arr))
    if n_unique > MAX_STRATA_FROM_OUTCOME:
        # Quantile bins collapse ties correctly; duplicates in edges (from a
        # zero-inflated column like spend) are dropped by qcut.
        y_label = pd.qcut(y_arr, q=min(N_STRATA_BINS, n_unique), labels=False, duplicates="drop")
    else:
        y_label = y_arr
    strat = pd.Series(
        np.asarray(t).astype(str) + "_" + np.asarray(y_label).astype(str), index=X.index
    )
    if stratify_extra is not None:
        extra = np.asarray(stratify_extra)
        if len(extra) != len(X):
            raise ValueError(
                f"stratify_extra has {len(extra)} rows, expected {len(X)} "
                "(it must align positionally with X)"
            )
        # Positional join rather than index alignment: callers may pass a raw
        # array; the label is built from values, not from index semantics.
        strat = pd.Series(
            strat.to_numpy().astype(str) + "_" + extra.astype(str), index=X.index
        )

    X_temp, X_test, y_temp, y_test, t_temp, t_test, s_temp, _s_test = train_test_split(
        X, y, t, strat, test_size=test_size, random_state=random_state, stratify=strat
    )

    splits: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {
        "test": (X_test, y_test, t_test)
    }

    if val_size <= 0:
        splits["train"] = (X_temp, y_temp, t_temp)
        return splits

    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp,
        y_temp,
        t_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=s_temp,
    )
    splits["train"] = (X_train, y_train, t_train)
    splits["val"] = (X_val, y_val, t_val)
    return splits


def one_hot_design(
    X: pd.DataFrame,
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str]]:
    """Build (and fit-ready) a one-hot ``ColumnTransformer`` plus column names.

    LightGBM handles categoricals natively, so the default path leaves
    ``zip_code``/``channel`` as ``category`` dtype. This exists for the models
    that cannot — econml's causal forest and the policy-tree interpreter both
    want a plain numeric matrix.

    Returns
    -------
    (transformer, feature_names):
        ``feature_names`` is the output column order, needed to label
        importances and policy-tree splits afterwards.
    """
    categorical = categorical if categorical is not None else CATEGORICAL_FEATURES
    numeric = numeric if numeric is not None else NUMERIC_FEATURES

    present_cat = [c for c in categorical if c in X.columns]
    present_num = [c for c in numeric if c in X.columns]
    if not present_cat and not present_num:
        raise ValueError("no recognised feature columns in X")

    try:
        # sklearn >= 1.2 uses sparse_output; older uses sparse.
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - defensive, sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        [("onehot", ohe, present_cat), ("num", "passthrough", present_num)],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    transformer.set_output(transform="pandas")

    return transformer, [*present_cat, *present_num]


def encode_splits(
    splits: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]], list[str]]:
    """One-hot encode the feature block of every split, fitting on train only.

    ``causalml`` and ``econml`` both coerce ``X`` to a float matrix internally,
    so the categorical ``zip_code``/``channel`` columns that LightGBM accepts
    natively would crash them. This fits the encoder on the **training** split
    and applies it to all three, so no information leaks from val/test into the
    encoding, and every model sees the same numeric matrix.

    Returns
    -------
    (encoded_splits, feature_names):
        Same structure as the input, with ``X`` replaced by the one-hot
        matrix. ``feature_names`` is the output column order, for labelling
        importances or policy-tree splits later.
    """
    if "train" not in splits:
        raise ValueError("splits must contain a 'train' entry to fit the encoder")

    transformer, _ = one_hot_design(splits["train"][0], categorical, numeric)
    transformer.fit(splits["train"][0])

    out: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
    for name, (X_i, y_i, t_i) in splits.items():
        out[name] = (transformer.transform(X_i), y_i, t_i)

    feature_names = list(transformer.get_feature_names_out())
    return out, feature_names


def split_balance_report(
    splits: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
) -> pd.DataFrame:
    """Per-split visibility into the stratification: sizes, arms, events, SMD.

    One row per split. ``ate`` is the treated-minus-control event-rate
    difference (the unadjusted experimental effect in that fold) and ``smd`` is
    its standardized version, ``(p_t - p_c) / sqrt((p_t(1-p_t) + p_c(1-p_c))/2)``.

    Why this exists: a stratified split is a *claim* — "the arms and the events
    are balanced across folds" — and claims should be checkable. If the SMD
    moves a lot between sibling splits, the Qini and profit numbers will wobble
    for reasons that have nothing to do with the model. Stable rows mean the
    stratification is doing its job; pass ``stratify_extra`` to
    :func:`make_splits` to also balance the money outcome on top of the
    ranking target.
    """
    rows = []
    # Canonical row order (train, val, test) for human readers; unknown split
    # names — the tests use synthetic ones — keep alphabetical after those.
    order: list[str] = [n for n in ("train", "val", "test") if n in splits]
    order += sorted(set(splits) - set(order))
    for name in order:
        _X, y_i, t_i = splits[name]
        t_arr = np.asarray(t_i, dtype=float)
        y_arr = np.asarray(y_i, dtype=float)
        treated = t_arr == 1
        n_t, n_c = int(treated.sum()), int((~treated).sum())
        p_t = float(y_arr[treated].mean()) if n_t else float("nan")
        p_c = float(y_arr[~treated].mean()) if n_c else float("nan")
        pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
        rows.append(
            {
                "n": int(len(y_arr)),
                "treated_share": round(float(treated.mean()), 4),
                "events_treated": int(y_arr[treated].sum()),
                "events_control": int(y_arr[~treated].sum()),
                "rate_treated": round(p_t, 4),
                "rate_control": round(p_c, 4),
                "ate": round(p_t - p_c, 4),
                "smd": round((p_t - p_c) / pooled, 4) if pooled > 0 else float("nan"),
            }
        )
    # The index must follow the same order the rows were built in, or labels
    # silently misalign with data (caught by test_split_balance_report_shape_and_math).
    return pd.DataFrame(rows, index=order)
