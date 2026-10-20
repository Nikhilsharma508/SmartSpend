"""The seeded holdout bundle — one construction, shared by every consumer.

``build_holdout(seed)`` rebuilds the exact split ``pipelines/train*.py`` used
(mens arm, ``visit`` target, stratified treatment x visit, one-hot encoded on
train) and returns the aligned arrays every downstream consumer needs: the
model-agnostic data, not any model's scores. The copilot planner, the model
comparison and any future per-model context all reuse it so a "shared split"
is a real shared object instead of three copy-pasted constructions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.causal.features import build_features, encode_splits, make_splits
from packages.datasets import hillstrom


@dataclass(frozen=True)
class HoldoutArrays:
    """The model-agnostic split the decision layer evaluates policies on.

    ``y_visit_te`` is the ranking outcome on test rows; ``y_money_te`` is the
    money outcome (conversions) the profit formula pays on; ``zip_code_te``
    is the fairness-stratification column kept aligned by construction.
    """

    X_tr: np.ndarray
    t_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    t_te: np.ndarray
    y_visit_te: np.ndarray
    y_money_te: np.ndarray
    zip_code_te: np.ndarray
    feature_names: list[str]

    @property
    def n_test(self) -> int:
        return int(len(self.y_money_te))


def build_holdout(seed: int = 42, target: str = "visit") -> HoldoutArrays:
    """Rebuild the seeded split for the mens arm exactly as training does."""
    df = hillstrom.fetch_hillstrom(arm="mens")
    X, y, t = build_features(df, target=target)
    enc, feature_names = encode_splits(make_splits(X, y, t, random_state=seed))
    X_tr, y_tr, t_tr = enc["train"]
    X_te, y_te, t_te = enc["test"]
    return HoldoutArrays(
        X_tr=X_tr.to_numpy(dtype=float),
        t_tr=t_tr.to_numpy(dtype=int),
        y_tr=y_tr.to_numpy(dtype=float),
        X_te=X_te.to_numpy(dtype=float),
        t_te=t_te.to_numpy(dtype=int),
        y_visit_te=y_te.to_numpy(dtype=float),
        y_money_te=df.loc[X_te.index, "conversion"].to_numpy(dtype=float),
        zip_code_te=df.loc[X_te.index, "zip_code"].to_numpy(),
        feature_names=list(feature_names),
    )
