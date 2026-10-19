"""Readable segments from a policy tree, plus the fairness audit of that policy.

The Day-4 question "which customers should we email?" is only actionable if a
human can read *why*. :func:`get_segment_profile` fits econml's honest
``DRPolicyTree`` — doubly-robust, so the split rules are chosen for treatment
heterogeneity rather than for outcome prediction — caps it at a shallow depth
so every leaf is a sentence ("recency > 6 and history <= 500 → email"), and
then **scores each segment on untouched holdout rows**: the treated-minus-
control visit lift with its interval, and the money-outcome lift when given.
The tree proposes; the holdout disposes.

The same policy is audited per protected-group bucket (``zip_code``): the
share each group would actually be contacted at, and its own holdout lift.
Both numbers matter — a policy that targets a group far above its population
share while that group's uplift interval straddles zero is over-reaching, and
this is where that becomes visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.decision.profit import Z_95, binary_diff_with_se
from packages.decision.schemas import FairnessRow, Segment, SegmentProfile

#: sklearn marks leaf nodes with this feature index.
_LEAF = -2


def _leaf_rules(tree_model, feature_names: list[str] | None) -> dict[int, list[str]]:
    """Map each leaf id to the decision path that reaches it, as readable strings."""
    th = tree_model.tree_
    n_features = th.n_features
    names = feature_names if feature_names and len(feature_names) == n_features else None

    rules: dict[int, list[str]] = {}

    def walk(node: int, path: list[str]) -> None:
        if th.feature[node] == _LEAF:
            rules[node] = path
            return
        f_idx = int(th.feature[node])
        fname = names[f_idx] if names else f"x{f_idx}"
        thr = float(th.threshold[node])
        walk(int(th.children_left[node]), [*path, f"{fname} <= {thr:.4g}"])
        walk(int(th.children_right[node]), [*path, f"{fname} > {thr:.4g}"])

    walk(0, [])
    return rules


def _diff_ci(y_t: np.ndarray, y_c: np.ndarray) -> tuple[float, float, float]:
    diff, se = binary_diff_with_se(y_t, y_c)
    if not np.isfinite(se):
        return float("nan"), float("nan"), float("nan")
    return diff, diff - Z_95 * se, diff + Z_95 * se


def get_segment_profile(
    X_train: np.ndarray | pd.DataFrame,
    t_train: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    X_eval: np.ndarray | pd.DataFrame,
    t_eval: np.ndarray | pd.Series,
    y_eval: np.ndarray | pd.Series,
    *,
    feature_names: list[str] | None = None,
    y_eval_money: np.ndarray | pd.Series | None = None,
    fairness_col: np.ndarray | pd.Series | None = None,
    max_depth: int = 3,
    min_samples_leaf: int = 200,
    random_state: int = 42,
) -> SegmentProfile:
    """Fit a shallow honest policy tree on train; report segments on holdout.

    Parameters
    ----------
    X_train, t_train, y_train:
        Fit data — must be the *training* split; the tree never sees holdout.
    X_eval, t_eval, y_eval:
        Untouched holdout; every uplift number below is measured here, not
        taken from the tree's leaf values.
    feature_names:
        Column names of the (encoded) X, used to render rules. Falls back to
        ``x0..xn`` when absent or misaligned.
    y_eval_money:
        Optional money outcome on the eval rows (``conversion`` when the tree
        ranks ``visit``) — adds a conversion-lift column per segment.
    fairness_col:
        Optional group labels aligned with the eval rows (e.g. ``zip_code``);
        triggers the fairness audit of the same policy.
    """
    from econml.policy import DRPolicyTree

    X_tr = np.asarray(X_train, dtype=float)
    X_ev = np.asarray(X_eval, dtype=float)
    t_tr = np.asarray(t_train, dtype=int)
    y_tr = np.asarray(y_train, dtype=float)
    t_ev = np.asarray(t_eval, dtype=int)
    y_ev = np.asarray(y_eval, dtype=float)
    y_money = np.asarray(y_eval_money, dtype=float) if y_eval_money is not None else None

    tree = DRPolicyTree(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
    )
    tree.fit(y_tr, t_tr, X=X_tr)

    policy_eval = np.asarray(tree.predict(X_ev)).astype(int)
    leaf_eval = np.asarray(tree.policy_model_.apply(X_ev)).astype(int)
    rules = _leaf_rules(tree.policy_model_, feature_names)

    segments: list[Segment] = []
    for segment_id, leaf in enumerate(sorted(np.unique(leaf_eval))):
        mask = leaf_eval == leaf
        treated = t_ev[mask] == 1
        uplift, lo, hi = _diff_ci(y_ev[mask][treated], y_ev[mask][~treated])
        money = None
        if y_money is not None and treated.any() and (~treated).any():
            money = round(
                float(y_money[mask][treated].mean() - y_money[mask][~treated].mean()), 5
            )
        segments.append(
            Segment(
                segment_id=segment_id,
                rules=rules.get(leaf, []),
                n_eval=int(mask.sum()),
                share=round(float(mask.mean()), 4),
                policy_arm="email" if np.bincount(policy_eval[mask]).argmax() else "no email",
                uplift_visit=round(uplift, 5),
                uplift_visit_ci_low=round(lo, 5),
                uplift_visit_ci_high=round(hi, 5),
                uplift_conversion=money,
            )
        )

    fairness: list[FairnessRow] = []
    if fairness_col is not None:
        groups = pd.Series(np.asarray(fairness_col)).reset_index(drop=True)
        policy_series = pd.Series(policy_eval).reset_index(drop=True)
        t_series = pd.Series(t_ev).reset_index(drop=True)
        y_series = pd.Series(y_ev).reset_index(drop=True)
        for group, idx in groups.groupby(groups).groups.items():
            idx = np.asarray(idx)
            treated = t_series.iloc[idx].to_numpy() == 1
            uplift, lo, hi = _diff_ci(y_series.iloc[idx].to_numpy()[treated],
                                      y_series.iloc[idx].to_numpy()[~treated])
            ci_includes_zero = bool(np.isfinite(lo) and lo <= 0 <= hi)
            fairness.append(
                FairnessRow(
                    group=str(group),
                    n_eval=int(len(idx)),
                    targeting_rate=round(float((policy_series.iloc[idx] != 0).mean()), 4),
                    uplift_visit=round(uplift, 5),
                    uplift_visit_ci_low=round(lo, 5),
                    uplift_visit_ci_high=round(hi, 5),
                    note="uplift CI includes zero" if ci_includes_zero else "",
                )
            )

    return SegmentProfile(
        segments=segments,
        fairness=fairness,
        note=(
            "Rules come from an honest DRPolicyTree capped at max_depth; every "
            "uplift number is re-measured on the untouched holdout as "
            "treated-minus-control within the segment, with a 95% normal-"
            "approximation interval. Small segments are noisy by construction — "
            "read the interval before the point."
        ),
    )
