"""Day-1 experiment validation: standardized mean differences and ATE with CIs.

Starter implementations for the notebook. Verify and own these before trusting
them — they are intentionally small and dependency-light (pandas + scipy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Rule of thumb from the project spec: flag any SMD above this.
SMD_FLAG_THRESHOLD = 0.1


def _levels(df: pd.DataFrame, col: str) -> list:
    """Return the values of ``col`` to compare, expanding categoricals to one-vs-rest."""
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        return list(df[col].cat.categories)
    return [None]  # None => treat the column as a single numeric/binary variable


def _series_for_level(df: pd.DataFrame, col: str, level) -> pd.Series:
    if level is None:
        return df[col].astype(float)
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        return (df[col] == level).astype(float)
    return (df[col] == level).astype(float)


def smd_table(
    df: pd.DataFrame,
    features: list[str],
    treatment_col: str = "treatment",
    *,
    threshold: float = SMD_FLAG_THRESHOLD,
) -> pd.DataFrame:
    """Standardized mean difference per feature, treated vs control.

    For categorical features each level is compared one-vs-rest. Uses the
    pooled-variance form ``(m1 - m0) / sqrt((v1 + v0) / 2)``, which is the
    convention in the project spec. Categorical levels are shown as ``col=level``.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    rows = []
    for col in features:
        for level in _levels(df, col):
            a = _series_for_level(treated, col, level).to_numpy()
            b = _series_for_level(control, col, level).to_numpy()
            m1, m0 = a.mean(), b.mean()
            v1, v0 = a.var(ddof=1), b.var(ddof=1)
            denom = np.sqrt((v1 + v0) / 2.0)
            smd = 0.0 if denom == 0 else (m1 - m0) / denom
            rows.append(
                {
                    "feature": col if level is None else f"{col}={level}",
                    "treated_mean": m1,
                    "control_mean": m0,
                    "smd": smd,
                    "abs_smd": abs(smd),
                    "flagged": abs(smd) > threshold,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_smd", ascending=False)
        .reset_index(drop=True)
    )


def ate_table(
    df: pd.DataFrame,
    outcomes: list[str],
    treatment_col: str = "treatment",
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Average treatment effect per outcome, treated minus control, with a 95% CI.

    Binary outcomes (0/1) use the normal-approximation two-proportion interval;
    continuous outcomes (e.g. ``spend``) use Welch's t interval. Both are the
    standard large-sample choices; with n ~ 21k per arm they agree closely with
    a bootstrap.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    rows = []
    for col in outcomes:
        a = treated[col].astype(float).to_numpy()
        b = control[col].astype(float).to_numpy()
        effect = a.mean() - b.mean()
        is_binary = set(np.unique(a)) <= {0.0, 1.0} and set(np.unique(b)) <= {0.0, 1.0}

        if is_binary:
            se = np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
            test_name = "two-proportion z"
        else:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            test_name = "Welch t"

        lo, hi = effect - z * se, effect + z * se
        rows.append(
            {
                "outcome": col,
                "treated_mean": a.mean(),
                "control_mean": b.mean(),
                "ate": effect,
                "se": se,
                "ci_low": lo,
                "ci_high": hi,
                "abs_ate_over_se": abs(effect / se) if se else np.nan,
                "test": test_name,
                "significant": (lo > 0) or (hi < 0),
            }
        )
    return pd.DataFrame(rows)
