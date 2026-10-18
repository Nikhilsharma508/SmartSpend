"""Uplift ranking metrics: a thin, honest wrapper over ``sklift.metrics``.

Why a wrapper at all when scikit-uplift already computes these? Three reasons,
all of them about the evaluation being auditable rather than convenient:

1. **One import site.** If a metric's behaviour ever needs a fix or a note, it
   is fixed once here rather than in eight notebook cells.
2. **Random is explicit.** Every table compares models against a *simulated*
   random ranking, so "beats random" is a measured number instead of an
   assertion. The random baseline is generated with a seeded RNG, so the
   comparison is reproducible.
3. **Units stay visible.** ``uplift_at_k`` can be read per-overall-customer or
   per-targeted-customer; the default here is stated in the docstring and
   returned with the strategy attached, because the two differ by ``1/k``.

Metrics are computed on a held-out split only. Nothing here fits anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_auc_score
from sklift.metrics import uplift_at_k as _sklift_uplift_at_k

#: k values reported by :func:`ranking_table`, chosen to match the project spec
#: (uplift@10% and uplift@30% are the two depths the plan calls out).
DEFAULT_KS = (0.1, 0.3)

#: Number of random permutations averaged into the random baseline. One draw is
#: enough in expectation but noisy at k=0.1 on 8.5k rows; 20 keeps the reference
#: line smooth while staying instant.
N_RANDOM_RANKINGS = 20


def qini_auc(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
    """Area under the Qini curve, normalised by the "perfect ranking" curve.

    Returns a value that can be negative: 0 means no better than random, and a
    negative score ranks persuadables *below* random. The perfect-ranking
    reference uses the observed treatment effect, so the score is comparable
    across models on the same split but not across datasets.
    """
    return float(qini_auc_score(y_true, uplift, treatment))


def uplift_auc(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
    """AUUC — area under the uplift curve (cumulative incremental outcome)."""
    return float(uplift_auc_score(y_true, uplift, treatment))


def uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float = 0.3,
    strategy: str = "overall",
) -> float:
    """Uplift captured in the top-``k`` fraction of the ranking.

    ``strategy="overall"`` divides by *all* customers, so the number reads as
    "incremental outcome per customer in the whole population, had we targeted
    the top k". ``"treated"`` divides by the targeted only. Overall is the
    default because it is the one that multiplies up to a campaign total.
    """
    return float(_sklift_uplift_at_k(y_true, uplift, treatment, strategy=strategy, k=k))


def random_uplift(
    y_true: np.ndarray,
    treatment: np.ndarray,
    *,
    n_random: int = N_RANDOM_RANKINGS,
    random_state: int = 42,
) -> np.ndarray:
    """A near-zero uplift score from averaging random rankings.

    The result is not exactly zero: with a finite sample the mean of a few
    permutations leaves a small residual, which is what we want — the random
    baseline should occupy real space on a Qini plot rather than sit exactly on
    the axis and flatter every model next to it.
    """
    rng = np.random.default_rng(random_state)
    n = len(np.asarray(y_true))
    acc = np.zeros(n, dtype=float)
    for _ in range(n_random):
        perm = rng.permutation(n)
        ranks = np.empty(n, dtype=float)
        ranks[perm] = np.arange(n, dtype=float)
        # Standardise each ranking so its spread matches a typical model
        # output; without this the Qini curve's random line would sit flat at
        # zero and the plot would flatter every model beside it.
        acc += (ranks - ranks.mean()) / (ranks.std() + 1e-12)
    return acc / n_random


def ranking_table(
    y_true: np.ndarray,
    treatment: np.ndarray,
    scores: dict[str, np.ndarray],
    ks: tuple[float, ...] = DEFAULT_KS,
) -> pd.DataFrame:
    """One row per model: Qini AUC, AUUC, and uplift at each ``k``.

    Parameters
    ----------
    scores:
        Mapping of model name to per-customer uplift score on the *same*
        held-out split. Include a random entry (see :func:`random_uplift`) so
        the table carries its own baseline.

    Returns
    -------
    pandas.DataFrame
        Sorted by Qini AUC descending, index named ``model``. Floats rounded
        for display only — the underlying values are full precision.
    """
    rows = []
    for name, score in scores.items():
        s = np.asarray(score)
        row: dict[str, float | str] = {
            "model": name,
            "qini_auc": qini_auc(y_true, s, treatment),
            "uplift_auc": uplift_auc(y_true, s, treatment),
        }
        for k in ks:
            row[f"uplift_at_{int(k * 100)}pct"] = uplift_at_k(
                y_true, s, treatment, k=k, strategy="overall"
            )
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model").sort_values(
        "qini_auc", ascending=False
    )
    return table.round(4)
