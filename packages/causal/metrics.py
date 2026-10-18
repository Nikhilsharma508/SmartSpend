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

    Reproduces scikit-uplift's definition exactly, but without its
    ``check_is_binary`` guard, so it also works for a continuous outcome such as
    ``spend``. The arithmetic is a plain **difference of means**, which is valid
    for any numeric outcome:

    - ``strategy="overall"`` (default): take the top ``k`` from the *whole*
      ranked list, then report ``mean(y | treated) − mean(y | control)`` among
      just those targeted. This is the number to compare models on, because it
      reflects the mix of customers a single campaign would actually reach.
    - ``strategy="by_group"``: rank each arm separately and take the top ``k``
      of each — where ``k`` scales **each arm's own count**, so the two subsets
      can differ in size — then difference the two means. This answers a
      different question — "how good are the best ``k`` of *each* arm?" — and
      is not the ranking a single campaign would use.

    ``k`` may be a float in ``(0, 1)`` (a fraction) or an int (an absolute
    count), matching sklift. Returns NaN (not a crash) if the top-``k`` happens
    to contain only one arm, since the difference is then undefined.

    Notes
    -----
    This is a *rate* difference, not a count: it does not multiply by ``k·n``.
    For the total-incremental-outcome reading, multiply by the number targeted
    (see ``packages.causal.benchmark.incremental_conversions_at_k``).
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(uplift, dtype=float)
    t = np.asarray(treatment, dtype=float)

    if strategy not in ("overall", "by_group"):
        raise ValueError(f"strategy must be 'overall' or 'by_group', got {strategy!r}")

    n_samples = len(y)
    # Descending by score; mergesort-then-reverse mirrors sklift so a tied
    # ranking breaks the same way here as it does there.
    order = np.argsort(s, kind="mergesort")[::-1]

    k_kind = np.asarray(k).dtype.kind
    if k_kind == "f":
        if not 0 < k < 1:
            raise ValueError(f"float k must be in (0, 1), got {k}")
        n_size = int(n_samples * k)
    elif k_kind == "i":
        if not 0 < k < n_samples:
            raise ValueError(f"int k must be in (0, {n_samples}), got {k}")
        n_size = int(k)
    else:
        raise ValueError(f"k must be a float fraction or int count, got {k!r}")

    if strategy == "overall":
        y_top, t_top = y[order][:n_size], t[order][:n_size]
        treated, control = y_top[t_top == 1], y_top[t_top == 0]
    else:  # by_group: top-k within each arm, ranked independently.
        # A float k scales *each arm's own count* (so the two subsets differ in
        # size); an int k is an absolute count taken from both. Reusing one
        # shared size here would silently diverge from scikit-uplift.
        ctrl_order = order[t[order] == 0]
        trmnt_order = order[t[order] == 1]
        n_ctrl = int(len(ctrl_order) * k) if k_kind == "f" else n_size
        n_trmnt = int(len(trmnt_order) * k) if k_kind == "f" else n_size
        control = y[ctrl_order][:n_ctrl]
        treated = y[trmnt_order][:n_trmnt]

    if len(treated) == 0 or len(control) == 0:
        return float("nan")  # one arm absent from the top-k -> undefined
    return float(treated.mean() - control.mean())


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

    Notes
    -----
    **Continuous outcomes** (``spend``) take a different path. sklift's
    ``qini_auc_score`` and ``uplift_auc_score`` require a binary ``y_true`` and
    raise on anything else, because the "perfect ranking" normaliser is only
    defined for 0/1. For a continuous target the table therefore drops those two
    columns and reports only the direct incremental-outcome-at-k, which is
    meaningful for any outcome type (it is just the treated-minus-control mean
    difference among the top k, scaled by k·n). The table is then sorted by the
    first ``uplift_at_*`` column instead of Qini AUC.
    """
    y_arr = np.asarray(y_true)
    # An outcome is binary if every value is 0 or 1 (int or float).
    is_binary = bool(np.all(np.isin(np.unique(y_arr), [0, 1])))

    rows = []
    for name, score in scores.items():
        s = np.asarray(score)
        row: dict[str, float | str] = {"model": name}
        if is_binary:
            row["qini_auc"] = qini_auc(y_true, s, treatment)
            row["uplift_auc"] = uplift_auc(y_true, s, treatment)
        for k in ks:
            row[f"uplift_at_{int(k * 100)}pct"] = uplift_at_k(
                y_true, s, treatment, k=k, strategy="overall"
            )
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")
    sort_col = "qini_auc" if is_binary else next(
        c for c in table.columns if c.startswith("uplift_at_")
    )
    return table.sort_values(sort_col, ascending=False).round(4)
