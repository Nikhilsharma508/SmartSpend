"""Day-3 calibration: do the models' numbers mean what they say?

Two different calibration questions live here, and conflating them is the most
common mistake in uplift evaluation.

**Outcome-model calibration (per arm).** A T-learner's uplift is the difference
of two predicted probabilities, and the optimizer multiplies that difference by
value-per-conversion. So a miscalibrated probability distorts every profit
figure even when the *ranking* is perfect. We score the treated and control
outcome models separately with **Brier score** and a **reliability curve**, then
optionally recalibrate with isotonic or Platt scaling — fitted on the
*validation* split only, never on test.

**CATE calibration (by decile).** Individual treatment effects are never
observed, so we cannot score one customer's effect. Instead we bin customers by
*predicted* uplift and compare the mean prediction against the *observed*
treated-minus-control outcome in each bin. The right shape is monotonic and
near the diagonal. A flat relationship means the model ranks but its magnitudes
are meaningless; an inverted one means it is actively wrong.

Neither check is a model-selection metric. Model choice stays with Qini AUC and
incremental profit — a model can be well calibrated and rank badly, or rank well
and be badly scaled.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

#: Number of quantile bins for reliability curves and CATE calibration.
N_BINS = 10

#: The two recalibration methods offered. Both are fit per arm on validation.
RecalMethod = Literal["isotonic", "platt"]


def brier_by_arm(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    treatment: np.ndarray,
) -> pd.DataFrame:
    """Brier score of a probability model, computed separately per arm.

    Parameters
    ----------
    y_true, p_pred:
        Observed outcome and predicted probability on the same rows.
    treatment:
        0/1 indicator, used only to split the scores.

    Returns
    -------
    pandas.DataFrame
        One row per arm with the Brier score and the base rate, plus the skill
        score against a constant base-rate predictor (1 − model/base). A model
        beating the base rate has skill > 0; that is the bar to clear, because
        predicting the base rate for everyone is the floor.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pred)
    t = np.asarray(treatment)

    rows = []
    for arm, label in [(1, "treated"), (0, "control")]:
        mask = t == arm
        y_arm, p_arm = y[mask], p[mask]
        base_rate = float(y_arm.mean())
        model_brier = float(brier_score_loss(y_arm, p_arm))
        base_brier = float(brier_score_loss(y_arm, np.full_like(p_arm, base_rate)))
        skill = 1.0 - model_brier / base_brier if base_brier > 0 else np.nan
        rows.append(
            {
                "arm": label,
                "n": int(mask.sum()),
                "base_rate": round(base_rate, 4),
                "brier_model": round(model_brier, 5),
                "brier_base_rate": round(base_brier, 5),
                "skill_vs_base": round(float(skill), 4),
            }
        )
    return pd.DataFrame(rows).set_index("arm")


def reliability_by_arm(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Reliability data (predicted vs observed rate) per arm, in quantile bins.

    Returns the points a reliability diagram is drawn from: ``mean_predicted``
    against ``mean_observed`` per bin, plus the bin count. Points on the
    diagonal mean well calibrated; above it means under-confident, below it
    over-confident.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pred)
    t = np.asarray(treatment)

    frames = []
    for arm, label in [(1, "treated"), (0, "control")]:
        mask = t == arm
        y_arm, p_arm = y[mask], p[mask]
        # A constant prediction (e.g. a model that outputs one value per arm)
        # is not an error — it is a single reliability point. Emit it as one
        # bin rather than skipping the arm, which would drop the evidence that
        # the constant is (or is not) at the right height.
        n_unique = len(np.unique(p_arm))
        if n_unique < 2:
            if len(p_arm) > 0:
                frames.append(
                    pd.DataFrame(
                        {
                            "arm": [label],
                            "mean_predicted": [float(p_arm.mean())],
                            "mean_observed": [float(y_arm.mean())],
                            "bin": [1],
                        }
                    )
                )
            continue
        frac_pred, frac_obs = calibration_curve(
            y_arm, p_arm, n_bins=n_bins, strategy="quantile"
        )
        frames.append(
            pd.DataFrame(
                {
                    "arm": label,
                    "mean_predicted": frac_pred,
                    "mean_observed": frac_obs,
                    "bin": np.arange(1, len(frac_pred) + 1),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["arm", "mean_predicted", "mean_observed", "bin"]
        )
    return pd.concat(frames, ignore_index=True)


def fit_recalibrator(
    p_val: np.ndarray,
    y_val: np.ndarray,
    method: RecalMethod = "isotonic",
):
    """Fit a per-arm recalibrator on **validation** data only.

    The returned callable maps a raw probability to a recalibrated one. Fitting
    on validation (not test) is the whole point: test must stay untouched until
    the final number is read, or the evaluation silently leaks.

    ``method="isotonic"`` is a non-parametric monotone map — flexible, but it
    overfits on small validation sets. ``"platt"`` fits a logistic (sigmoid) —
    more stable, but it can only correct confidence, not ranking.
    """
    p = np.asarray(p_val, dtype=float)
    y = np.asarray(y_val, dtype=int)

    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p, y)
        return ir.predict
    if method == "platt":
        lr = LogisticRegression()
        lr.fit(p.reshape(-1, 1), y)
        return lambda x: lr.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown recalibration method {method!r}; use 'isotonic' or 'platt'")


def cate_calibration_by_decile(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Predicted vs observed uplift per decile of the ranking.

    Customers are binned by *predicted* uplift (quantile bins), then within each
    bin we compute the predicted mean and the **observed** treated-minus-control
    outcome difference. This is the uplift analogue of a reliability curve.

    A usable model shows a **monotonic** increase in observed uplift down the
    deciles and points near the diagonal. Read the two failure modes:

    - **Flat observed uplift** — the model ranks but its magnitudes carry no
      information (only the top decile may separate).
    - **Inverted** — the model is actively wrong; targeting its top decile
      would *reduce* outcome.

    The observed per-bin difference is noisy (each bin has ~n/10 customers split
    across arms), so treat single bins as indicative and the *trend* as the
    signal.
    """
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(uplift_score, dtype=float)
    t = np.asarray(treatment, dtype=float)

    # Quantile bins of the score, lowest first.
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-12  # include the minimum in the first bin
    decile = np.digitize(score, edges[1:-1], right=False) + 1

    rows = []
    for d in range(1, n_bins + 1):
        mask = decile == d
        if mask.sum() == 0:
            continue
        t_d, y_d = t[mask], y[mask]
        treated = t_d == 1
        # Observed uplift = difference in means between arms *within this bin*.
        if treated.any() and (~treated).any():
            observed = float(y_d[treated].mean() - y_d[~treated].mean())
        else:
            observed = np.nan
        rows.append(
            {
                "decile": d,
                "n": int(mask.sum()),
                "predicted_uplift": float(score[mask].mean()),
                "observed_uplift": observed,
            }
        )

    table = pd.DataFrame(rows)

    # Diagnostics: is the observed column monotonic, and does it track the
    # predicted column? Both are reported rather than asserted, because the
    # right answer depends on the model.
    obs = table["observed_uplift"].to_numpy()
    valid = ~np.isnan(obs)
    if valid.sum() >= 2:
        diffs = np.diff(obs[valid])
        table.attrs["monotonic_increasing"] = bool((diffs >= 0).all())
        table.attrs["frac_deciles_monotonic"] = float((diffs >= 0).mean())
        # Rank correlation between predicted and observed across deciles.
        from scipy.stats import spearmanr

        corr, _ = spearmanr(
            table.loc[valid, "predicted_uplift"], obs[valid]
        )
        table.attrs["spearman_pred_vs_obs"] = round(float(corr), 4)
    else:
        table.attrs["monotonic_increasing"] = False
        table.attrs["frac_deciles_monotonic"] = np.nan
        table.attrs["spearman_pred_vs_obs"] = np.nan

    return table.round(5)
