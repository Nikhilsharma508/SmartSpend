"""Qini / uplift-curve plotting, built on the shared figure convention.

One figure carries the whole Day-2 argument: every uplift model, the response
baseline, and a measured random line, drawn on the same axes. Keeping it in a
module (rather than inline in the notebook) means the figure is identical on
Day 2 and Day 3, and that the comparison a decision rests on cannot drift
between runs.

The curve itself is the *cumulative incremental outcome* against targeting
depth. A model worth deploying pulls above the random line early and keeps
diverging; a response model typically tracks random, which is the visual this
module exists to produce.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from packages.plotting import save_fig

#: Distinct, colourblind-safe cycle reused across every figure so model names
#: are recognisable from one plot to the next.
MODEL_COLOURS = {
    "response": "#4c72b0",   # blue
    "s-learner": "#55a868",  # green
    "t-learner": "#c44e52",  # red
    "x-learner": "#8172b3",  # purple
    "dr-learner": "#ccb974", # yellow-brown
    "causal forest": "#64b5cd",  # cyan
    "random": "#8c8c8c",     # grey
}


def cumulative_uplift_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcome as customers are targeted top-down.

    Returns ``(depth, cumulative_uplift)`` where ``depth`` is the fraction of
    customers targeted (0..1) and ``cumulative_uplift`` is the running
    difference in mean outcome between treated and control *within the targeted
    set*, scaled by the number targeted so it reads as a count.

    This is the standard Qini construction: sort by score descending, then walk
    the ranking accumulating (treated outcome - control outcome) among those
    selected so far.
    """
    y = np.asarray(y_true, dtype=float)
    t = np.asarray(treatment, dtype=float)
    order = np.argsort(-np.asarray(uplift))
    y_sorted, t_sorted = y[order], t[order]

    is_treated = t_sorted == 1
    cum_treated_y = np.cumsum(y_sorted * is_treated)
    cum_treated_n = np.cumsum(is_treated)
    cum_control_y = np.cumsum(y_sorted * (1 - is_treated))
    cum_control_n = np.cumsum(1 - is_treated)

    # Incremental outcome among those targeted so far, expressed per all customers.
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_treated = np.where(cum_treated_n > 0, cum_treated_y / cum_treated_n, 0.0)
        rate_control = np.where(cum_control_n > 0, cum_control_y / cum_control_n, 0.0)
    incremental_rate = rate_treated - rate_control

    cumulative = incremental_rate * np.arange(1, len(y) + 1)
    depth = np.arange(1, len(y) + 1) / len(y)

    # Downsample to n_points for a small, smooth line without changing shape.
    idx = np.linspace(0, len(depth) - 1, min(n_points, len(depth))).astype(int)
    return depth[idx], cumulative[idx]


def perfect_qini(
    y_true: np.ndarray, treatment: np.ndarray, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """The reference curve for a hypothetical perfect ranking.

    A perfect model targets every incremental converter first, so its curve is
    a straight line from the origin to the total observed lift. Models are read
    as a fraction of this line.
    """
    y = np.asarray(y_true, dtype=float)
    t = np.asarray(treatment, dtype=float)
    total_lift = y[t == 1].mean() - y[t == 0].mean()
    depth = np.linspace(0, 1, n_points)
    return depth, depth * total_lift * len(y)


def plot_qini(
    y_true: np.ndarray,
    treatment: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    title: str = "Qini curve — cumulative incremental outcome vs targeting depth",
    name: str | None = "qini_curves",
    show_perfect: bool = True,
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """Draw every model's Qini curve on one figure, and save it.

    Parameters
    ----------
    scores:
        Model name -> per-customer uplift score on the same held-out split.
        Lowercase names matching :data:`MODEL_COLOURS` get their usual colour.
    name:
        Figure stem for :func:`packages.plotting.save_fig`; ``None`` skips
        saving (useful in tests).
    show_perfect:
        Draw the perfect-ranking reference as a dashed grey line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if show_perfect:
        px, py = perfect_qini(y_true, treatment)
        ax.plot(px, py, ls="--", lw=1, color="#333333", label="perfect")

    for model, score in scores.items():
        x, y_curve = cumulative_uplift_curve(y_true, score, treatment)
        colour = MODEL_COLOURS.get(model.lower())
        ax.plot(x, y_curve, lw=2, label=model, color=colour)

    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("Fraction of customers targeted")
    ax.set_ylabel("Cumulative incremental outcome")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if name:
        save_fig(fig, name, title)

    return fig
