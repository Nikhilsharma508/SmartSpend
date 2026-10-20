"""Compare every registered artifact on the same seed-42 holdout, once.

This is the model lab's engine: load each catalog artifact the registry holds,
score the *shared* test rows, and price the ranking with the same benchmark
construction the ledger records (incremental conversions valued at the
decided constants). It never fits anything and never claims a winner: the
single-seed numbers describe the artifact on file, and honesty about ranking
quality comes from the 10-seed benchmark in ``docs/concepts.md``, which this
module quotes in the returned note and per-row warnings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.causal.benchmark import BusinessAssumptions
from packages.causal.catalog import (
    MODEL_SPECS,
    ModelSpec,
    score_meaning,
)
from packages.causal.holdout import HoldoutArrays, build_holdout
from packages.causal.metrics import qini_auc, uplift_at_k
from packages.causal.registry import load_meta, load_model
from packages.decision.schemas import ModelComparison, ModelComparisonRow

#: Fold the 10-seed benchmark reading (docs/concepts.md) into the rows so no
#: consumer can quote a one-seed Qini as if it were a winner claim.
_BENCHMARK_READING: dict[str, str] = {
    "response": "chosen champion: best 10-seed profit, simplest fit (statistical tie on Qini).",
    "dr": "statistical tie with the response champion across 10 seeds — challenger, not winner.",
    "s": "weak: 10-seed Qini inside random noise — research only.",
    "x": "statistically at random across 10 seeds — research only.",
    "t": "statistically at random across 10 seeds — research only.",
    "forest": "statistically at random across 10 seeds — research only.",
}


def compare_registered_models(
    *,
    models_dir: Path | None = None,
    seed: int | None = None,
) -> ModelComparison:
    """Score every *registered* catalog artifact on the shared test split.

    Parameters
    ----------
    models_dir:
        Override the registry location (tests).
    seed:
        Force the split seed; None uses each artifact's own recorded
        ``random_state`` (all catalog artifacts were trained on 42).
    """
    assumptions = BusinessAssumptions.from_yaml()
    splits: dict[int, HoldoutArrays] = {}
    rows: list[ModelComparisonRow] = []

    for spec in MODEL_SPECS:
        try:
            meta = load_meta(spec.name, models_dir=models_dir)
        except (FileNotFoundError, ValueError):
            continue  # not registered yet — the lab lists only what exists
        model, _ = load_model(spec.name, models_dir=models_dir)
        artifact_seed = int(seed if seed is not None else meta.get("random_state", 42))
        if artifact_seed not in splits:
            splits[artifact_seed] = build_holdout(artifact_seed)
        data = splits[artifact_seed]
        score = np.asarray(model.predict(data.X_te)).ravel()
        k = float(meta.get("k", 0.3))
        n_k = max(int(round(k * len(data.y_money_te))), 1)
        from packages.causal.benchmark import incremental_conversions_at_k, profit_at_k  # noqa: E402

        incr = incremental_conversions_at_k(data.y_money_te, score, data.t_te, k=k)
        profit = profit_at_k(data.y_money_te, score, data.t_te, k, assumptions)
        rows.append(
            ModelComparisonRow(
                name=spec.name,
                label=label_for(meta, spec),
                learner=str(meta.get("learner", spec.learner)),
                tier=spec.tier,
                score_kind=spec.score_kind,
                supports_intervals=spec.supports_intervals,
                research_only=spec.research_only,
                use_case=spec.use_case,
                warning=spec.warning,
                qini_auc=round(float(qini_auc(data.y_visit_te, score, data.t_te)), 4),
                uplift_at_k=round(
                    float(uplift_at_k(data.y_visit_te, score, data.t_te, k=k)), 4
                ),
                incremental_conversions_at_k=round(float(incr), 2),
                profit_at_k_usd=round(float(profit), 2),
                profit_per_contact_usd=round(float(profit) / n_k, 4),
                n_test=data.n_test,
                trained_at=str(meta.get("created_at", "")),
                benchmark_warning=_BENCHMARK_READING.get(spec.learner),
            )
        )

    note = (
        "Every registered artifact scored on its own recorded seed's shared "
        "test split; profit is the Day-3 formula (value on conversions, cost "
        "on contacts). Single-seed numbers describe the artifact on file — "
        "the 10-seed benchmark in docs/concepts.md is the honest envelope, "
        "and it is a tie between champion and DR, not a ranking."
    )
    if not rows:
        note = (
            "No catalog models are registered yet — run "
            "`uv run python pipelines/train_all.py` to train the six artifacts."
        )
    return ModelComparison(rows=rows, note=note)


def label_for(meta: dict, spec: ModelSpec) -> str:
    """The artifact's display label, trusting the catalog over ad-hoc meta."""
    label = meta.get("label")
    return str(label) if label else spec.label


def score_meaning_text(score_kind: str) -> str:
    """Public re-export so the API and prompt builders share one sentence."""
    return score_meaning(score_kind)
