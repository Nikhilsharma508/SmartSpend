"""Train and register a *finalized* model with full provenance.

    uv run python pipelines/train.py                        # response baseline on visit, seed 42
    uv run python pipelines/train.py --name alt-dr --learner dr

The protocol mirrors the Day-3 benchmark exactly — one seeded 64/16/20
stratified split, fit on **train only**, evaluated once on the untouched test —
so the numbers written into the model's metadata trace to the ledger in
``docs/concepts.md`` instead of being a fresh, unverifiable measurement. ``val``
stays reserved for calibration and ``test`` is looked at exactly once, which is
the same discipline the benchmark follows.

Every artifact gets its semantics (tier, score meaning, Research-Mode
restrictions) from the single source of truth in ``packages/causal/catalog.py``
— the same mapping ``pipelines/train_all.py`` loops over. Writes
``models/<name>.joblib`` + ``models/<name>.meta.json`` (git-ignored;
regenerable from this script plus the checksum-verified raw data). Day 6's API
loads the registered model and never trains.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.causal.benchmark import (  # noqa: E402
    BusinessAssumptions,
    incremental_conversions_at_k,
    profit_at_k,
)
from packages.causal.catalog import MODEL_SPECS, get_spec, spec_for_learner  # noqa: E402
from packages.causal.features import build_features, encode_splits, make_splits  # noqa: E402
from packages.causal.learners import LEARNER_NAMES, LearnerName, make_learner  # noqa: E402
from packages.causal.metrics import qini_auc, uplift_at_k  # noqa: E402
from packages.causal.registry import MODELS_DIR, save_model  # noqa: E402
from packages.datasets.hillstrom import FEATURES, RAW_SHA256, fetch_hillstrom  # noqa: E402

#: Distributions whose versions are recorded in the metadata sidecar, so a
#: future load can tell "same artifact" from "artifact from another stack".
TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "causalml", "econml")

#: Artifact stems the catalog already names (packages/causal/catalog.py).
_CATALOG_NAMES = {s.name for s in MODEL_SPECS}


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover - defensive
            out[pkg] = None
    return out


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - defensive
        return None


def train_and_register(
    *,
    name: str = "champion",
    learner: LearnerName = "response",
    target: str = "visit",
    seed: int = 42,
    k: float = 0.3,
    models_dir: Path | None = None,
) -> dict:
    """Fit ``learner`` on the training split, score it once on test, register it.

    Returns the metadata dict that was written alongside the artifact.
    """
    assumptions = BusinessAssumptions.from_yaml()
    df = fetch_hillstrom(arm="mens")
    X, y, t = build_features(df, target=target)
    # One-hot encode exactly like the benchmark's make_fold does, so the fit
    # sees the same numeric matrix and the metadata metrics trace to the
    # Day-3 ledger instead of being a fresh, unverifiable measurement.
    splits, _feature_names = encode_splits(make_splits(X, y, t, random_state=seed))
    X_tr, y_tr, t_tr = splits["train"]
    X_te, y_te, t_te = splits["test"]

    # Money outcome on the test rows, aligned by index — the same splitter
    # produced both frames, so .loc cannot misalign (Day-3 profit rule: value
    # on conversions, never on the ranking proxy).
    y_money_te = df.loc[X_te.index, "conversion"].to_numpy(dtype=float)

    model = make_learner(learner, target=target, random_state=seed)
    model.fit(X_tr.to_numpy(), t_tr.to_numpy(), y_tr.to_numpy())
    score = np.asarray(model.predict(X_te.to_numpy())).ravel()

    n_test = len(X_te)
    n_k = max(int(round(k * n_test)), 1)
    incr = incremental_conversions_at_k(y_money_te, score, t_te, k=k)
    profit = profit_at_k(y_money_te, score, t_te, k, assumptions)
    # Uniform reference: targeting a random fraction of the list has expected
    # incremental conversions = n_k * overall ATE on the money outcome.
    ate_money = float(
        y_money_te[np.asarray(t_te) == 1].mean() - y_money_te[np.asarray(t_te) == 0].mean()
    )
    profit_uniform = float(
        ate_money * n_k * assumptions.value_per_conversion_usd
        - n_k * assumptions.cost_per_contact_usd
    )

    meta = {
        "learner": learner,
        "target": target,
        "random_state": seed,
        "fit_scope": (
            "train only (64%); val reserved for calibration, test untouched — "
            "matches the Day-3 benchmark protocol"
        ),
        "train_rows": int(len(X_tr)),
        "data_sha256": RAW_SHA256,
        "k": k,
        "split": (
            "stratified treatment x visit, 64/16/20, one-hot encoded on train "
            "(packages.causal.features.make_splits + encode_splits)"
        ),
        "features": list(FEATURES),
        "metrics": {
            "n_test": n_test,
            "n_targeted": n_k,
            "qini_auc": round(float(qini_auc(y_te, score, t_te)), 4),
            "uplift_at_k": round(float(uplift_at_k(y_te, score, t_te, k=k)), 4),
            "incremental_conversions_at_k": round(incr, 2),
            "profit_at_k_usd": round(profit, 2),
            "profit_per_contact_usd": round(profit / n_k, 4),
            "uniform_reference_profit_usd": round(profit_uniform, 2),
        },
        "assumptions": {
            "margin_rate": assumptions.margin_rate,
            "cost_per_contact_usd": assumptions.cost_per_contact_usd,
            "value_per_conversion_usd": assumptions.value_per_conversion_usd,
        },
        "packages": _package_versions(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "registered_by": "pipelines/train.py",
        # Semantics block from the model catalog (packages/causal/catalog.py):
        # what tier this artifact holds and what its score means. Unknown
        # artifacts keep their learner's semantics; a fabricated artifact name
        # never silently becomes "the champion".
        "label": (
            get_spec(name).label
            if name in _CATALOG_NAMES
            else getattr(spec_for_learner(learner), "label", learner)
        ),
        "tier": getattr(spec_for_learner(learner), "tier", "research"),
        "score_kind": getattr(spec_for_learner(learner), "score_kind", "relative_uplift"),
        "supports_intervals": bool(
            getattr(spec_for_learner(learner), "supports_intervals", False)
        ),
        "research_only": bool(getattr(spec_for_learner(learner), "research_only", True)),
        "use_case": (
            getattr(
                spec_for_learner(learner),
                "use_case",
                "Manually trained artifact; see the model card before use.",
            )
        ),
    }

    path = save_model(model, name, meta=meta, models_dir=models_dir)
    return {"model_path": str(path), **meta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="champion", help="artifact stem in models/")
    parser.add_argument(
        "--learner", default="response", choices=LEARNER_NAMES, help="which model to fit"
    )
    parser.add_argument("--target", default="visit", help="ranking outcome to model")
    parser.add_argument("--seed", type=int, default=42, help="split + fit seed")
    parser.add_argument("--k", type=float, default=0.3, help="targeting depth for metrics")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="override the models/ location"
    )
    args = parser.parse_args(argv)

    meta = train_and_register(
        name=args.name,
        learner=args.learner,  # type: ignore[arg-type]
        target=args.target,
        seed=args.seed,
        k=args.k,
        models_dir=args.models_dir,
    )
    print(f"registered {args.name!r} -> {meta['model_path']}")
    print(json.dumps(meta["metrics"], indent=2))
    print(f"models dir: {args.models_dir or MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
