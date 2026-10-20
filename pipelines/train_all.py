"""Train and register every model in the catalog — all six benchmark learners.

    uv run python pipelines/train_all.py                 # train whatever is missing
    uv run python pipelines/train_all.py --force         # retrain all six
    uv run python pipelines/train_all.py --only dr-learner

The catalog (``packages/causal/catalog.py``) is the single source of truth for
artifact names, learners and roles:

    champion         response  (champion)   P(visit | emailed)
    dr-learner       dr        (challenger) estimated incremental effect
    s-learner        s         (research)   relative uplift score
    t-learner        t         (research)   relative uplift score
    x-learner        x         (research)   relative uplift score
    causal-forest    forest    (research)   effect + interval

All six share the same protocol — seed 42, target ``visit``, k = 0.30 — so
their seed-42 metrics in the sidecars compare cleanly on one split, exactly
like the Day-3 10-seed benchmark they extend. Skipping existing artifacts is
the default: registering a model twice means the provenance question "which
artifact produced this number?" has two answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.causal.catalog import MODEL_SPECS  # noqa: E402
from packages.causal.registry import MODELS_DIR, is_registered  # noqa: E402
from pipelines.train import train_and_register  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even when the artifact is already registered",
    )
    parser.add_argument(
        "--only",
        metavar="MODEL",
        default=None,
        help=(
            "train a single artifact: one of "
            f"{[s.name for s in MODEL_SPECS]} (also accepts a learner name)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="split + fit seed")
    parser.add_argument("--k", type=float, default=0.3, help="targeting depth for metrics")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="override the models/ location",
    )
    args = parser.parse_args(argv)

    selected = list(MODEL_SPECS)
    if args.only is not None:
        by_name = next((s for s in MODEL_SPECS if s.name == args.only), None)
        by_learner = next((s for s in MODEL_SPECS if s.learner == args.only), None)
        spec = by_name or by_learner
        if spec is None:
            print(
                f"error: --only {args.only!r} matches no catalog entry; expected one of "
                f"{[s.name for s in MODEL_SPECS]}",
                file=sys.stderr,
            )
            return 2
        selected = [spec]

    trained: list[dict] = []
    skipped: list[str] = []
    for spec in selected:
        if not args.force and is_registered(spec.name, models_dir=args.models_dir):
            skipped.append(spec.name)
            print(f"skip {spec.name!r} (already registered; use --force to retrain)")
            continue
        print(f"training {spec.name!r} ({spec.learner}, tier: {spec.tier})…")
        meta = train_and_register(
            name=spec.name,
            learner=spec.learner,  # type: ignore[arg-type]
            target="visit",
            seed=args.seed,
            k=args.k,
            models_dir=args.models_dir,
        )
        trained.append(meta)

    print(f"\nmodels dir: {args.models_dir or MODELS_DIR}")
    if trained:
        print("\n| artifact | learner | tier | qini | dconv@k | profit@k (USD) |")
        print("| --- | --- | --- | ---: | ---: | ---: |")
        for meta in trained:
            metrics = meta["metrics"]
            print(
                f"| {meta.get('label', meta['learner'])} | {meta['learner']} "
                f"| {meta.get('tier', '?')} | {metrics['qini_auc']:.4f} | "
                f"{metrics['incremental_conversions_at_k']:.2f} "
                f"| {metrics['profit_at_k_usd']:.2f} |"
            )
        print(json.dumps({"trained": [m["label"] for m in trained], "skipped": skipped}))
    elif skipped:
        print(f"all {len(skipped)} catalog artifacts already registered — nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
