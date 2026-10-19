"""Model registry: persist the *finalized* model, never the exploratory fits.

Every model trained so far — notebooks 03/04, the benchmark loop, the test
suite — is in-memory by design: those exist to be *measured*, not served. This
module is the one place a model is written to disk, and it holds exactly one
kind of artifact: the finalized model the pre-registered champion rule picked
and ``docs/concepts.md`` quotes.

Layout (``models/`` is git-ignored; regenerate everything with
``pipelines/train.py``)::

    models/<name>.joblib        the fitted model
    models/<name>.meta.json     provenance: how it was trained and how it scored

The metadata sidecar is what makes the artifact auditable — data checksum,
split protocol, seed, package versions, and the holdout metrics the model
posted under the decided business assumptions. Day 6's API loads from here
and never trains.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib

#: Repo-root ``models/`` directory. Git-ignored: artifacts are regenerable
#: from ``pipelines/train.py`` + the checksum-verified raw data.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

#: Fields every metadata sidecar must carry. A model without provenance is a
#: file nobody can trust, so ``load_model`` refuses to serve one.
REQUIRED_META_FIELDS: tuple[str, ...] = (
    "learner",
    "target",
    "random_state",
    "fit_scope",
    "train_rows",
    "data_sha256",
    "metrics",
    "assumptions",
    "packages",
    "created_at",
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _models_dir(models_dir: Path | None) -> Path:
    return Path(models_dir) if models_dir is not None else MODELS_DIR


def _validate_name(name: str) -> None:
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"invalid model name {name!r}: use letters, digits, '_', '.', '-' "
            "(no path separators — names map to files inside models/)"
        )


def _validate_meta(meta: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_META_FIELDS if f not in meta]
    if missing:
        raise ValueError(
            f"metadata is missing required field(s) {missing}; a model without "
            f"provenance cannot be registered (required: {list(REQUIRED_META_FIELDS)})"
        )


def save_model(
    model: Any,
    name: str,
    *,
    meta: dict[str, Any],
    models_dir: Path | None = None,
) -> Path:
    """Persist ``model`` and its metadata sidecar; return the model path.

    Parameters
    ----------
    model:
        Any fitted estimator joblib can serialize (the registry wraps the
        factory output from :func:`packages.causal.learners.make_learner`).
    name:
        Artifact stem — written as ``<name>.joblib`` + ``<name>.meta.json``.
    meta:
        Provenance dict; must carry every field in
        :data:`REQUIRED_META_FIELDS` or this raises. See ``pipelines/train.py``
        for the canonical construction.
    """
    _validate_name(name)
    _validate_meta(meta)
    directory = _models_dir(models_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / f"{name}.joblib"
    meta_path = directory / f"{name}.meta.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return model_path


def load_model(
    name: str, *, models_dir: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    """Load a registered model and its metadata.

    Raises
    ------
    FileNotFoundError
        No ``<name>.joblib`` under the models directory.
    ValueError
        The metadata sidecar is missing, unparseable, or lacks required
        fields — a model whose provenance cannot be checked is not served.
    """
    _validate_name(name)
    directory = _models_dir(models_dir)
    model_path = directory / f"{name}.joblib"
    meta_path = directory / f"{name}.meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"no registered model {name!r} in {directory} "
            "(train one with `uv run python pipelines/train.py`)"
        )
    if not meta_path.exists():
        raise ValueError(f"model {name!r} has no metadata sidecar at {meta_path}")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as err:  # pragma: no cover - defensive
        raise ValueError(f"metadata for {name!r} is not valid JSON: {err}") from err
    _validate_meta(meta)
    return joblib.load(model_path), meta


def list_models(models_dir: Path | None = None) -> list[str]:
    """Names of every registered model (files with both halves present)."""
    directory = _models_dir(models_dir)
    if not directory.exists():
        return []
    return sorted(
        p.stem
        for p in directory.glob("*.joblib")
        if (directory / f"{p.stem}.meta.json").exists()
    )
