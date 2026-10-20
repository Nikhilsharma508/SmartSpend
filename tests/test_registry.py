"""Tests for the model registry in ``packages/causal/registry.py``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from packages.causal.learners import make_learner
from packages.causal.registry import (
    REQUIRED_META_FIELDS,
    is_registered,
    list_models,
    load_model,
    save_model,
)


@pytest.fixture()
def fitted_model():
    """A tiny fitted response baseline — LightGBM on 300 rows, ~1s."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    t = pd.Series(rng.integers(0, 2, size=300))
    y = pd.Series(rng.integers(0, 2, size=300))
    model = make_learner("response", random_state=0)
    model.fit(X.to_numpy(), t.to_numpy(), y.to_numpy())
    return model


@pytest.fixture()
def valid_meta():
    return {f: "x" for f in REQUIRED_META_FIELDS}


def test_round_trip_preserves_predictions(tmp_path, fitted_model, valid_meta):
    X_new = pd.DataFrame(
        np.random.default_rng(1).normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)]
    )
    before = fitted_model.predict(X_new.to_numpy())

    save_model(fitted_model, "champion", meta=valid_meta, models_dir=tmp_path)
    loaded, meta = load_model("champion", models_dir=tmp_path)

    np.testing.assert_allclose(loaded.predict(X_new.to_numpy()), before)
    assert meta == valid_meta
    assert (tmp_path / "champion.joblib").exists()
    assert json.loads((tmp_path / "champion.meta.json").read_text()) == valid_meta


def test_save_rejects_incomplete_metadata(tmp_path, fitted_model, valid_meta):
    del valid_meta["data_sha256"]
    with pytest.raises(ValueError, match="data_sha256"):
        save_model(fitted_model, "m", meta=valid_meta, models_dir=tmp_path)


def test_save_rejects_unsafe_name(tmp_path, fitted_model, valid_meta):
    for bad in ("../escape", "a/b", "", ".hidden"):
        with pytest.raises(ValueError, match="invalid model name"):
            save_model(fitted_model, bad, meta=valid_meta, models_dir=tmp_path)


def test_load_missing_model_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipelines/train.py"):
        load_model("nope", models_dir=tmp_path)


def test_load_rejects_model_without_metadata(tmp_path, fitted_model, valid_meta):
    save_model(fitted_model, "half", meta=valid_meta, models_dir=tmp_path)
    (tmp_path / "half.meta.json").unlink()
    with pytest.raises(ValueError, match="no metadata sidecar"):
        load_model("half", models_dir=tmp_path)


def test_load_rejects_metadata_missing_fields(tmp_path, fitted_model, valid_meta):
    save_model(fitted_model, "bad", meta=valid_meta, models_dir=tmp_path)
    broken = {k: v for k, v in valid_meta.items() if k != "metrics"}
    (tmp_path / "bad.meta.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="metrics"):
        load_model("bad", models_dir=tmp_path)


def test_load_rejects_metadata_missing_semantics_fields(tmp_path, fitted_model, valid_meta):
    """The five catalog semantics fields are as required as the data hash.

    An artifact trained before the multi-model era (tier/score_kind/… absent)
    cannot serve: the copilot must know what a score means, never guess.
    ``label`` is deliberately optional — the catalog supplies a display label.
    """
    for field in ("tier", "score_kind", "supports_intervals", "research_only", "use_case"):
        save_model(fitted_model, "legacy", meta=valid_meta, models_dir=tmp_path)
        legacy = {k: v for k, v in valid_meta.items() if k != field}
        (tmp_path / "legacy.meta.json").write_text(json.dumps(legacy))
        with pytest.raises(ValueError, match=field):
            load_model("legacy", models_dir=tmp_path)


def test_is_registered_matches_list_models(tmp_path, fitted_model, valid_meta):
    assert list_models(tmp_path) == []
    save_model(fitted_model, "half", meta=valid_meta, models_dir=tmp_path)
    (tmp_path / "half.joblib").unlink()
    assert not is_registered("half", models_dir=tmp_path)  # sidecar only -> no
    save_model(fitted_model, "half", meta=valid_meta, models_dir=tmp_path)
    (tmp_path / "half.meta.json").unlink()
    assert not is_registered("half", models_dir=tmp_path)  # joblib only -> no
    save_model(fitted_model, "whole", meta=valid_meta, models_dir=tmp_path)
    assert is_registered("whole", models_dir=tmp_path)


def test_list_models_requires_both_halves(tmp_path, fitted_model, valid_meta):
    assert list_models(tmp_path) == []
    save_model(fitted_model, "b-model", meta=valid_meta, models_dir=tmp_path)
    save_model(fitted_model, "a-model", meta=valid_meta, models_dir=tmp_path)
    (tmp_path / "orphan.joblib").write_bytes(b"")  # no sidecar -> not listed
    assert list_models(tmp_path) == ["a-model", "b-model"]
