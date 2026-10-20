"""API surface tests for the model lab and per-model overview gating.

Runs the FastAPI app without its lifespan (no warm thread, no MLflow writes)
and monkeypatches the registry lookups so no trained artifact is required —
the shapes and the 404/422 contract are what these tests pin down.
"""

from __future__ import annotations

from app.api.v1 import models as models_mod
from app.api.v1 import overview as overview_module
from app.main import app
from fastapi.testclient import TestClient

from packages.causal import catalog

client = TestClient(app)


def _fake_registered(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(models_mod, "is_registered", lambda name, models_dir=None: name in names)
    monkeypatch.setattr(
        models_mod,
        "load_meta",
        lambda name, models_dir=None: {
            "created_at": "2026-10-20T12:00:00",
            "metrics": {"qini_auc": 0.01, "profit_at_k_usd": 100.0},
        },
    )


# --- /api/v1/models ----------------------------------------------------------


def test_models_lists_whole_catalog_with_mode(monkeypatch):
    _fake_registered(monkeypatch, ["champion"])
    body = client.get("/api/v1/models").json()
    assert body["default"] == "champion"
    assert [m["name"] for m in body["models"]] == [s.name for s in catalog.MODEL_SPECS]
    listed = {m["name"]: m for m in body["models"]}
    assert listed["champion"]["registered"] is True
    assert listed["dr-learner"]["registered"] is False
    assert listed["s-learner"]["selectable"] is False  # research-locked
    assert "response_probability" in body["score_meanings"]


def test_models_research_mode_flips_selectable(monkeypatch):
    _fake_registered(monkeypatch, ["champion"])
    body = client.get("/api/v1/models?research_mode=true").json()
    assert body["research_mode"] is True
    listed = {m["name"]: m for m in body["models"]}
    assert listed["t-learner"]["selectable"] is True
    assert listed["t-learner"]["added_to_allow_list_by_mode"] is True
    assert listed["champion"]["selectable"] is True


# --- /api/v1/models/{name} ---------------------------------------------------


def test_model_detail_returns_card(monkeypatch):
    _fake_registered(monkeypatch, ["champion"])

    class _Card:
        def model_dump(self):
            return {"name": "champion", "learner": "response"}

    monkeypatch.setattr(models_mod, "get_model_card", lambda name: _Card())
    r = client.get("/api/v1/models/champion")
    assert r.status_code == 200
    assert r.json()["name"] == "champion"


def test_model_detail_unknown_name_404(monkeypatch):
    _fake_registered(monkeypatch, [])
    r = client.get("/api/v1/models/not-a-model")
    assert r.status_code == 404
    assert "unknown model" in r.json()["detail"]


# --- the gating contract on the overview route -------------------------------


def test_overview_research_only_model_blocked_in_normal_mode():
    r = client.get("/api/v1/overview?model=s-learner")
    assert r.status_code == 422
    assert "research-only" in r.json()["detail"]


def test_overview_unknown_model_404():
    r = client.get("/api/v1/overview?model=nonsense")
    assert r.status_code == 404
    assert "unknown model" in r.json()["detail"]


def test_segments_routes_share_the_same_contract():
    assert client.get("/api/v1/segments/profile?model=nonsense").status_code == 404
    assert client.get("/api/v1/segments/model-card?model=s-learner").status_code == 422


# --- overview wiring still passes the model through --------------------------


def test_overview_route_passes_model_to_build(monkeypatch):
    seen: dict[str, str | bool] = {}

    def fake_build(model: str) -> dict:
        seen["model"] = model
        return {"meta": {}, "context": {"model_name": model}}

    monkeypatch.setattr(overview_module, "build_overview", fake_build)
    r = client.get("/api/v1/overview?model=dr-learner")
    assert r.status_code == 200
    assert seen["model"] == "dr-learner"
    assert r.json()["context"]["model_name"] == "dr-learner"
