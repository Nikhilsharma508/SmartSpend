from app.api.v1 import overview as overview_module
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_overview_serves_the_live_payload(monkeypatch) -> None:
    """The route is a thin wrapper over build_overview (the expensive part).

    Monkeypatched so the wiring is tested without paying the ~20 s policy-tree
    fit; the numbers themselves are covered by the decision-layer tests.
    """
    payload = {
        "meta": {"generated": "2026-10-20", "generator": "test"},
        "context": {"n_test": 8523},
        "profit_curve": {"points": []},
    }
    monkeypatch.setattr(overview_module, "build_overview", lambda: payload)

    r = client.get("/api/v1/overview")

    assert r.status_code == 200
    assert r.json() == payload


def test_recorded_fixtures_endpoint_is_gone() -> None:
    """The app must never fall back to a recorded snapshot."""
    assert client.get("/api/fixtures").status_code == 404
