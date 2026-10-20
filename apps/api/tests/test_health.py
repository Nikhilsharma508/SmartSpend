from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_importable_as_uvicorn_module() -> None:
    """`npm run start` loads `apps.api.app.main:app` from the repo root.

    Inside that package every intra-app import must be relative: an absolute
    `from app...` works under pytest's `apps/api` pythonpath but fails for
    uvicorn (ModuleNotFoundError: No module named 'app').
    """
    from apps.api.app.main import app as uvicorn_app

    assert uvicorn_app.title
