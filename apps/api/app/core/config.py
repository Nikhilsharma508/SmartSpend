"""Application settings, loaded from environment / .env."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repo root (apps/api/app/core/config.py → up four).
ROOT = Path(__file__).resolve().parents[4]
# The copilot gate reads AZURE_* straight from the environment (the evals do the
# same); pydantic-settings alone would leave os.environ untouched.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "SmartSpend API"
    debug: bool = False
    # SQLite by default (no sign-in, single file, WAL-friendly). Swap to a
    # PostgreSQL URL without code changes.
    database_url: str = "sqlite:///data/app/smartspend.db"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    champion_path: str = "models/champion.joblib"
    champion_sidecar: str = "models/champion.provenance.json"


settings = Settings()

# Resolve relative SQLite paths against the repo root so the same DB is used
# regardless of the process's working directory (uvicorn runs from repo root,
# alembic from apps/api).
_db_url = settings.database_url
if _db_url.startswith("sqlite:///") and not _db_url.startswith("sqlite:////"):
    _rel = _db_url.removeprefix("sqlite:///")
    _abs = (Path(__file__).resolve().parents[4] / _rel).resolve()
    _abs.parent.mkdir(parents=True, exist_ok=True)
    settings.database_url = f"sqlite:///{_abs}"