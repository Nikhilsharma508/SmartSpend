#!/usr/bin/env bash
# Full setup from a fresh clone (or after `npm run delete`):
#   tools → Python env → web deps → .env → data → champion → database → MLflow.
#
# Idempotent: existing artifacts are verified/skipped, never re-downloaded or
# re-trained. Run it again any time; it is also the repair path.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

step "checking tools"
command -v uv >/dev/null 2>&1 \
  || fail "uv is not on PATH (install: https://docs.astral.sh/uv/)"
if command -v mise >/dev/null 2>&1; then
  # Non-interactive first run; both are safe to repeat.
  mise trust >/dev/null 2>&1 || true
  mise install
fi
command -v node >/dev/null 2>&1 \
  || fail "node is not on PATH (install mise, or node 22)"
command -v pnpm >/dev/null 2>&1 \
  || fail "pnpm is not on PATH (install mise, or pnpm 10)"
printf '    node %s · pnpm %s · %s\n' "$(node -v)" "$(pnpm -v)" "$(uv --version)"

step "Python environment — uv sync"
uv sync

step "web dependencies — pnpm install"
pnpm -C apps/web install --frozen-lockfile

step "environment file"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    created .env from .env.example"
  echo "    fill in AZURE_*/NEUROMETRIC_* to enable the live gate and generative writer;"
  echo "    without them the copilot falls back to the keyword router + template writer."
else
  echo "    .env present — left untouched"
fi

step "raw Hillstrom data — download if missing, then verify the checksum"
uv run python pipelines/download.py

step "champion model"
if [ -f models/champion.joblib ] && [ -f models/champion.meta.json ]; then
  echo "    models/champion.joblib already registered — skipping training"
else
  echo "    no registered champion — training the response baseline (seed 42)"
  uv run python pipelines/train.py
fi

step "app database — alembic upgrade head"
uv run alembic -c apps/api/alembic.ini upgrade head

step "MLflow — backfill the champion (idempotent)"
uv run python pipelines/log_mlflow.py || echo "    warning: MLflow backfill skipped"

step "smoke checks"
uv run python - <<'PY'
from packages.causal.registry import load_model

_, meta = load_model("champion")
print(f"    champion loads: {meta['learner']} on {meta['target']} (seed {meta['random_state']})")
PY
uv run python -c "from apps.api.app.main import app; print(f'    API imports: {app.title}')"

printf '\nSetup complete.\n'
printf '  npm run start    # FastAPI :8000 + web planner :5173\n'
printf '  npm run mlflow   # MLflow UI :5000\n'
