#!/usr/bin/env bash
# Open the local MLflow UI for SmartSpend's tracking store.
#
# The API/pipelines read MLFLOW_TRACKING_URI (default: sqlite:///data/mlflow/mlflow.db),
# so this script resolves the same setting instead of guessing. The champion is
# backfilled first, which makes a fresh store show something useful.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"

export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p data/mlflow
PORT="${MLFLOW_PORT:-5000}"
URL="http://127.0.0.1:$PORT"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -d .venv ] || fail "no .venv — run 'npm run setup' first"
command -v uv >/dev/null 2>&1 || fail "uv is not on PATH"

if [ -n "$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)" ]; then
  fail "port $PORT is already in use (set MLFLOW_PORT to another port)"
fi

# Ask the app for its configured store, so .env edits are honoured.
URI="$(uv run python -c \
  'from apps.api.app.core.config import settings; print(settings.mlflow_tracking_uri)' \
  2>/dev/null || true)"
[ -n "$URI" ] || URI="sqlite:///data/mlflow/mlflow.db"

# MLflow stores are repo-relative; hand the CLI absolute URIs.
if [ "${URI#file:}" != "$URI" ]; then
  store="${URI#file:}"
  if [ "${store#/}" = "$store" ]; then
    store="$REPO_ROOT/${store#./}"
  fi
  URI="file://$store"
elif [ "${URI#sqlite:///}" != "$URI" ]; then
  store="${URI#sqlite:///}"
  if [ "${store#/}" = "$store" ]; then
    URI="sqlite:///$REPO_ROOT/$store"
  fi
fi

echo "==> tracking store: $URI"
echo "==> seeding the champion (no-op once logged)"
uv run python pipelines/log_mlflow.py || echo "    warning: backfill skipped"

echo "==> starting MLflow UI on $URL"
uv run mlflow ui --backend-store-uri "$URI" --host 127.0.0.1 --port "$PORT" \
  >"${TMPDIR:-/tmp}/smartspend-mlflow.log" 2>&1 &
UI_PID=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$UI_PID" 2>/dev/null || true
  wait "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 40); do
  if curl -sf --max-time 1 "$URL" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    echo "error: MLflow exited during startup:" >&2
    tail -20 "${TMPDIR:-/tmp}/smartspend-mlflow.log" >&2
    exit 1
  fi
  sleep 0.5
done

if ! curl -sf --max-time 2 "$URL" >/dev/null 2>&1; then
  echo "error: MLflow did not become ready; log:" >&2
  tail -20 "${TMPDIR:-/tmp}/smartspend-mlflow.log" >&2
  exit 1
fi

if [ "${MLFLOW_NO_OPEN:-0}" != 1 ]; then
  if command -v open >/dev/null 2>&1; then
    open "$URL" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  fi
fi
echo "    $URL  (Ctrl+C to stop)"

wait "$UI_PID"
