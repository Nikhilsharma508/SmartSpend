#!/usr/bin/env bash
# SmartSpend one-command runbook: `npm run start` (or `bash scripts/start_app.sh`).
# Starts the FastAPI service when it exists (apps/api/app/main.py) plus the web
# planner. Until apps/api lands, this runs the web alone in offline playback
# mode — that is the honest default, not a degraded one.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"

# --- node toolchain (mise-pinned; see mise.toml) ---
if ! command -v node >/dev/null 2>&1; then
  echo "==> node not found — installing via mise (mise.toml pins node 22 / pnpm 10)"
  mise install >/dev/null 2>&1 || true
fi
if ! command -v node >/dev/null 2>&1; then
  echo "error: node is not on PATH even after 'mise install'" >&2
  echo "       try: export PATH=\"\$HOME/.local/bin:\$HOME/.local/share/mise/shims:\$PATH\"" >&2
  exit 1
fi

# --- web deps ---
if [ ! -d apps/web/node_modules ]; then
  echo "==> installing web deps (first run)"
  pnpm -C apps/web install
fi

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# --- backend, only once it exists ---
if [ -f apps/api/app/main.py ]; then
  echo "==> starting FastAPI on :8000"
  command -v uv >/dev/null 2>&1 || { echo "error: uv is not on PATH (~/.local/bin); the backend needs it" >&2; exit 1; }
  [ -d .venv ] || uv sync
  uv run uvicorn apps.api.app.main:app --port 8000 --reload &
  PIDS+=($!)
  echo "    API docs → http://localhost:8000/docs"
else
  echo "==> apps/api not built yet — web runs in offline playback mode (fixtures only)"
fi

echo "==> starting web on :5173 → http://localhost:5173"
pnpm -C apps/web dev &
PIDS+=($!)

wait
