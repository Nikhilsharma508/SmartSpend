#!/usr/bin/env bash
# SmartSpend one-command runbook: `npm run start` (or `bash scripts/start_app.sh`).
# Starts the FastAPI service when it exists (apps/api/app/main.py) plus the web
# planner. Until apps/api lands, this runs the web alone in offline playback
# mode — that is the honest default, not a degraded one.
#
# Port discipline: a previous run that was suspended (Ctrl+Z) or killed badly
# can leave a stopped uvicorn/vite holding a port with a socket that never
# answers. This script frees stale *SmartSpend* listeners (and reuses a healthy
# running API), but refuses to touch a port held by anything else.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

# --- helpers ---------------------------------------------------------------
port_pids() { lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true; }

is_ours() {
  local pid="$1" cmd cwd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [ -n "$cmd" ] || return 1
  case "$cmd" in
    *"$REPO_ROOT"*) return 0 ;;
    *"apps.api.app.main"*) return 0 ;;
  esac
  # pnpm/vite launch with a repo-relative binary path, so fall back to cwd.
  cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)"
  case "$cwd" in
    "$REPO_ROOT" | "$REPO_ROOT"/*) return 0 ;;
  esac
  return 1
}

kill_tree() {
  local pid="$1" sig="${2:-TERM}" child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null || true
}

# Free PORT if every listener belongs to this repo; otherwise abort loudly.
free_port() {
  local port="$1" label="$2" pids pid foreign=""
  pids="$(port_pids "$port")"
  [ -n "$pids" ] || return 0

  for pid in $pids; do
    is_ours "$pid" || foreign="$foreign $pid"
  done
  if [ -n "$foreign" ]; then
    echo "error: :$port ($label) is held by another program — stop it and retry:" >&2
    for pid in $foreign; do
      printf '       pid %s  %s\n' "$pid" "$(ps -p "$pid" -o command= 2>/dev/null | cut -c1-120)" >&2
    done
    exit 1
  fi

  echo "==> freeing stale $label on :$port (pid $(echo $pids | tr '\n' ' '))"
  for pid in $pids; do kill_tree "$pid" TERM; done
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    [ -z "$(port_pids "$port")" ] && return 0
    sleep 0.3
  done
  for pid in $(port_pids "$port"); do kill_tree "$pid" KILL; done
  sleep 0.3
  if [ -n "$(port_pids "$port")" ]; then
    echo "error: could not free :$port" >&2
    exit 1
  fi
}

api_healthy() { curl -fsS --max-time 1 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; }
web_healthy() { curl -fsS --max-time 1 "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1; }

wait_for_api() {
  local pid="$1" tries="${2:-100}" i
  for ((i = 0; i < tries; i++)); do
    kill -0 "$pid" 2>/dev/null || return 1
    api_healthy && return 0
    sleep 0.3
  done
  return 1
}

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
API_OWNED=0
cleanup() {
  trap - EXIT INT TERM
  local pid i pids
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill_tree "$pid" TERM
  done
  # Children (vite, the uvicorn worker) can outlive the shell we started for a
  # moment. Wait briefly, then make sure the ports we own are actually free so
  # the next `npm run start` never sees a half-dead listener.
  for i in 1 2 3 4 5 6; do
    pids="$(port_pids "$WEB_PORT")"
    [ "$API_OWNED" = 1 ] && pids="$pids $(port_pids "$API_PORT")"
    [ -z "${pids// /}" ] && return 0
    sleep 0.3
  done
  for pid in $pids; do
    is_ours "$pid" && kill_tree "$pid" KILL
  done
  return 0
}
trap cleanup EXIT INT TERM

# --- preflight: one owner per port, no silent hopping ----------------------
BACKEND=0
if [ -f apps/api/app/main.py ]; then
  BACKEND=1
  command -v uv >/dev/null 2>&1 || { echo "error: uv is not on PATH (~/.local/bin); the backend needs it" >&2; exit 1; }
  [ -d .venv ] || uv sync

  if [ -n "$(port_pids "$API_PORT")" ] && api_healthy; then
    REUSE=1
    for pid in $(port_pids "$API_PORT"); do is_ours "$pid" || REUSE=0; done
    if [ "$REUSE" = 1 ]; then
      echo "==> reusing the healthy API already on :$API_PORT"
      BACKEND=2
    else
      free_port "$API_PORT" "API"
    fi
  else
    free_port "$API_PORT" "API"
  fi
fi

# Whole stack already up and healthy from this repo? Leave it alone.
WEB_UP=0
if [ -n "$(port_pids "$WEB_PORT")" ] && web_healthy; then
  WEB_UP=1
  for pid in $(port_pids "$WEB_PORT"); do is_ours "$pid" || WEB_UP=0; done
fi
if [ "$WEB_UP" = 1 ] && { [ "$BACKEND" = 0 ] || [ "$BACKEND" = 2 ]; }; then
  trap - EXIT INT TERM
  echo "==> SmartSpend is already running: http://localhost:$WEB_PORT (API :$API_PORT)"
  exit 0
fi

free_port "$WEB_PORT" "web"

# --- backend, only once it exists ---
if [ "$BACKEND" = 1 ]; then
  echo "==> starting FastAPI on :$API_PORT"
  uv run uvicorn apps.api.app.main:app --port "$API_PORT" --reload &
  API_PID=$!
  API_OWNED=1
  PIDS+=("$API_PID")
  if ! wait_for_api "$API_PID"; then
    echo "error: the API did not become healthy on :$API_PORT — see the uvicorn output above" >&2
    exit 1
  fi
  echo "    API docs → http://localhost:$API_PORT/docs"
elif [ "$BACKEND" = 0 ]; then
  echo "==> apps/api not built yet — web runs in offline playback mode (fixtures only)"
fi

echo "==> starting web on :$WEB_PORT → http://localhost:$WEB_PORT"
pnpm -C apps/web exec vite --port "$WEB_PORT" --strictPort &
PIDS+=($!)

wait
