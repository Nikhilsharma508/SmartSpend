#!/usr/bin/env bash
# Wipe every generated artifact so `npm run setup` can rebuild from zero.
#
# Removes the Python env, web deps, data, model, database, MLflow store
# and caches. Keeps secrets and pins: .env, .env.example, uv.lock, pnpm-lock.
#
#   npm run delete              # asks for confirmation
#   npm run delete -- --yes     # no prompt (CI / deliberate reset)
#   npm run delete -- --dry-run # show what would go, remove nothing
set -euo pipefail
cd "$(dirname "$0")/.."
ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -y | --yes) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h | --help)
      sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf 'error: unknown option %s\n' "$arg" >&2
      exit 2
      ;;
  esac
done

# Directories and files that setup recreates.
TARGETS=(
  .venv
  apps/web/node_modules
  apps/web/dist
  apps/web/coverage
  apps/web/.eslintcache
  models
  mlflow.db
  mlflow.db-shm
  mlflow.db-wal
  mlruns
  mlartifacts
  data/app
  .pytest_cache
  .ruff_cache
  .mypy_cache
  .coverage
  htmlcov
)

size_of() {
  du -sh "$1" 2>/dev/null | awk '{print $1}'
}

printf 'This permanently removes:\n'
for target in "${TARGETS[@]}"; do
  if [ -e "$target" ]; then
    printf '  %-28s %s\n' "$target" "$(size_of "$target")"
  fi
done
for dir in data/raw data/processed; do
  if [ -d "$dir" ]; then
    printf '  %-28s %s\n' "$dir/*" "$(size_of "$dir")"
  fi
done

# Loose caches anywhere in the tree (never .git).
CACHES=()
while IFS= read -r -d '' cache; do
  CACHES+=("$cache")
done < <(
  find . -path ./.git -prune -o -type d \
    \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \
    -o -name .ruff_cache -o -name .mypy_cache -o -name .ipynb_checkpoints \) \
    -print0 2>/dev/null
)
if [ "${#CACHES[@]}" -gt 0 ]; then
  printf '  %-28s %s dirs\n' "__pycache__ / caches" "${#CACHES[@]}"
fi

printf '\nKept: .env (secrets), .env.example, uv.lock, pnpm-lock.yaml, source.\n'

if [ "$DRY_RUN" = 1 ]; then
  printf '\n(dry run) nothing removed.\n'
  exit 0
fi

# Never pull the rug out from under a live stack.
BUSY=()
for port in 8000 5173 5000; do
  if [ -n "$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)" ]; then
    BUSY+=("$port")
  fi
done
if [ "${#BUSY[@]}" -gt 0 ]; then
  printf '\nerror: SmartSpend is still running on port(s) %s.\n' "${BUSY[*]}" >&2
  printf 'Stop it with Ctrl+C (npm run start), then run this again.\n' >&2
  exit 1
fi

if [ "$ASSUME_YES" != 1 ]; then
  printf "\nType 'delete' to continue: "
  read -r answer
  [ "$answer" = "delete" ] || {
    printf 'aborted.\n'
    exit 1
  }
fi

for target in "${TARGETS[@]}"; do
  rm -rf -- "$target"
done
rm -rf -- data/raw/* data/processed/*
for cache in "${CACHES[@]:-}"; do
  [ -n "$cache" ] && rm -rf -- "$cache"
done

printf '\nDeleted. Rebuild everything with: npm run setup\n'
