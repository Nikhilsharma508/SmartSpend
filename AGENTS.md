# AGENTS.md

SmartSpend (working title UpliftIQ) — uplift/CATE modelling, a budget optimizer, and a
grounded GenAI campaign copilot. The installed distribution name is **`smartspend`**
(`pyproject.toml`), which is also the name used in prose and docs. It is also not an
importable package: `import smartspend` fails. Code lives in `packages/`.

`ROADMAP.md` and `PROJECT.md` at the root are **planning inputs**, and both are partly
stale (see "Decisions that override the docs"). Treat this file and the code as current.

## Commands

Run everything from the repo root with `uv`.

```bash
uv sync                                        # create .venv, install (CI uses --frozen)
uv run python pipelines/download.py            # fetch raw CSV if missing, then verify
                                               #   --verify-only: no network; --force: re-download
uv run ruff check .                            # lint (line-length 100)
uv run mypy packages pipelines                 # typecheck (scope is packages + pipelines only)
uv run pytest                                  # all tests (skip, not fail, if raw CSV absent;
                                               #   first run ~35s, warm runs ~2s)
uv run pytest tests/test_hillstrom.py::test_raw_shape_and_columns   # single test
uv run jupyter lab                             # open research/notebooks/
uv add <pkg>        /  uv add --dev <pkg>      # change deps — never hand-edit pyproject deps
```

Order matters in CI: `ruff check .` → `mypy packages pipelines` → `pytest`.

Root npm automation (mise shims on PATH first):

```bash
npm run setup                     # idempotent end-to-end bootstrap: mise, uv sync,
                                  #   pnpm install, .env, data, champion, alembic, MLflow, smoke
npm run start                     # FastAPI :8000 + web :5173 (see below)
npm run mlflow                    # MLflow UI :5000 (MLFLOW_PORT overrides); seeds the champion
npm run delete                    # destructive reset; keeps .env + lockfiles. --dry-run / --yes
```

## Frontend (apps/web)

Node toolchain comes from **mise** (pinned in `mise.toml`: node 22, pnpm 10). Most
commands need the shims on PATH:

```bash
export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"
npm run start                      # boots everything: FastAPI :8000 then web :5173
pnpm -C apps/web dev               # vite only on :5173 (proxies /api and /health → :8000)
pnpm -C apps/web lint && pnpm -C apps/web typecheck \
  && pnpm -C apps/web test         # web quality gates before any commit (then build)
```

- **Ports are strict and self-healing.** `npm run start` is safe to re-run: it frees a
  stale SmartSpend listener left by a suspended/killed run (a stopped uvicorn still
  holds `:8000` but never answers, which looks like a proxy timeout), reuses a healthy
  stack that is already up, and refuses a port held by any other program. Vite runs
  with `strictPort` so it never silently hops to 5174/5175, and the API is health-checked
  before the web server starts.

- **The UI is live, with no fallback data.** Pages read `/api/v1/overview` (computed
  per process from the registered champion by the Day-4 tools) and the copilot posts
  `/api/v1/copilot`; the header badge polls `/health` and says `api offline` rather
  than showing stale numbers. Never hand-invent a UI number. `apps/web/tests/overview.json`
  is a unit-test snapshot only — the shipped app never reads it. The first request after
  boot waits for the ~20 s DRPolicyTree fit, which the API warms in a background thread.
- `src/components/kokonutui/`, `src/components/charts/` (Bklit) and
  `src/components/shimmering-text.tsx` are registry-vendored and lint-relaxed in
  `eslint.config.js`. `charts/profit-curve-chart.tsx` and
  `charts/segment-uplift-chart.tsx` are ours — keep them fully linted.
- The shadcn CLI reads the **root** `tsconfig.json` for the `@/*` alias (hand-added
  there); pin the CLI (`pnpm dlx shadcn@4.21.0 add … -y`) and run it from `apps/web`.
  Registries: `@kokonutui/*`, `@bklit/*` via `components.json`.
- The live app wiring, the copilot question taxonomy and the LLM architecture panel
  are defined in `docs/concepts.md` — keep that file in sync with the UI the same way
  as notebooks.

## Git

- **Notebooks use `nbstripout` as a repo-local git filter** (`.git/info/attributes` + local
  git config). Outputs and environment metadata are stripped on commit, so never commit
  outputs and never clear them by hand. `nbstripout --install` drops the `--extra-keys`
  flag, so if the filter is ever reinstalled re-set the clean command to
  `"<repo>/.venv/bin/python3" -m nbstripout --extra-keys "metadata.kernelspec metadata.language_info"`.
- **After executing a notebook, `git status` lists it as modified while `git diff` is
  empty.** That is expected, not a bug to chase.

## Where things live

- **Hillstrom dataset facts** (columns, quirks, checksum, leakage rules) live in
  [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md) — read it instead of
  rediscovering the data. The loader that encodes those facts (`FEATURES`,
  `LEAKY_COLUMNS`, `RAW_SHA256`) is [`packages/datasets/hillstrom.py`](packages/datasets/hillstrom.py);
  use it rather than `sklift.datasets.fetch_hillstrom`. Its Appendix A holds the original
  challenge prose verbatim — do not re-wrap that block.
- **Figures**: every plot a notebook produces goes through
  [`packages/plotting.py`](packages/plotting.py) `save_fig(fig, name)` into
  `docs/figures/<name>.png` (committed), named in snake_case so it reads as a filename.
  Read them straight from that folder to review results; notebooks link back to each file.
- **Concepts and every measured number** live in [`docs/concepts.md`](docs/concepts.md).
- **Profit assumptions** live in [`configs/business_assumptions.yaml`](configs/business_assumptions.yaml)
  (single source; the copilot reads it) and mirror into the README profit table.
- **Python 3.11 is pinned** (`.python-version`). Do not use the system `python3` (3.13):
  `causalml` and `econml` have no 3.13 support, which is why the pin exists.
- **`packages/` is a plain directory tree, not an installed distribution.** `import
  packages.*` only resolves when the CWD is the repo root (pytest sets `pythonpath`, the
  notebook bootstraps `sys.path`). Running a script from elsewhere fails with
  `ModuleNotFoundError: No module named 'packages'`.
- **Most of the skeleton is real now.** Real code: `packages/datasets/`,
  `packages/causal/diagnostics.py`, `packages/decision/`, `packages/copilot/`,
  `packages/tracking/`, `packages/plotting.py`, `pipelines/download.py`, `tests/`,
  `evals/`, `apps/api` (FastAPI + SQLite + MLflow) and `apps/web` (live Day-6
  frontend). The rest of `packages/causal/` is still a placeholder. Do not assume
  planned modules exist.
- `data/raw/` and `data/processed/` are git-ignored (`.gitkeep` only); the raw CSV is
  fetched and checksum-verified by `pipelines/download.py`.

## Decisions that override the docs

- **Profit is reported in USD only.** Ignore the ₹/INR framing throughout `PROJECT.md`;
  there is no conversion step.
- **Day 6 UI is FastAPI + React** (as in `PROJECT.md`). `ROADMAP.md`'s "Streamlit campaign
  planner" is superseded.
- **Profit assumptions are decided** (30% margin, $0.10/contact, value per
  conversion $34.91) — see `configs/business_assumptions.yaml`; change only with a
  recorded reason.
- Scikit-uplift's PyPI name is `scikit-uplift`; the import name is `sklift`.
- **No containers in this build.** Docker/Compose are dropped (personal project);
  the one-command local stack is `npm run start` (`scripts/start_app.sh`).
- **App storage is SQLite** (`data/app/`, git-ignored) through SQLAlchemy +
  Alembic — chosen over the documented PostgreSQL to keep the local run
  sign-in-free, with the Postgres swap remaining a one-URL change (PostgreSQL 18
  is the documented production option). No auth on local services.
- **MLflow tracks locally in SQLite only** (MLflow 3.17, `mlflow.db`): the champion's
  params/metrics, and one GenAI trace per `/api/v1/copilot` request with child spans for
  gate, plan, each Day-4 tool, write and validate. No artifacts are logged — the joblib
  and provenance sidecar in `models/` remain the load-path code of record. `npm run mlflow`
  serves the store; the UI's **GenAI → Traces** view is the copilot audit view.

## Conventions

- Notebooks are research only; anything reused moves into `packages/`.
- **Keep `docs/concepts.md` in sync.** When a notebook or script introduces a term,
  a metric or a result, add it to that file in the same change — definitions in the
  numbered sections, values in the findings ledger. It is the canonical concept +
  numbers doc; notebooks only explain briefly and link back.
- `research/notebooks/` is split by concern: `00_hillstrom_eda` is descriptive only,
  `01_hillstrom_randomization_ate` introduces the causal vocabulary (arm, SMD, ATE).
  Keep causal concepts out of `00`.
- ruff ignores `E402`/`E401`/`I001` in `*.ipynb` because cells bootstrap `sys.path` before
  importing project packages — do not "fix" that ordering.
- mypy runs with `ignore_missing_imports = true` and `files = ["packages", "pipelines"]`;
  tests and notebooks are intentionally unchecked.
