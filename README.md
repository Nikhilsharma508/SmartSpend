# SmartSpend — Causal Promo Targeting with a GenAI Campaign Copilot

Decide **who should get a discount** by estimating each customer's *incremental*
response to it (CATE / uplift), then let a marketer plan a campaign in plain
language through a grounded GenAI copilot.

> Name: **SmartSpend** (working title: UpliftIQ; other candidates: CausalEdge, Incremental Profit Engine).
> Repo directory: `SmartSpend`.

## The problem in one line

Standard response models (click / conversion prediction) rank **"sure things"**
highest — customers who would have bought anyway — so budget is spent on discounts
that change nothing. An uplift model ranks **persuadables** highest instead.

| Group | Buys without promo? | Buys with promo? | Target? |
| --- | --- | --- | --- |
| Persuadables | No | Yes | **Yes** — the only incremental-value group |
| Sure things | Yes | Yes | No — wasted margin |
| Lost causes | No | No | No — wasted contact cost |
| Sleeping dogs | Yes | No | No — the promo causes harm |

## Project status

**Days 1–5 complete.**

Day 1 — data, scaffold, environment, experiment validation.

- [x] Repo skeleton, `uv` environment with pinned versions
- [x] Structured dataset spec — [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md)
- [x] Concept reference — [`docs/concepts.md`](docs/concepts.md)
- [x] Hillstrom loader (`packages/datasets/hillstrom.py`) + experiment diagnostics
- [x] EDA notebook — [`research/notebooks/00_hillstrom_eda.ipynb`](research/notebooks/00_hillstrom_eda.ipynb)
- [x] Randomization + ATE notebook — [`research/notebooks/01_hillstrom_randomization_ate.ipynb`](research/notebooks/01_hillstrom_randomization_ate.ipynb)
- [x] SMD table reviewed — 0 of 11 flagged, largest `|SMD|` 0.014 (notebook 01 §2)
- [x] ATE table (95% CI) interpreted — `visit` +7.66 pp, `conversion` +0.68 pp, `spend` +$0.77 (notebook 01 §3)
- [x] Binary arm chosen and justified — `mens`, beats Womens on every outcome (notebook 01 §1)
- [x] Profit assumptions decided — 30% margin, $0.10/contact, `v` = $34.91 (notebook 01 §4)

Day 2 — feature pipeline, uplift metrics, response baseline + S/T/X learners.

- [x] Centralized figure saving — [`packages/plotting.py`](packages/plotting.py) → `docs/figures/`
- [x] Feature pipeline + stratified splits — [`packages/causal/features.py`](packages/causal/features.py)
- [x] Uplift metrics (Qini AUC, AUUC, uplift@k) + random baseline — [`packages/causal/metrics.py`](packages/causal/metrics.py)
- [x] Qini plotting — [`packages/causal/plotting.py`](packages/causal/plotting.py)
- [x] Baselines notebook — [`research/notebooks/02_hillstrom_uplift_baselines.ipynb`](research/notebooks/02_hillstrom_uplift_baselines.ipynb)
- [x] Qini comparison + interpretation — X-learner leads narrowly (0.0236), S-learner shrinks to ~0, response baseline is competitive (notebook 02 §4–5)

Day 3 — DR/causal-forest challengers, calibration, 10-seed benchmark.

- [x] Learner factory (response/S/T/X/DR/forest, one `fit/predict/effect_interval` API) — [`packages/causal/learners.py`](packages/causal/learners.py)
- [x] Calibration module (Brier, reliability, Platt/isotonic, CATE-by-decile) — [`packages/causal/calibration.py`](packages/causal/calibration.py)
- [x] Benchmark runner — 10 seeds, mean ± spread, profit valued on conversions — [`packages/causal/benchmark.py`](packages/causal/benchmark.py)
- [x] Learners + calibration notebook — [`research/notebooks/03_hillstrom_learners_calibration.ipynb`](research/notebooks/03_hillstrom_learners_calibration.ipynb)
- [x] Champion chosen on Qini + profit (pre-registered rule) — statistical tie at the top → **response baseline**; no uplift model clears it (notebook 03 §8)

Day 4 — decision layer: registry, profit curve, multi-treatment, policy tree, fairness, power.

- [x] Champion registry — `pipelines/train.py` → `models/champion.joblib` + provenance sidecar (`packages/causal/registry.py`); Day-6 API loads, never trains
- [x] Stratified-sampling visibility — `split_balance_report` (per-fold SMD/events) + `stratify_extra` money-outcome option (`packages/causal/features.py`)
- [x] Decision tools (5, typed with Pydantic) — `packages/decision/`: `optimize_budget`, `compare_scenarios`, `get_segment_profile`, `design_ab_test`, `get_model_card`
- [x] Decision notebook — [`research/notebooks/04_hillstrom_decision_layer.ipynb`](research/notebooks/04_hillstrom_decision_layer.ipynb): profit curve (break-even 2%, peak 30% $858 with its CI), scenario table, multi-arm IPW, policy tree, zip_code fairness audit, power sizing
- [x] Fairness audit — real contact disparity: Urban 59% vs 41% population share (zip-proxy split); group *effects* overlap cleanly; flagged in the model card, escalated to Day 6

Day 5 — campaign copilot: gate (Decisions API) → plan (Day-4 tools) → write (pluggable LLM) → validate (V1–V4).

- [x] Copilot pipeline — [`packages/copilot/`](packages/copilot/): router (3 named questions on one API call, calibrated thresholds, keyword fallback), planner (rule-based scenario extraction, cached holdout context, per-intent tool dispatch), providers (Decisions client; template / ollama / neurometric writers), validator (number-trace with citations, USD-only, causal-language rubric, section contract), MLflow GenAI trace per request
- [x] Config — [`configs/copilot.yaml`](configs/copilot.yaml): thresholds (incl. `t_oos_confidence`), 1% trace tolerance, writer pick, prompt v2
- [x] Seed eval — [`evals/seed_cases.yaml`](evals/seed_cases.yaml) (15 labelled cases) + [`evals/run_seed_eval.py`](evals/run_seed_eval.py) (fast-path/intent/tool accuracy, pass@1/2, p50/p95)
- [x] Copilot notebook — [`research/notebooks/05_hillstrom_copilot.ipynb`](research/notebooks/05_hillstrom_copilot.ipynb): live gate, 3-writer comparison on one brief, 4 use cases end-to-end, seed eval, validator torture (5/5 caught), fast-path timing, [`day5_copilot_latency.png`](docs/figures/day5_copilot_latency.png)
- [x] Live calibration — fast-path 1.00, intent/tools 1.00 on the seed set; briefs ~8 s total (ollama ~3.6 s of it), fast paths ~1.2 s gate-only; first-draft validator pass 0.44–0.78 → ~0.5–1.0 after the one rewrite

Days 6–7: see [`ROADMAP.md`](ROADMAP.md).

## Datasets

| Dataset | Rows | Treatment | Outcomes | Role |
| --- | --- | --- | --- | --- |
| [Hillstrom MineThatData](docs/datasets/hillstrom.md) | 64,000 | Mens / Womens e-mail / none | visit, conversion, spend | **Primary** — profit, explainability, copilot |

Pinned raw Hillstrom SHA-256:
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). Python is pinned to **3.11**
(`causalml` / `econml` do not yet support 3.13).

```bash
uv sync                              # create .venv and install everything
uv run python pipelines/download.py  # fetch the raw Hillstrom CSV if missing, then verify
uv run jupyter lab                   # open research/notebooks/
```

Run the tests and linters:

```bash
uv run pytest
uv run ruff check .
uv run mypy packages pipelines
```

## Profit assumptions

Stated openly, since every profit figure depends on them. **Decided on Day 1**
(notebook 01 §4; rationale in `configs/business_assumptions.yaml` — change only
with a recorded reason).

| Assumption | Value | Notes |
| --- | --- | --- |
| Reporting currency | USD ($) | Hillstrom spend is already USD |
| Gross margin on incremental spend | 30 % | retail gross-margin placeholder — assumption, not observed |
| Cost per contact (e-mail) | 0.10 USD | assumed fixed send cost per customer |
| Value per conversion (`v`) | 34.91 USD | $116.36 mean spend per converter × 30% margin — the conversion-route profit check |

Formula used throughout:

```text
profit(k) = v · Δconv(k) − c · n(k)
```

`v` = value per conversion, `Δconv(k)` = incremental conversions from targeting the
top k, `c` = cost per contact, `n(k)` = number targeted. `v` and `c` must share a
currency. Edit `configs/business_assumptions.yaml` — the copilot reads it.

## Headline results

**Day-1 baseline (notebook 01):** uniformly contacting the `mens` arm earns
≈ **+$0.13 per contact** (+$131–$138 per 1,000 contacted, after send costs) at
the assumptions above — the number every Day-2 model must beat.

**Day 2 (notebook 02):** ranking models on `visit`, test split. The honest
headline: uplift models do **not** cleanly beat the response baseline here, and
the signal is weak overall. The X-learner leads narrowly; the S-learner collapses
toward random. Full table and reading in
[`docs/concepts.md`](docs/concepts.md).

| Metric (single split) | Response model (baseline) | Best uplift model (X-learner) |
| --- | --- | --- |
| Qini AUC | 0.0199 | **0.0236** |
| uplift@30% (incremental visits/customer) | 0.0899 | **0.1015** |

**Day 3 (notebook 03):** the same comparison repeated over **10 seeds**, with
six learners and profit valued on **conversions** (not visits). The honest
headline: **no uplift model beats the response baseline.** Response tops Qini
(0.0231 ± 0.0094) *and* profit ($569 per test split = $0.223/contact vs
$0.137 uniform, +62%), but its edge over DR (0.0048) is inside its own spread —
declared a **statistical tie** and broken toward response on profit +
parsimony. DR is the best actual uplift model; X, T and the forest are
statistically at random on `visit`. Recalibration: Platt wins, modestly.
Full benchmark, noise floors and the champion reasoning in
[`docs/concepts.md`](docs/concepts.md).

| Metric (10-seed means) | Response model (champion) | Best uplift model (DR-learner) |
| --- | --- | --- |
| Qini AUC | **0.0231 ± 0.0094** | 0.0183 ± 0.0091 |
| uplift@30% (incremental visits/customer) | **0.0912** | 0.0911 |
| Incremental conversions at 30% targeted | **23.6** (9.2 per 1,000 contacted) | 20.7 (8.1 per 1,000) |
| Incremental profit per $10k spend | **$22,300** | $18,200 |
| Copilot numeric faithfulness (briefs fully traced to tools, live) | 0.56 first draft → 0.67 after one rewrite (Day 5, 9 briefs) | — |

*Profit per $10k spend = net profit per contact (10-seed mean, after contact
cost) × 100,000 contacts; $10k buys 100k contacts at $0.10 each.*

**Day 4 (notebook 04):** the champion becomes a **registered artifact**
(`models/champion.joblib` + provenance sidecar; the ledger's single-split row
reproduced exactly) and five typed tools turn the ranking into decisions.
Honest headlines: the profit curve breaks even at 2% depth and peaks at 30%
($858 on the champion's split — but its CI barely clears zero, and 16 of 25
depths' CIs include it: one split cannot price a depth). Target-all earns
more in absolute dollars ($1,451) at 3.3× the budget — $0.17/contact vs the
champion's $0.34. "Which email, if any?" is answerable via IPW but its
intervals are ~3× wider, and the budget-limited multi-arm policy lost money
on this split (−$70). The policy tree's 6 readable segments **all** clear
zero uplift on the holdout — trees are effect maps, not profit oracles. The
fairness audit found a **real proxy-attribute risk**: Urban customers
contacted at 59% vs their 41% share (the tree splits on `zip_code_Urban`),
while group effects overlap cleanly. And a +1pp lift test needs 15,510/arm
(16 days at 2,000/day) — Day 3's wide intervals were a sample-size story all
along. Full numbers in [`docs/concepts.md`](docs/concepts.md).

**Day 5 (notebook 05):** the campaign copilot — gate → plan → write →
validate. Live headline: the three-question Decisions gate routes all 15
seed cases correctly (**fast-path accuracy 1.00**, ~1.2 s gate-only for
decline/redirect/clarify with zero writer tokens); every use case produces a
brief whose numbers trace to Day-4 tools (V1–V4, one rewrite, then explicit
`[UNVERIFIED]` flags); and the validator torture test catches **5/5**
hand-corrupted drafts, each by the owning rule. Writer verdict on the same
brief: ollama `gpt-oss:120b-cloud` passes at ~3.6 s; `neurometric/goal-setter`
fails number-trace + causal-language (prompt-shaped wrong — the provider
abstraction made that finding a config flip); the deterministic template is
the always-on floor. Seed-set latency p50 4.9 s / p95 9.5 s; first drafts pass
the number-trace 0.44–0.78 of the time and the rewrite closes most of the
rest — the validator, not the model, is what makes the brief trustworthy.
Full numbers in [`docs/concepts.md`](docs/concepts.md).

## Repo structure

```text
.
├── packages/          # importable Python: datasets, causal, decision, copilot
├── pipelines/         # download / train / evaluate scripts
├── research/notebooks # hands-on analysis (notebooks are research only)
├── data/raw           # git-ignored; verified by pipelines/download.py
├── data/processed     # git-ignored; regenerable Parquet
├── configs/           # business assumptions and model config YAML
├── docs/              # dataset specs, model card, ADRs
├── evals/             # copilot eval cases + runner (Day 5–6)
├── apps/              # FastAPI service + React frontend (Day 6)
├── infra/             # Docker / compose
└── .github/workflows/ # CI: lint, types, tests, eval smoke run
```

Boundaries that keep it honest: notebooks are research only; reusable code lives in
`packages/`; the API only ever loads a registered model, never trains.

## Risks and limitations

| Risk | Mitigation |
| --- | --- |
| Profit depends on assumed value and cost | State assumptions here + in every brief; show a range |
| Hillstrom spend is USD | Report in USD throughout; no conversion step |
| Results come from past experiments, not live deployment | Frame as "estimated on held-out randomized data" |
| Zip code can proxy for protected attributes | Router policy check + fairness note on targeting rates |
| An LLM could state unsupported numbers or causal claims | Grounding validator; causal-language judge rubric |

## References

- scikit-uplift docs — <https://www.uplift-modeling.com/>
- MineThatData Hillstrom challenge (original prose: [`docs/datasets/hillstrom.md`](docs/datasets/hillstrom.md), Appendix A)
