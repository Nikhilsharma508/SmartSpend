# SmartSpend — 7-Day Build Plan

Primary dataset: Hillstrom (treatment = `segment`, rank on `visit`, value with `spend`)

Legend:
- [YOU] = hands-on: analysis, modeling decisions, interpretation, design
- [AGENT] = vibe-code: setup, boilerplate, wrappers, UI, tests (review each diff)
- `[x]` = done · `[~]` = in progress · `[ ]` = not started

**Progress:** Days 1–5 ✅ · Day 6 next · Day 7 not started.

> **Figure convention.** Every plot a notebook produces is saved through
> `packages/plotting.py::save_fig(fig, name)` into `docs/figures/<name>.png`
> (committed), with a descriptive snake_case name. Read them straight from that
> folder to review results; the notebook links back to each file.

---

## Day 1 — Data understanding and experiment validation ✅

- [x] [AGENT] Create the env, pin versions (sklift, causalml, econml, lightgbm, statsmodels), set up the repo skeleton and README template
- [x] [AGENT] Write a data loader for Hillstrom (`fetch_hillstrom`) with one cleaned DataFrame as output
- [x] [YOU] EDA: arm sizes, outcome rates per arm, spend distribution (zero-inflation), feature distributions
- [x] [YOU] Randomization check: SMD per feature across the 3 arms; flag any SMD > 0.1
- [x] [YOU] ATE per arm vs control for visit, conversion and spend, with 95% CIs
- [x] [YOU] Choose the binary arm (Mens or Womens vs No E-Mail) based on the observed lift; write down why
- [x] [YOU] Fix profit assumptions: margin % on spend, cost per email, currency (USD only)

Done when: notebook 01 has the SMD table, ATE table and a written assumptions block ✅
*(Work lives across `00_hillstrom_eda.ipynb` + `01_hillstrom_randomization_ate.ipynb` — the deliberate EDA/causal split.)*

**Day-1 checklist** *(relocated from `PROJECT.md`)*:

- [x] Fresh environment with pinned versions (`uv` + `.python-version` on Python 3.11).
- [x] Randomization checked with standardized mean differences for the Hillstrom arms; any SMD above 0.1 flagged.
- [x] ATE with a confidence interval for each outcome (`visit`, `conversion`, `spend`; notebook 01).
- [x] Profit assumptions written down (30% margin, $0.10/contact, USD only) in `configs/business_assumptions.yaml`, mirrored in the README.
- [x] Repo structure set up with the headline results table scaffolded.

---

## Day 2 — Baselines and first uplift models ✅

- [x] [AGENT] Feature pipeline: one-hot `channel` and `zip_code`, keep `segment` out of features
- [x] [AGENT] Train/val/test split stratified on treatment × visit; a reusable Qini/AUUC/uplift@k plotting module
- [x] [YOU] Response baseline: classifier on treated users only (the "CTR-style" model)
- [x] [YOU] S-, T- and X-learners with LightGBM on `visit`
- [x] [YOU] Compare Qini curves against the baseline and random; explain why the S-learner can shrink uplift toward zero
- [x] [YOU] Note the first finding: does the response model rank "sure things" on top?

Done when: one Qini plot with all models and a short written interpretation ✅
*(Notebook 02 + `docs/figures/day2_qini_models_vs_baseline.png`. Finding: uplift models do **not** cleanly beat the response baseline on `visit`; X-learner leads narrowly at Qini 0.0236, S-learner shrinks to ~0 — weak heterogeneity, sets up Day-3 DR/causal-forest challengers.)*

---

## Day 3 — Advanced models and evaluation depth ✅

- [x] [YOU] DR-learner on `visit`; DR-learner on `spend` (regression CATE challenger)
- [x] [YOU] Causal forest (econml) with effect intervals
- [x] [YOU] CATE calibration by decile: predicted vs observed uplift
- [x] [YOU] Brier score and reliability curves for the treated and control outcome models and the baseline
- [x] [YOU] Decide on recalibration (isotonic or Platt, fitted per arm on validation only) → **Platt, per arm, validation only**
- [x] [AGENT] Seed/split loop runner (10 runs) that aggregates mean ± spread into a benchmark table
- [x] [YOU] Pick the champion model and write the justification (Qini + profit, not Brier) → **champion: response baseline**

Done when: the benchmark table is filled in and the champion model is chosen with reasons ✅
*(Notebook 03 + 4 figures `docs/figures/day3_*.png`. Finding: over 10 seeds **no
uplift model beats the response baseline** — response tops Qini (0.0231 ± 0.0094)
and profit ($569/test split, $0.223/contact vs $0.137 uniform), but its edge over
DR (0.0048) is inside its own spread → pre-registered rule declares a statistical
tie, broken on profit + parsimony. X/T/forest are statistically at random on
`visit`; Qini and profit disagree on ordering (DR 2nd/5th); profit is valued on
**conversions** (valuing visits overstates ~14×); Platt beats isotonic modestly;
single-split spend orderings sit inside the random noise floor (σ ≈ 1.0 at 10%).
All numbers in `docs/concepts.md` findings ledger, Day 3.)*

---

## Day 4 — Decision layer ✅

- [x] [YOU] Profit curve on holdout: break-even at 2% depth; peak at 30% = $858 (CI [140, 1,576]); 16/25 depth CIs include zero — no optimal-depth claim from one split (notebook 03 §3)
- [x] [YOU] Compare against target-all, random and response-score targeting at a fixed budget — target-all $1,451 at 3.3× the budget (infeasible at 30%); champion $858 vs random $435, CIs overlap (notebook 03 §4)
- [x] [YOU] Multi-treatment: "which email, if any?" per customer — per-arm DR + argmax policy valued with IPW; argmax-all $500 vs mens-all $396 (inside IPW noise); budget-limited top-30% lost $70 (notebook 03 §5)
- [x] [YOU] Policy tree: readable segment rules, sanity-checked — econml `DRPolicyTree` (causalml's `SingleTreeCateInterpreter` was removed upstream in 0.18); 6 segments; every holdout-graded CI excludes zero, contradicting the tree's "no email" labels — trees are effect maps, not profit oracles (notebook 03 §6)
- [x] [YOU] Fairness check: targeting rate and uplift by `zip_code` — real contact disparity (Urban 59% vs 41% population share via a `zip_code_Urban` split); group effects overlap cleanly; flagged in the model card, escalated to Day 6 (notebook 03 §7)
- [x] [YOU] Power analysis for the next test: realistic MDE → sample size and duration — +1pp on the 10.6% visit rate needs 15,510/arm ≈ 16 days at 2,000/day (notebook 03 §8)
- [x] [YOU] (added) Champion registry — `pipelines/train.py` → `models/champion.joblib` + provenance sidecar (`packages/causal/registry.py`); the ledger's single-split row reproduces exactly
- [x] [YOU] (added) Stratified-sampling visibility — `split_balance_report` (per-fold SMD/events) + `stratify_extra` money-outcome option (a verified no-op on Hillstrom: conversion nests inside visit)
- [x] [AGENT] Wrap these as 5 typed tools (optimize_budget, compare_scenarios, get_segment_profile, design_ab_test, get_model_card) with Pydantic schemas and pytest tests — `packages/decision/`, 16 tests in `tests/test_decision.py`

Done when: all 5 tools are callable from Python and the tests pass — ✅ (notebook 03 §9 + test suite)

---

## Day 5 — Campaign Copilot ✅

- [x] [YOU] Router — three named Decisions API questions on one call (intent choice, policy-risk noul, missing-inputs noul); thresholds calibrated live on the 15 seed cases (fast-path accuracy 0.533 → 1.00): risk ≥ 0.70 decline, confident out-of-scope ≥ 0.90 redirect, missing ≥ 0.60 clarify (beats an *uncertain* out-of-scope), confidence < 0.50 redirect; deterministic keyword fallback when the API is unreachable
- [x] [YOU] Scenario schema — `CampaignScenario` + rule-based extraction: USD amounts (`$`, commas, `k` suffix), cost-vs-budget disambiguation, depth %, MDE pp, daily volume; documented defaults land in `assumptions_used`, absent money in `missing_fields`
- [x] [YOU] Brief writer — prompt v2 (numbers verbatim from tools, never derived; causal-care wording; exact per-intent headers) + the deterministic template; pluggable providers: template / ollama `gpt-oss:120b-cloud` / NEUROMETRIC `goal-setter` (measured verdict: ollama passes V1–V4 at ~3.6 s, goal-setter fails V1+V3, template is the always-on floor)
- [x] [YOU] Validator — V1 number-trace (1% tolerance, fraction↔percent, per-kind floors, LLM-typography normalization, citations with tool paths), V2 USD-only, V3 causal deny-list + interval phrasing, V4 section headings; one rewrite with the errors attached, then explicit `[UNVERIFIED]` flags; torture test 5/5 (each mutation caught by the owning rule)
- [x] [YOU] 15 seed cases — `evals/seed_cases.yaml` + `evals/run_seed_eval.py`; live: fast-path 1.00, intent/tools 1.00, validator pass@1 0.44–0.78 → pass@2 0.50–1.00 (the rewrite closes the gap), p50 4.9 s / p95 9.5 s; gate-only fast paths 1.16–1.25 s, zero writer tokens
- [x] [AGENT] `packages/copilot/` — router / planner / writer / validator / providers / trace + the `run_request` orchestrator; JSONL trace per stage under `traces/`; 64 copilot tests (175 repo-wide)
- [x] [AGENT] Number extraction — USD/`%`/counts/signed decimals with the V1 citation paths; normalizes LLM typography (U+2010+ dashes used as minus, no-break digit grouping like `2 443`) — the main live first-draft failure mode, found and fixed by running the pipeline live

Done when: an end-to-end brief is generated for all 4 use cases and passes the validator — ✅ (notebook 04 §3: all four intents routed correctly, all drafts pass V1–V4; measured latencies and provider verdicts in `docs/concepts.md` §7 + Day-5 ledger)

---

## Day 6 — Copilot evaluation and app

- [ ] [YOU] Expand the eval set to ~50 cases, including adversarial ones (fake numbers in the prompt, proxy-attribute targeting)
- [ ] [YOU] Run the eval; do failure analysis by category; fix the prompts and thresholds
- [ ] [YOU] Router ablation: with vs without the Decisions API (latency, cost, decline accuracy); write the verdict
- [ ] [AGENT] Eval runner that outputs a metrics report (routing, extraction, tool calls, faithfulness, latency, cost)
- [ ] [AGENT] FastAPI service + React campaign planner (budget inputs, profit curve, segment rules, copilot chat, CSV download)

Done when: the eval report meets targets or the gaps are explained, and the app runs locally

---

## Day 7 — Ship

- [ ] [AGENT] Deploy (Render / HF Spaces / Streamlit Cloud), set up the demo recording
- [ ] [YOU] README: problem, headline results table, key plots, what didn't work, limitations
- [ ] [YOU] Model card: data, assumptions, fairness note, when not to use it
- [ ] [YOU] Write-up (blog or LinkedIn) and resume bullets with your final numbers
- [ ] [YOU] Record a 2-minute demo: question → brief → target list

Done when: the repo is public, the app is deployed, the README results are filled in, and the demo video is linked
