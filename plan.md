# UpliftIQ — 7-Day Build Plan

Primary dataset: Hillstrom (treatment = `segment`, rank on `visit`, value with `spend`)

Legend:
- [YOU] = hands-on: analysis, modeling decisions, interpretation, design
- [AGENT] = vibe-code: setup, boilerplate, wrappers, UI, tests (review each diff)
- `[x]` = done · `[~]` = in progress · `[ ]` = not started

**Progress:** Day 1 complete · Day 2 in progress · Days 3–7 not started.

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
- [x] [YOU] Fix profit assumptions: margin % on spend, cost per email, currency handling (USD or a stated INR rate)

Done when: notebook 01 has the SMD table, ATE table and a written assumptions block ✅
*(Work lives across `01_hillstrom_eda.ipynb` + `02_hillstrom_randomization_ate.ipynb` — the deliberate EDA/causal split.)*

---

## Day 2 — Baselines and first uplift models

- [ ] [AGENT] Feature pipeline: one-hot `channel` and `zip_code`, keep `segment` out of features
- [ ] [AGENT] Train/val/test split stratified on treatment × visit; a reusable Qini/AUUC/uplift@k plotting module
- [ ] [YOU] Response baseline: classifier on treated users only (the "CTR-style" model)
- [ ] [YOU] S-, T- and X-learners with LightGBM on `visit`
- [ ] [YOU] Compare Qini curves against the baseline and random; explain why the S-learner can shrink uplift toward zero
- [ ] [YOU] Note the first finding: does the response model rank "sure things" on top?

Done when: one Qini plot with all models and a short written interpretation

---

## Day 3 — Advanced models and evaluation depth

- [ ] [YOU] DR-learner on `visit`; DR-learner on `spend` (regression CATE challenger)
- [ ] [YOU] Causal forest (econml) with effect intervals
- [ ] [YOU] CATE calibration by decile: predicted vs observed uplift
- [ ] [YOU] Brier score and reliability curves for the treated and control outcome models and the baseline
- [ ] [YOU] Decide on recalibration (isotonic or Platt, fitted per arm on validation only)
- [ ] [AGENT] Seed/split loop runner (10 runs) that aggregates mean ± spread into a benchmark table
- [ ] [YOU] Pick the champion model and write the justification (Qini + profit, not Brier)

Done when: the benchmark table is filled in and the champion model is chosen with reasons

---

## Day 4 — Decision layer

- [ ] [YOU] Profit curve on holdout: incremental spend × margin − email cost at each targeting depth; find the break-even depth
- [ ] [YOU] Compare against target-all, random and response-score targeting at a fixed budget
- [ ] [YOU] Multi-treatment: "which email, if any?" per customer (causalml multi-arm)
- [ ] [YOU] Policy tree (SingleTreeCateInterpreter): extract readable segment rules, sanity-check them
- [ ] [YOU] Fairness check: targeting rate and uplift by `zip_code`
- [ ] [YOU] Power analysis for the next test: choose a realistic MDE, get sample size and duration
- [ ] [AGENT] Wrap these as 5 typed tools (optimize_budget, compare_scenarios, get_segment_profile, design_ab_test, get_model_card) with Pydantic schemas and pytest tests

Done when: all 5 tools are callable from Python and the tests pass

---

## Day 5 — Campaign Copilot

- [ ] [YOU] Design the router: exact Decisions API questions (intent choice, policy-risk predicate, missing-input predicate) and their thresholds
- [ ] [YOU] Finalize the Scenario schema: required fields, documented defaults, assumption flags
- [ ] [YOU] Write the brief writer prompt and the brief template (recommendation, targets, intervals, baseline, assumptions, risks)
- [ ] [YOU] Define the validator rules: number-matching tolerance, unit conversions, required content, causal-language rubric
- [ ] [YOU] Hand-write 15 seed eval cases (normal, ambiguous, out-of-scope, policy-risk)
- [ ] [AGENT] Implement router.py, planner.py, writer.py, validator.py, providers.py (with a Responses API fallback for the router)
- [ ] [AGENT] Number-extraction parser (₹, $, %, lakh/crore, commas)

Done when: an end-to-end brief is generated for all 4 use cases and passes the validator

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
