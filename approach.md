# Approach — Day 5 (Campaign Copilot) implementation plan

> **What this file is.** The approved, detailed plan the next session executes. Written
> 2026-10-19 after the architecture Q&A with the user. Days 1–4 are done and pushed
> (`fd52483`); **nothing of Day 5 is built yet** — this document is the handoff.
> It is deleted or trimmed once Day 5 lands; `docs/concepts.md` + `plan.md` remain the
> canonical truth afterwards.

---

## 0. Decided inputs (do not relitigate)

| # | Decision | Source |
| --- | --- | --- |
| 1 | **Decisions API** = the Azure endpoint in `.env` (`AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_DEPLOYMENT_NAME`), request shape `{"model", "state", "questions"}`, question types `noul` / `choice` / `score`, answers keyed by question **name** | `research/notebooks/decision_api.ipynb` (verified live — source of truth) |
| 2 | `docs/decision_api.md` (OpenAI `/v1/decisions` shape) is **context only** — its *design rules* apply (no prose out of it, gate-then-generate, threshold confidences, always an `other` choice, name every question), its *field names* do not | user-provided doc |
| 3 | **Generative writers**: `TemplateWriter` (no LLM, always ships) + **Ollama** (`gpt-oss:120b-cloud`) + **NEUROMETRIC** (`neurometric/goal-setter` on an OpenAI-compatible gateway) — raw HTTP clients, no langchain dependency in the package | notebook cells `e14b2bce`–`80abcd2d`; `.env` names `NEUROMETRIC_BASE_URL`, `NEUROMETRIC_API` (names only, values never read into code/docs) |
| 4 | **Architecture approved**: Gate → Plan → Write → Validate; <1s fast paths for decline/clarify/out-of-scope; heavy artifacts (champion model, policy-tree segments) precomputed and cached at boot; validator code — not model intelligence — enforces numeric faithfulness | user Q&A this session |
| 5 | Profit in **USD only**; tools compute, LLM narrates; every number cited; `docs/concepts.md` synced in the same change; `ruff → mypy → pytest` green before every commit; push on every commit; standing rule: nothing valuable lives only in shell output | AGENTS.md + standing user rules |

`.env` variable **names** (values are private, never printed/logged/committed):
`AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_DEPLOYMENT_NAME`, `NEUROMETRIC_BASE_URL`,
`NEUROMETRIC_API`. `.gitignore` already covers `.env` (allows `!.env.example` only).

**Implementation session must first** (step 0 of the commit sequence, §8): commit the
user's untracked reference material `docs/decision_api.md` +
`research/notebooks/decision_api.ipynb` + the `pyproject.toml`/`uv.lock` changes
(`python-dotenv`), and update `description.md`/`plan.md` to the architecture in §3 —
per the user's instruction that md files track the real architecture, not the sketch.

---

## 1. Day-5 done criterion (from `plan.md`)

> An end-to-end brief is generated for all 4 use cases and passes the validator.

The 4 use cases (`description.md`): **plan a campaign**, **what-if**, **explain
segments**, **design the next test**. Plus the user-added bar this session: the
pipeline must show *visibility* (structured trace of every stage), *speed* (fast
gate-only paths), and *accuracy* (validator-enforced), and the md files must be
updated to match reality.

---

## 2. Request flow — the whole pipeline

```text
marketer's plain-language question
  │
  ├─[1] GATE ─ DecisionsClient ─ 1 HTTP call, ~0.2–0.6s, 3 NAMED questions
  │      intent          : choice  {plan, what_if, explain, test_design, out_of_scope}
  │      policy_risk     : noul    (targets/excludes people by a sensitive attribute
  │                                  or a proxy for one?)
  │      missing_inputs  : noul    (is a required parameter absent?)
  │      │
  │      ├ policy_risk  ≥ T_risk   → Decline  (template, one-line why)      <1s total
  │      ├ intent = out_of_scope
  │      │  or confidence < T_conf → Redirect (template)                   <1s total
  │      ├ missing_inputs ≥ T_miss → Clarify  (template or writer, ONE q)  <1s total
  │      └ else routed intent ↓
  │
  ├─[2] PLAN ─ Planner (pure Python)
  │      intent → Scenario schema (validated) → Day-4 tools
  │      champion + policy-tree segments loaded ONCE at boot and cached;
  │      only the per-request tools run (optimize_budget / compare_scenarios /
  │      get_segment_profile / design_ab_test — all fast, <2s worst case)
  │      │
  │      tool exception → typed ToolFailure, surfaced to [3] as text
  │      (the writer can never invent a number for a failed tool)
  │
  ├─[3] WRITE ─ pluggable Writer (see §4)
  │      tool outputs (JSON, numbers verbatim) + house template + hard rules
  │      → draft brief in the provider's output contract
  │      TemplateWriter  : deterministic, 0ms, always available (the fallback)
  │      OllamaWriter    : POST {ollama}/v1/chat/completions   (gpt-oss:120b-cloud)
  │      NeurometricWriter: POST {NEUROMETRIC_BASE_URL}/chat/completions
  │                         model neurometric/goal-setter
  │
  └─[4] VALIDATE ─ pure Python, no LLM
         V1 number-trace : every $ / % / count / days in the prose must appear in a
                           tool output within tolerance (rel 1% or per-unit abs)
         V2 units        : USD everywhere, thousands separators, depth as %
         V3 causal rubric: deny "will buy", "this customer will", "guaranteed",
                           "proves"; require interval/bounds phrasing where CI exists
         V4 sections     : recommendation, targets, intervals, baseline,
                           assumptions, risks — all present
         ├ PASS → BriefPackage{ prose, citations[], scenario, trace_id, csv path }
         └ FAIL → ONE rewrite call: same writer, draft + validator error list
                  └ FAIL again → return partial brief with explicit
                    [UNVERIFIED: <what failed>] flags — never silently pass
```

**Latency budget** (p50): decline/clarify/redirect **<1s** (gate only, zero writer
tokens — this is the "client wants a quick decision" path); full brief
**4–12s** (tools ~1–2s + writer 2–8s + validate <50ms + possible rewrite +3–8s).
**Cost per request**: decisions ≈ pennies (cheap/fast per user); writer tokens
~3K in / ~0.7K out — negligible on both providers; template = zero.

---

## 3. Why the split (the LLM part, plainly)

Three of the four stages are **code** and must stay code:

- the **gate** returns typed probabilities, not prose — perfect for the Decisions API;
- the **plan** computes causal quantities — an LLM must never do this math;
- the **validate** checks facts — an LLM grading its own numbers is circular.

The **only** place a generative model earns its keep is *narration*: turning verified
JSON into plain language a marketer wants to read. Therefore the prompt contains
**all numbers pre-computed**, the model is instructed to copy them **verbatim**, and
the validator re-checks every one afterwards. Intelligence is not the safety lever;
the validator is. That is why free/smaller models suffice and top-tier is unnecessary.

### 3.1 Provider abstraction (writer.py + providers.py)

```text
                    ┌────────────────────────────────────────────┐
 DraftRequest ────► │ WriterBase.write(draft) -> DraftResult      │
 (tool outputs,     └───────────────┬────────────────────────────┘
  scenario, intent,                │
  question, retry_errors=[])       ├── TemplateWriter
                                   │     deterministic render from the same
                                   │     house template; 0 ms; always works
                                   ├── OllamaWriter
                                   │     POST {OLLAMA_BASE_URL:-http://localhost:11434}
                                   │       /v1/chat/completions   (OpenAI-compatible)
                                   │     model: "gpt-oss:120b-cloud"   (config)
                                   └── NeurometricWriter
                                         POST {NEUROMETRIC_BASE_URL}/chat/completions
                                         model: "neurometric/goal-setter" (config)
                                         Authorization: Bearer {NEUROMETRIC_API}

get_writer(config) -> WriterBase     # configs/copilot.yaml selects:
                                     #   writer.provider: template | ollama | neurometric
                                     # per-request override allowed (tests, ablation)
```

All three clients are **thin `requests` code exactly like `decision_api.ipynb`** —
timeout, `raise_for_status`, typed `ProviderError` on failure. No langchain in the
package (the notebook's langchain calls stay as the user's reference; if the user
wants the notebook runnable inside the project env, add `langchain-ollama` +
`langchain-openai` as **dev** deps — open question §9).

**Provider default**: start with `ollama` as the configured default; `neurometric`
is one config flip away. Notebook 06 eyeballs both on the same 4 briefs + validator
pass rate + latency; the winner becomes the committed default (evidence, not vibes).

### 3.2 Every task that can use a generative model — per-task architecture

| Task | What it produces | LLM or template? | Input size | Output contract |
| --- | --- | --- | --- | --- |
| **T1 Brief body** (plan / what-if / test-design) | The written campaign brief | **LLM default**, TemplateWriter is the fallback and the retry-degradation target | ~3.0K tok (system+template 1.1K, tool JSON 1.8K, question 50) | Markdown, ≤600 words, sections per §3 V4 |
| **T2 Explain-segments narrative** | Plain-language description of the policy-tree segments | **LLM** (this is where prose adds real value over a table); template renders the table as fallback | ~1.5K tok | Markdown, ≤250 words, one paragraph per segment |
| **T3 Clarifying question** (missing inputs) | The one follow-up question | **Template default** (fast, free, zero-variance); optional LLM polish via config flag | ~0.4K tok if LLM | 1–3 sentences, asks ONLY for the missing fields |
| **T4 Decline / out-of-scope redirect** | Refusal with reason | **Template only** — refusals must be deterministic and rubric-controlled | — | 1–3 sentences + pointer to what IS supported |
| **T5 Rewrite retry** (validator failed) | Fixed draft | **LLM only** (same writer), one retry, prompt = draft + validator error list | ~3.5K tok | Same contract as T1 |
| **T6 Router fallback** (decisions endpoint down) | Routed intent | **No LLM** — deterministic keyword scorer over the 5 intents, lower confidence, logged as `router=keyword_fallback` | — | Same Route object |
| **T7 Eval-case drafting** (Day 6) | Adversarial seed cases | **LLM offline**, human-curated before use | ad hoc | YAML cases |

**Recommendation** (user may override): ship T1/T2/T5 LLM-backed + T3/T4 template +
T6 keyword fallback. Clarify/decline stay template because they are the *fast* path —
the user explicitly wants quick decisions there — and their wording carries policy,
which is exactly what you want deterministic.

### 3.3 Prompt design (T1; T2/T5 share the skeleton)

```text
SYSTEM (fixed, versioned in configs/copilot.yaml as prompt_version):
  role          : analyst at a causal-promo vendor; you narrate, never compute
  hard rules    : every number must appear VERBATIM in the DATA block;
                  never invent, never re-round, never average;
                  say "the model estimates / the holdout shows", never "will";
                  where an interval exists you MUST show it;
                  if a section's data says UNAVAILABLE, write why, do not fill it
  house style   : sections exactly in order: Recommendation / Who to target /
                  What it earns (with baseline comparison) / Uncertainties /
                  Assumptions / Next test
DATA (JSON, injected verbatim by writer.py):
  { scenario, tools: { name: [outputs...], assumptions, model_card boundaries } }
USER:
  the marketer's original question + intent label
```

Number-injection format matters: tool outputs go in as a JSON block with stable
keys, and the brief template tells the model which key to cite for which sentence
(e.g. "recommendation must cite `profit_curve.recommended_*`"), so the validator's
number-trace (V1) has a deterministic target to match against.

### 3.4 What the validator enforces per task

- **T1**: V1–V4 (full battery) + retry once on failure.
- **T2**: V1 (segment sizes/uplifts verbatim), V3 (no causal overreach per segment).
- **T3**: must mention every field listed by the planner as missing — no more.
- **T4**: must contain the decline reason enum text (rubric-controlled), no numbers at all.
- **T5**: re-run the full battery; second failure → degraded output with `[UNVERIFIED]`.

---

## 4. Component plan (file by file)

```text
packages/copilot/
  __init__.py        exports run_request(question, *, config=None) -> BriefPackage
  schemas.py         Pydantic: Route, ClarifyRequest, Decline, Scenario,
                     ToolRun, DraftRequest, DraftResult, Citation,
                     BriefPackage, ValidationReport, TraceEvent
  providers.py       DecisionsClient, WriterBase, TemplateWriter, OllamaWriter,
                     NeurometricWriter, get_writer(), ProviderError
  router.py          build_questions(), parse_route(), apply_thresholds(),
                     keyword_fallback_route()   (T6)
  planner.py         route -> Scenario -> calls Day-4 tools; boot-time cache of
                     champion + segment profile; ToolFailure handling
  writer.py          prompt assembly (§3.3), DraftRequest construction,
                     retry loop (draft -> validate -> one rewrite)
  validator.py       V1 number-trace, V2 units, V3 causal rubric, V4 sections
  trace.py           TraceEvent -> traces/<date>.jsonl (gitignored), one row per stage

configs/copilot.yaml (new; thresholds + prompts, single source)
  router:   { t_policy_risk: 0.70, t_missing: 0.60, t_confidence: 0.50 }
  validator:{ rel_tolerance: 0.01, max_rewrites: 1 }
  writer:   { provider: ollama, ollama_model: "gpt-oss:120b-cloud",
              neurometric_model: "neurometric/goal-setter", timeout_s: 60 }
  prompts:  { prompt_version: v1, brief_sections: [...] }

evals/seed_cases.yaml        the 15 hand-written Day-5 cases (4 use cases +
                             ambiguous + out-of-scope + policy-risk + missing-input)
.env.example (new)           names only, placeholder values, committed
traces/                      gitignored output dir (add to .gitignore)
```

`.env` names needed so far: exactly the 5 that exist. Ollama's base URL defaults to
`http://localhost:11434` (config-overridable); if the `gpt-oss:120b-cloud` model
turns out to need an Ollama Cloud token, that becomes one more `.env` name —
resolved by a live smoke test in notebook 06 (open question §9).

**Scenario schema** (planner input, the contract between gate and tools): intent,
budget_usd (optional), depth (default 0.30), cost_per_contact override (default
from assumptions yaml), mde (for test design, default 0.01), segment question text.
Documented defaults; missing required field per intent → the `missing_inputs`
noul in the gate asked it *before* planning, so the planner never guesses money.

---

## 5. Tests plan (`tests/test_copilot.py` + provider unit tests)

- **Unit, no network**: router threshold matrix (each gate path fires at its
  threshold), parse of noul/choice/score answers incl. `type == refusal`,
  planner → tool selection per intent (mock tool calls, assert args),
  validator catches each failure class (invented number, unit slip, causal
  phrasing, missing section), retry loop reaches `[UNVERIFIED]` after max_rewrites,
  template writer output passes the validator for all 4 intents,
  keyword fallback routes the 4 use cases.
- **HTTP mocked** (`monkeypatch` requests): DecisionsClient happy/refusal/500/timeout;
  Ollama + Neurometric clients: payload shape (messages, model, no forbidden fields),
  parse, ProviderError mapping.
- **Live smoke, skipped without env** (`pytest.mark.skipif` on missing `.env` names):
  one decisions call + one tiny call per configured writer — never in CI, opt-in locally.
- **Seed eval** (`evals/run_seed_eval.py`): the 15 cases end-to-end with the
  configured writer; report: route accuracy, validator pass@1 / pass@2, latency,
  tokens. Notebook 06 prints it; Day 6 grows it to ~50 + the router ablation
  (decisions vs keyword fallback: latency, cost, decline accuracy).

---

## 6. Notebook 06 (`06_hillstrom_copilot.ipynb`) — research only

§0 setup · §1 the gate live (3 questions printed with probabilities) · §2 writer
provider comparison on the same brief (ollama vs neurometric vs template: latency,
validator result, eyeball) · §3 the 4 use cases end-to-end (briefs + traces) ·
§4 validator torture test (hand-corrupted drafts: invented $, "will buy", missing
section — each caught) · §5 fast-path timing (decline/clarify <1s claim measured).
Figures via `save_fig`: `day6_copilot_latency.png` (per-stage bars), and Day-6's
eval gets `day6_router_ablation.png`. Numbers → concepts ledger.

---

## 7. Docs sync (same change as the code)

- `docs/concepts.md` **§7 Campaign copilot**: definitions — gate/router with
  thresholds, scenario schema, writer providers, number-trace validation, causal
  language rubric, trace/visibility, `[UNVERIFIED]` degradation; ledger gains the
  Day-5 rows after notebook 06 runs (latencies, validator pass rates, provider verdict).
- `README.md`: status → Days 1–5, Day-5 checklist block, config/.env names section
  (names only), architecture one-liner.
- `plan.md`: Day 5 checkboxes with measured outcomes when done.
- `description.md`: replace the top-level flow sketch with the §3 architecture
  (the user's explicit ask: md reflects the real pipeline).

---

## 8. Commit sequence (each green: ruff → mypy → pytest, push after each)

1. `docs: decision-api reference material` — user's `docs/decision_api.md`,
   `research/notebooks/decision_api.ipynb`, pyproject/uv.lock (python-dotenv);
   `.env.example`; this `approach.md`; description.md/plan.md architecture update.
2. `feat(copilot): schemas + providers (decisions, ollama, neurometric, template)` + unit tests.
3. `feat(copilot): router, planner, writer, validator, trace` + unit tests.
4. `feat(research): notebook 06 — gate live, provider comparison, 4 use cases, validator torture` + figures.
5. `docs: sync Day 5` — concepts §7 + ledger, README, plan.md.

---

## 9. Open questions / risks (resolve during implementation, ask user if material)

1. **Ollama cloud auth**: `gpt-oss:120b-cloud` may need a token/base-URL beyond
   `localhost:11434` for raw HTTP — smoke-test early (notebook 06 §2); if it needs
   a key, add one `.env` name and tell the user.
2. **Neurometric model fitness**: `goal-setter` is the only model seen on that
   gateway; it may be prompt-shaped for goal-setting, not narration. If its briefs
   fail the validator too often, recommend the user expose a general model on the
   gateway, or keep ollama as default (evidence from §2 comparison).
3. **langchain deps for the user's notebook**: keep out of the package; optionally
   add `langchain-ollama`/`langchain-openai` as dev extras if the user wants the
   notebook runnable in-project (ask).
4. **Threshold calibration**: the T_risk/T_miss/T_conf numbers start as stated in
   configs and get calibrated on the 15 seed cases (Day 5) then re-checked on ~50 (Day 6).
5. **Clarify loop depth**: exactly ONE clarifying round, then a polite decline
   (prevents loops; recorded in the brief's trace).
6. **Trace volume**: JSONL per request, gitignored; Day 6's eval reads the same
   format — no second telemetry system.
