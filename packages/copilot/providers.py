"""Thin provider clients for the copilot — notebook-style raw ``requests``.

Four providers, no langchain (its calls stay in the user's reference notebook
``research/notebooks/decision_api.ipynb``; the package never needs them):

- :class:`DecisionsClient` — the Azure Decisions API gate, request shape
  ``{"model", "state", "questions"}`` exactly as verified in that notebook.
- :class:`TemplateWriter` — deterministic house-template render, no network,
  always available; the fallback every generative path degrades to.
- :class:`OllamaWriter` — OpenAI-compatible ``/v1/chat/completions`` on Ollama
  (``gpt-oss:120b-cloud`` by default).
- :class:`NeurometricWriter` — OpenAI-compatible ``/chat/completions`` on the
  NEUROMETRIC gateway (``neurometric/goal-setter`` by default).

Prompt assembly (:func:`build_prompt`) lives here rather than in ``writer.py``
because the provider owns its prompt contract: this module defines both the
messages and the clients that send them, which keeps the import graph acyclic
(``writer.py`` orchestrates providers, never the reverse).

Every failure raises :class:`ProviderError` carrying the provider *name* —
never a key, endpoint or raw response body (``.env`` values are private).
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import requests

from packages.causal.benchmark import BusinessAssumptions
from packages.copilot.schemas import (
    BRIEF_SECTIONS,
    DraftRequest,
    DraftResult,
    Intent,
    ToolRun,
)

#: Environment variable names the live clients read (names only — never values).
DECISIONS_ENV_VARS = ("AZURE_ENDPOINT", "AZURE_API_KEY", "AZURE_DEPLOYMENT_NAME")
NEUROMETRIC_ENV_VARS = ("NEUROMETRIC_BASE_URL", "NEUROMETRIC_API")

#: Path suffix of the Decisions endpoint, exactly as the notebook verified it.
DECISIONS_PATH = "/providers/microsoft/v1/systemone"


class ProviderError(RuntimeError):
    """A provider was unreachable, refused, or returned an unparseable body."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


# ---------------------------------------------------------------------------
# [1] GATE — the Decisions API client
# ---------------------------------------------------------------------------


class DecisionsClient:
    """One-call client for the Azure Decisions API (gate/router).

    ``endpoint`` is the bare host from ``AZURE_ENDPOINT``; the API path is
    appended exactly as ``decision_api.ipynb`` does it.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 45.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls, *, timeout: float = 45.0) -> DecisionsClient:
        """Build from ``AZURE_*``; raise listing the missing *names* if any."""
        missing = [v for v in DECISIONS_ENV_VARS if not os.getenv(v)]
        if missing:
            raise ProviderError(
                "decisions", f"missing environment variable(s): {missing} (see .env.example)"
            )
        return cls(
            endpoint=os.environ["AZURE_ENDPOINT"],
            api_key=os.environ["AZURE_API_KEY"],
            model=os.environ["AZURE_DEPLOYMENT_NAME"],
            timeout=timeout,
        )

    def decide(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Ask every named question in one request; return the raw JSON body.

        A *refusal* is a normal answer here (some question will simply be
        refused); the router turns it into the keyword fallback. Network and
        HTTP errors raise :class:`ProviderError` instead.
        """
        payload: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        url = f"{self.endpoint}{DECISIONS_PATH}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as err:
            raise ProviderError("decisions", f"request failed: {err}") from err
        except ValueError as err:
            raise ProviderError("decisions", f"response was not JSON: {err}") from err


# ---------------------------------------------------------------------------
# [3] WRITE — prompt assembly + the writer abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatPrompt:
    """The exact system/user pair a chat-completions writer receives."""

    system: str
    user: str


#: Fixed system prompt, versioned by ``configs/copilot.yaml`` ``prompt_version``.
#: The DATA block carries every number; the model's only job is narration.
SYSTEM_PROMPT = """\
You are an analyst at a causal-promotion targeting vendor. You NARRATE results \
computed by marketing-science tools; you never compute, estimate or round anything.

Hard rules:
1. Every number in your brief must appear VERBATIM in the DATA block below. \
Never invent, re-round, average or extrapolate a number, and never DERIVE one: \
no counts re-expressed as percentages or shares, no relative or percent changes \
computed from an interval, no arithmetic between two DATA figures. Quote a count \
as a count and a fraction as a fraction (the DATA's own percent form is fine).
2. Where the DATA shows an interval, show that interval in the same sentence.
3. Causal care: say "the model estimates" or "the holdout shows" - never "will", \
"guaranteed", "proves", or "because they are ...". A policy segment shows WHO \
responds more, never WHY.
4. If a tool in the DATA failed, write "[tool unavailable]" in that spot; do \
not quote the error text and do not fill the gap with a number.
5. Currency is USD everywhere. Keep depth figures as percentages.

Write in plain markdown for a marketing decision-maker. Use exactly these \
section headers, in this order: {sections}. Keep it under about 400 words.\
"""


#: Provenance-only tool fields the writer never quotes. They stay in the API
#: result but are omitted from the LLM payload and the corresponding trace span.
_OMIT_FROM_NARRATION: dict[str, frozenset[str]] = {
    "get_model_card": frozenset({"data_sha256"}),
}


def trim_tool_output(name: str, output: Any) -> Any:
    """Drop provenance-only fields (e.g. the dataset SHA) before narration/tracing."""
    omit = _OMIT_FROM_NARRATION.get(name)
    if not omit or not isinstance(output, dict):
        return output
    return {key: value for key, value in output.items() if key not in omit}


def _tool_payload(request: DraftRequest) -> dict[str, Any]:
    """Tool runs as one JSON-able dict: success output, or the failure stub."""
    tools: dict[str, Any] = {}
    for run in request.tool_runs:
        tools[run.name] = (
            trim_tool_output(run.name, run.output)
            if run.ok
            else {"UNAVAILABLE": run.error or "tool failed"}
        )
    return {
        "scenario": request.scenario.model_dump(exclude={"question"}),
        "tools": tools,
        "assumptions_used": request.scenario.assumptions_used,
    }


def build_prompt(request: DraftRequest) -> ChatPrompt:
    """Assemble the messages for one draft (T1/T2 body; T5 rides on retries)."""
    sections = " / ".join(s.title() for s in BRIEF_SECTIONS[request.intent])
    system = SYSTEM_PROMPT.format(sections=sections)

    user = (
        f"QUESTION: {request.question}\n"
        f"INTENT: {request.intent.value}\n"
        f"DATA (JSON, computed by the tools - quote these numbers verbatim):\n"
        f"{json.dumps(_tool_payload(request), indent=1, default=str)}\n"
    )
    if request.retry_errors:
        user += (
            "Your previous draft FAILED these validation checks - fix exactly these "
            "and keep everything else:\n"
            + "\n".join(f"- {e}" for e in request.retry_errors)
            + "\n"
        )
    return ChatPrompt(system=system, user=user)


class WriterBase(ABC):
    """Anything that can turn a :class:`DraftRequest` into prose."""

    name: ClassVar[str]

    @abstractmethod
    def write(self, request: DraftRequest) -> DraftResult:
        """Render/draft the brief; raise :class:`ProviderError` when impossible."""


class TemplateWriter(WriterBase):
    """Deterministic house-template render — no LLM, no network, always works.

    It is both the zero-dependency fallback and the validator's sanity floor:
    tests assert its output passes the full battery for all four intents, so
    the copilot can always answer something true.
    """

    name: ClassVar[str] = "template"

    def write(self, request: DraftRequest) -> DraftResult:
        started = time.perf_counter()
        text = _render_template(request)
        return DraftResult(text=text, provider=self.name, latency_s=time.perf_counter() - started)


# --- deterministic template helpers ----------------------------------------


def _usd(v: float) -> str:
    return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"


def _pct(frac: float) -> str:
    """Percent at one decimal, trailing zeros stripped — stays inside the
    validator's 1% tolerance without writing '30.0%' for a clean 30%."""
    text = f"{frac * 100:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _runs_by_name(request: DraftRequest) -> dict[str, dict[str, Any] | None]:
    return {r.name: (r.output if r.ok else None) for r in request.tool_runs}


def _fmt_assumptions(request: DraftRequest) -> str:
    lines = [
        "- value per conversion and cost per contact come from "
        "configs/business_assumptions.yaml (fixed, not estimated per request)"
    ]
    lines += [f"- {a}" for a in request.scenario.assumptions_used]
    return "\n".join(lines)


def _render_template(request: DraftRequest) -> str:
    """House template per intent; every printed number comes from a tool run."""
    tools = _runs_by_name(request)
    intent = request.intent

    if intent is Intent.EXPLAIN:
        return _render_explain(request, tools)
    if intent is Intent.TEST_DESIGN:
        return _render_test_design(request, tools)
    return _render_plan_or_whatif(request, tools)


def _render_plan_or_whatif(request: DraftRequest, tools: dict[str, dict[str, Any] | None]) -> str:
    point = tools.get("profit_point") or {}
    curve = tools.get("optimize_budget") or {}
    card = tools.get("get_model_card") or {}
    design = tools.get("design_ab_test") or {}
    assumptions_block = f"## Assumptions\n{_fmt_assumptions(request)}\n"

    if request.intent is Intent.WHAT_IF:
        base = tools.get("compare_scenarios_base") or {}
        variant = tools.get("compare_scenarios_variant") or {}
        delta = tools.get("what_if_delta") or {}
        champ_b, champ_v = _champion(base), _champion(variant)
        recommendation = (
            f"## Recommendation\nPrice the change before committing: under the variant the same "
            f"ranking earns {_usd(champ_v.get('profit_usd', 0.0))} versus "
            f"{_usd(champ_b.get('profit_usd', 0.0))} today - a difference of "
            f"{_usd(delta.get('profit_delta_usd', 0.0))}.\n"
        )
        earns = (
            f"## What it earns\n"
            f"- changed input: {delta.get('changed', 'the stated assumption')}\n"
            f"- champion profit, base: {_usd(champ_b.get('profit_usd', 0.0))} "
            f"(95% interval [{_usd(champ_b.get('profit_ci_low_usd', 0.0))}, "
            f"{_usd(champ_b.get('profit_ci_high_usd', 0.0))}])\n"
            f"- champion profit, variant: {_usd(champ_v.get('profit_usd', 0.0))} "
            f"(95% interval [{_usd(champ_v.get('profit_ci_low_usd', 0.0))}, "
            f"{_usd(champ_v.get('profit_ci_high_usd', 0.0))}])\n"
            f"- change in profit: {_usd(delta.get('profit_delta_usd', 0.0))}\n"
            f"- contacts bought by the same budget: {champ_b.get('n_targeted', 0):,.0f} -> "
            f"{champ_v.get('n_targeted', 0):,.0f}\n"
        )
    else:
        depth = point.get("depth", 0.0)
        n_k = point.get("n_targeted", 0)
        cpc = curve.get("assumptions", {}).get("cost_per_contact_usd", 0.10)
        break_even = curve.get("break_even_depth")
        recommendation = (
            f"## Recommendation\nTarget the top {_pct(depth)} of the ranked customer list "
            f"({n_k:,.0f} contacts at {_usd(cpc)} each). The sweep does not clear its own contact "
            f"cost until {_pct(break_even if break_even is not None else depth)}, so do not widen "
            f"the list beyond that on this evidence.\n"
        )
        if point:
            earns = (
                f"## What it earns\nThe champion ranking is estimated to deliver "
                f"{point.get('incremental_conversions', 0.0)} incremental conversions "
                f"(standard error {point.get('incremental_conversions_se', 0.0)}) and a profit of "
                f"{_usd(point.get('profit_usd', 0.0))} (95% interval "
                f"[{_usd(point.get('profit_ci_low_usd', 0.0))}, "
                f"{_usd(point.get('profit_ci_high_usd', 0.0))}]).\n"
            )
        else:
            earns = "## What it earns\n[tool unavailable]\n"

    uncertainties = (
        f"## Uncertainties\nThe interval above is a normal approximation on one holdout; when it "
        f"spans zero the honest read is a range, not a promise. The depth sweep only breaks even "
        f"from {_pct(curve.get('break_even_depth') or 0.0)}, and its point-estimate optimum "
        f"({_pct(curve.get('recommended_depth') or 0.0)}, "
        f"{_usd(curve.get('recommended_profit_usd') or 0.0)}) is a single-split estimate, not a "
        f"measurement.\n"
    )
    tail = ""
    if request.intent is Intent.PLAN:
        if design:
            mde_pp = f"{design.get('mde_absolute', 0.01) * 100:g}"
            tail = (
                f"## Next test\nConfirm on live traffic before scaling: at a "
                f"+{mde_pp} percentage-point MDE "
                f"on the visit rate, plan for {design.get('n_per_arm', 0):,.0f} "
                f"customers per arm ({design.get('n_total', 0):,.0f} total, "
                f"{design.get('duration_days', 0)} days at "
                f"{design.get('daily_eligible', 0):,.0f} eligible per day).\n"
            )
        else:
            tail = "## Next test\n[tool unavailable]\n"
        if card:
            tail += (
                f"\nModel: {card.get('name', 'champion')} "
                f"({card.get('learner', '?')} on {card.get('target', '?')}).\n"
            )
    return recommendation + earns + uncertainties + assumptions_block + tail


def _champion(comparison: dict[str, Any]) -> dict[str, Any]:
    """The champion row of a ScenarioComparison dump (empty dict if absent)."""
    for s in comparison.get("scenarios", []):
        if s.get("strategy") == "champion ranking":
            return s
    return {}


def _render_explain(request: DraftRequest, tools: dict[str, dict[str, Any] | None]) -> str:
    profile = tools.get("get_segment_profile") or {}
    segments = profile.get("segments", [])
    fairness = profile.get("fairness", [])

    target = [s for s in segments if s.get("policy_arm") == "email"]
    avoid = [s for s in segments if s.get("policy_arm") != "email"]
    top = max(target, key=lambda s: s.get("uplift_visit", 0.0), default=None)

    recommendation = (
        "## Recommendation\n"
        + (
            f"Prioritise the {len(target)} segment(s) the policy tree labels for email "
            f"({', '.join('segment ' + str(s['segment_id']) for s in target)}) and hold back the "
            f"{len(avoid)} labelled no-email."
            if target
            else "The fitted policy tree labels no segment for email on this holdout - do not "
            "target on this evidence."
        )
        + "\n"
    )
    lines = ["## Segments"]
    for s in segments:
        rules = ", ".join(s.get("rules", [])) or "root (no split rule)"
        arm = s.get("policy_arm", "?")
        lo = s.get("uplift_visit_ci_low", 0.0)
        hi = s.get("uplift_visit_ci_high", 0.0)
        share = _pct(s.get("share", 0.0))
        lines.append(
            f"- Segment {s.get('segment_id')} ({rules}): {share} of the holdout "
            f"({s.get('n_eval', 0):,.0f} customers); policy says **{arm}**; measured visit uplift "
            f"{s.get('uplift_visit', 0.0):+.4f} (95% interval [{lo:+.4f}, {hi:+.4f}])."
        )
    if top:
        lines.append(
            f"The strongest email-labelled segment is segment {top.get('segment_id')} at "
            f"{top.get('uplift_visit', 0.0):+.4f} on visits."
        )
    segments_block = "\n".join(lines) + "\n"

    fairness_lines = ["## Uncertainties", profile.get("note", "")]
    for row in fairness:
        lo = row.get("uplift_visit_ci_low", 0.0)
        hi = row.get("uplift_visit_ci_high", 0.0)
        fairness_lines.append(
            f"- {row.get('group', '?')}: contacted at {_pct(row.get('targeting_rate', 0.0))} "
            f"({row.get('n_eval', 0):,.0f} customers), uplift "
            f"{row.get('uplift_visit', 0.0):+.4f} (95% interval [{lo:+.4f}, {hi:+.4f}]). "
            f"{row.get('note', '')}"
        )
    uncertainties = "\n".join(fairness_lines) + "\n"
    assumptions = f"## Assumptions\n{_fmt_assumptions(request)}\n"
    return recommendation + segments_block + uncertainties + assumptions


def _render_test_design(request: DraftRequest, tools: dict[str, dict[str, Any] | None]) -> str:
    design = tools.get("design_ab_test") or {}
    recommendation = (
        f"## Recommendation\nRun the confirmatory test for {design.get('duration_days', 0)} days: "
        f"{design.get('n_per_arm', 0):,.0f} customers per arm "
        f"({design.get('n_total', 0):,.0f} total) at {design.get('daily_eligible', 0):,.0f} "
        f"eligible per day, splitting traffic evenly between email and no-email.\n"
    )
    baseline = design.get("baseline_rate", 0.0)
    mde_abs = f"{design.get('mde_absolute', 0.0) * 100:g}"
    mde_rel = design.get("mde_relative", 0.0)
    watch = (
        f"## What to watch\nPrimary metric: visit rate, baseline {baseline:.4f} "
        f"in control. Detect an absolute lift of {mde_abs} percentage "
        f"points ({mde_rel:.2%} relative) at alpha "
        f"{design.get('alpha', 0.05)} and power {design.get('power', 0.8)}; "
        f"call it success only if the confidence interval excludes zero.\n"
    )
    uncertainties = (
        "## Uncertainties\nThis sizing uses the classic two-proportion normal approximation with "
        "the baseline treated as known, so keep headroom; the baseline itself is estimated from "
        "one holdout.\n"
    )
    assumptions = f"## Assumptions\n{_fmt_assumptions(request)}\n"
    return recommendation + watch + uncertainties + assumptions


# --- generative chat writers ----------------------------------------------


class ChatWriterBase(WriterBase):
    """Shared thin client for OpenAI-compatible ``/chat/completions`` APIs."""

    name: ClassVar[str] = "chat"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def write(self, request: DraftRequest) -> DraftResult:
        prompt = build_prompt(request)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except requests.RequestException as err:
            raise ProviderError(self.name, f"request failed: {err}") from err
        except (ValueError, KeyError, IndexError, TypeError) as err:
            raise ProviderError(self.name, f"unexpected response shape: {err}") from err
        usage = data.get("usage") or {}
        return DraftResult(
            text=content,
            provider=self.name,
            latency_s=time.perf_counter() - started,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


class OllamaWriter(ChatWriterBase):
    """Ollama's OpenAI-compatible endpoint; no key by default (local/cloud CLI auth).

    ``base_url`` is the *server root* (``http://localhost:11434``); Ollama serves
    the OpenAI-compatible API under ``/v1``, which is appended when missing so a
    root URL and a ``/v1`` URL both resolve to ``/v1/chat/completions``.
    """

    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        super().__init__(model, base, api_key=api_key, timeout=timeout)


class NeurometricWriter(ChatWriterBase):
    """The NEUROMETRIC gateway; ``NEUROMETRIC_API`` rides as a Bearer token."""

    name: ClassVar[str] = "neurometric"

    @classmethod
    def from_env(
        cls, model: str, *, base_url: str | None = None, timeout: float = 60.0
    ) -> NeurometricWriter:
        missing = [v for v in NEUROMETRIC_ENV_VARS if not os.getenv(v)]
        if missing:
            raise ProviderError("neurometric", f"missing environment variable(s): {missing}")
        return cls(
            model=model,
            base_url=base_url or os.environ["NEUROMETRIC_BASE_URL"],
            api_key=os.environ["NEUROMETRIC_API"],
            timeout=timeout,
        )


def default_assumptions() -> BusinessAssumptions:
    """The decided profit constants (single source: business_assumptions.yaml)."""
    return BusinessAssumptions.from_yaml()


def tool_run(name: str, output: Any) -> ToolRun:
    """Wrap a tool's Pydantic output (or raw dict) as a successful :class:`ToolRun`."""
    if hasattr(output, "model_dump"):
        output = output.model_dump()
    return ToolRun(name=name, ok=True, output=dict(output))
