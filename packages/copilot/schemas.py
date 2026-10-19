"""Typed schemas for the Day-5 campaign copilot.

The pipeline is Gate → Plan → Write → Validate; every stage boundary is one of
these models, so a request is a serialisable object at every point — which is
exactly what the JSONL trace in :mod:`packages.copilot.trace` records.

Naming note: the planner's input is :class:`CampaignScenario`, deliberately
*not* ``Scenario`` — :mod:`packages.decision.schemas` already owns that name
for "one evaluated targeting strategy", and the two must never be confused.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

#: Repo-root configs/ directory (same convention as the registry module).
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class Intent(StrEnum):
    """The five routes the gate can produce (four use cases + the catch-all)."""

    PLAN = "plan"
    WHAT_IF = "what_if"
    EXPLAIN = "explain"
    TEST_DESIGN = "test_design"
    OUT_OF_SCOPE = "out_of_scope"


#: Which template answered the request. Fast paths never touch a writer.
FastPath = Literal["brief", "clarify", "decline", "redirect"]

#: The section contract per intent — the prompt instructs the writer to use
#: these headers, the template emits them, and validator rule V4 requires them.
BRIEF_SECTIONS: dict[Intent, tuple[str, ...]] = {
    Intent.PLAN: ("recommendation", "what it earns", "uncertainties", "assumptions", "next test"),
    Intent.WHAT_IF: ("recommendation", "what it earns", "uncertainties", "assumptions"),
    Intent.EXPLAIN: ("recommendation", "segments", "uncertainties", "assumptions"),
    Intent.TEST_DESIGN: ("recommendation", "what to watch", "uncertainties", "assumptions"),
}


class Route(BaseModel):
    """Gate output: the routed intent plus the two gate probabilities."""

    intent: Intent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    policy_risk: float = Field(ge=0.0, le=1.0)
    missing_inputs: float = Field(ge=0.0, le=1.0)
    router: Literal["decisions", "keyword_fallback"] = "decisions"
    answers: dict[str, Any] = Field(
        default_factory=dict, description="Raw gate answers, kept verbatim for the trace."
    )


class CampaignScenario(BaseModel):
    """Planner input: everything the tools need; every default is flagged.

    ``missing_fields`` is what the gate judged absent — it drives the one
    clarifying question, so the planner never has to invent money.
    """

    intent: Intent
    question: str
    budget_usd: float | None = Field(default=None, gt=0)
    cost_per_contact_usd: float | None = Field(default=None, gt=0)
    depth: float | None = Field(
        default=None, gt=0, le=0.5,
        description="Targeting depth; when budget is given it is derived from it.",
    )
    mde: float | None = Field(default=None, gt=0, description="Absolute MDE for test design.")
    daily_eligible: int | None = Field(default=None, gt=0)
    assumptions_used: list[str] = Field(
        default_factory=list, description="Every default filled in, quoted in the brief."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields absent from the question (drives the clarify text).",
    )


class ToolRun(BaseModel):
    """One tool invocation: the output dict on success, error text on failure.

    A failed tool stays in the list — the writer sees the failure and can
    never invent a number for a tool that did not run.
    """

    name: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None


class DraftRequest(BaseModel):
    """Everything a writer sees: question, scenario, tool outputs verbatim."""

    intent: Intent
    question: str
    scenario: CampaignScenario
    tool_runs: list[ToolRun]
    retry_errors: list[str] = Field(
        default_factory=list, description="Validator failures from the previous draft (T5 retry)."
    )


class DraftResult(BaseModel):
    """One draft from one provider, with its cost/latency evidence."""

    text: str
    provider: str
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


class Citation(BaseModel):
    """One number in the brief traced back to where it came from."""

    value: float
    source: str = Field(
        description="Tool-output path, e.g. 'optimize_budget.recommended_profit_usd'."
    )


class ValidationCheck(BaseModel):
    """One validator rule's verdict (V1 number-trace, V2 units, ...)."""

    rule: str
    passed: bool
    detail: str


class ValidationReport(BaseModel):
    """The full battery for one draft; citations are V1's positive findings."""

    checks: list[ValidationCheck]
    citations: list[Citation] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]


class BriefPackage(BaseModel):
    """The copilot's answer: prose (or a fast path) plus its verification."""

    question: str
    intent: Intent
    fast_path: FastPath
    prose: str = ""
    clarify_question: str | None = None
    decline_reason: str | None = None
    scenario: CampaignScenario | None = None
    tool_runs: list[ToolRun] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    validation: ValidationReport | None = None
    degraded: bool = Field(
        default=False, description="True when [UNVERIFIED] flags were appended after max rewrites."
    )
    provider: str = Field(default="template", description="Writer that produced the prose.")
    router: Literal["decisions", "keyword_fallback"] = Field(
        default="decisions", description="Gate that produced the route (live or keyword)."
    )
    draft_attempts: int = Field(
        default=1, description="Drafts made before this one (0-based retry count)."
    )
    trace_id: str = ""
    latency_s: float = 0.0
    stage_latencies: dict[str, float] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """One row in traces/<date>.jsonl — the visibility layer, stage by stage."""

    ts: str
    trace_id: str
    stage: str
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Configuration (configs/copilot.yaml — thresholds, tolerances, writer pick)
# --------------------------------------------------------------------------


class RouterConfig(BaseModel):
    """Gate thresholds. Calibrated on evals/seed_cases.yaml (Day 5/6).

    Applied by :func:`packages.copilot.router.decide_fast_path` in order:
    decline > confident out_of_scope redirect > clarify > cautious redirect > brief.
    """

    t_policy_risk: float = Field(default=0.70, ge=0.0, le=1.0)
    t_missing: float = Field(default=0.60, ge=0.0, le=1.0)
    t_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    t_oos_confidence: float = Field(default=0.90, ge=0.0, le=1.0)


class PlannerConfig(BaseModel):
    """Documented defaults the planner fills in and flags as assumptions."""

    default_depth: float = Field(default=0.30, gt=0, le=0.5)
    max_depth: float = Field(default=0.50, gt=0, le=1.0)
    default_mde: float = Field(default=0.01, gt=0)
    default_daily_eligible: int = Field(default=2000, gt=0)


class ValidatorConfig(BaseModel):
    """V1 tolerance, retry budget, and the bare constants prose may cite."""

    rel_tolerance: float = Field(default=0.01, gt=0)
    max_rewrites: int = Field(default=1, ge=0)
    allowed_constants: list[float] = Field(default_factory=lambda: [95.0])


class WriterConfig(BaseModel):
    """Which provider writes briefs, and how to reach each one."""

    provider: Literal["template", "ollama", "neurometric"] = "ollama"
    timeout_s: float = Field(default=60.0, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b-cloud"
    neurometric_model: str = "neurometric/goal-setter"


class PromptConfig(BaseModel):
    """Prompt provenance: what wrote the prose can be audited from the trace."""

    prompt_version: str = "v1"


class TraceConfig(BaseModel):
    """JSONL trace destination; disabled turns the logger into a no-op."""

    enabled: bool = True
    dir: str = "traces"


class CopilotConfig(BaseModel):
    """The whole pipeline's knobs — loaded from configs/copilot.yaml."""

    router: RouterConfig = Field(default_factory=RouterConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    writer: WriterConfig = Field(default_factory=WriterConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)


def load_config(path: Path | None = None) -> CopilotConfig:
    """Load ``configs/copilot.yaml``; a missing file falls back to defaults.

    Raises
    ------
    ValueError
        The YAML is unparseable or a value fails its schema (a typo'd
        threshold should stop the request, not silently route it).
    """
    path = path or CONFIGS_DIR / "copilot.yaml"
    if not path.exists():
        return CopilotConfig()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as err:
        raise ValueError(f"copilot config {path} is not valid YAML: {err}") from err
    return CopilotConfig.model_validate(raw)
