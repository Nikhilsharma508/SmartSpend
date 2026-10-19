"""The campaign copilot: Gate → Plan → Write → Validate, per request.

One public entry point, :func:`run_request`, runs the whole pipeline::

    marketer question
      ─[1] GATE─── Decisions API, 3 named questions (router.route_question)
      ─[2] PLAN─── pure Python: Scenario + Day-4 tools (planner.run_tools)
      ─[3] WRITE── pluggable writer: template / ollama / neurometric
      ─[4] VALIDATE─ code: number-trace, units, causal rubric, sections
      → BriefPackage (prose or a fast path, citations, trace id)

Gate-only paths (decline / redirect / clarify) are deterministic templates —
they never touch a writer, which is what keeps them fast and policy-exact.
Tools compute, the writer narrates, the validator decides: nothing reaches
the user whose numbers have not been traced back to a tool output.
"""

from __future__ import annotations

import time
from pathlib import Path

from packages.copilot.planner import HoldoutContext, extract_scenario, get_context, run_tools
from packages.copilot.providers import DecisionsClient, ProviderError
from packages.copilot.router import decide_fast_path, keyword_fallback_route, route_question
from packages.copilot.schemas import (
    BriefPackage,
    CopilotConfig,
    DraftRequest,
    load_config,
)
from packages.copilot.trace import make_logger, new_trace_id
from packages.copilot.writer import get_writer, write_brief

__all__ = ["BriefPackage", "CopilotConfig", "HoldoutContext", "run_request"]

#: T4 — refusals are rubric-controlled text, never generated.
DECLINE_POLICY_TEXT = (
    "I can't help target or exclude people by a sensitive attribute or a proxy "
    "for one (age, gender, race, religion, income, or location standing in for "
    "those). If it's useful, ask me to plan a campaign ranked by estimated "
    "incremental response instead — that targets behaviour the model can measure, "
    "not who people are."
)

#: T4 — out-of-scope gets a short redirect to what IS supported.
REDIRECT_TEXT = (
    "I'm built for four jobs: plan a campaign under a budget, price a what-if "
    "change, explain the model's customer segments, and size the next A/B test. "
    "Your question didn't match any of them — try rephrasing around one of "
    "those four."
)

_FIELD_LABELS = {
    "budget": "your budget in USD (e.g. $500)",
    "cost per contact": "the cost per contact (if it isn't the standard $0.10)",
    "depth": "how deep you're willing to target (e.g. the top 30%)",
}


def _clarify_text(missing_fields: list[str]) -> str:
    """T3 — one clarifying question, asking only for what is actually missing."""
    fields = ", ".join(_FIELD_LABELS.get(f, f) for f in missing_fields) or "a budget in USD"
    return (
        f"To make a recommendation I need one more number: {fields}. "
        "Reply with it and I'll run the optimizer immediately."
    )


def run_request(
    question: str,
    *,
    config: CopilotConfig | None = None,
    client: DecisionsClient | None = None,
    writer_provider: str | None = None,
    context: HoldoutContext | None = None,
) -> BriefPackage:
    """Answer one marketer question end-to-end; never raises for bad input.

    Dependencies are injectable (config, gate client, writer provider,
    holdout context) so tests run the identical pipeline with fakes.
    A missing ``.env`` degrades gracefully: the keyword router replaces the
    Decisions API, and the template writer replaces a generative one.
    """
    config = config if config is not None else load_config()
    trace_id = new_trace_id()
    trace = make_logger(trace_id, config.trace.enabled, config.trace.dir)
    started = time.perf_counter()
    stages: dict[str, float] = {}

    # ── [1] GATE ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if client is None:
        try:
            client = DecisionsClient.from_env()
        except ProviderError:
            client = None
    route = (
        route_question(question, client)
        if client is not None
        else keyword_fallback_route(question)
    )
    fast_path = decide_fast_path(route, config)
    stages["gate"] = time.perf_counter() - t0
    trace.log(
        "gate",
        "routed",
        intent=route.intent.value,
        confidence=route.intent_confidence,
        policy_risk=route.policy_risk,
        missing_inputs=route.missing_inputs,
        router=route.router,
        fast_path=fast_path,
    )

    # ── fast paths: no tools, no writer, no tokens ─────────────────────────
    if fast_path == "decline":
        trace.log("fast_path", "decline", reason="policy_risk")
        return BriefPackage(
            question=question,
            intent=route.intent,
            fast_path="decline",
            prose=DECLINE_POLICY_TEXT,
            decline_reason="policy_risk",
            trace_id=trace_id,
            latency_s=time.perf_counter() - started,
            stage_latencies=stages,
        )
    if fast_path == "redirect":
        trace.log("fast_path", "redirect", reason="out_of_scope_or_low_confidence")
        return BriefPackage(
            question=question,
            intent=route.intent,
            fast_path="redirect",
            prose=REDIRECT_TEXT,
            decline_reason="out_of_scope",
            trace_id=trace_id,
            latency_s=time.perf_counter() - started,
            stage_latencies=stages,
        )

    # ── [2] PLAN ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    scenario = extract_scenario(question, route.intent, config)
    if fast_path == "clarify":
        stages["plan"] = time.perf_counter() - t0
        clarify = _clarify_text(scenario.missing_fields)
        trace.log("fast_path", "clarify", missing=scenario.missing_fields)
        return BriefPackage(
            question=question,
            intent=route.intent,
            fast_path="clarify",
            prose=clarify,
            clarify_question=clarify,
            scenario=scenario,
            trace_id=trace_id,
            latency_s=time.perf_counter() - started,
            stage_latencies=stages,
        )

    holdout = context if context is not None else get_context()
    tool_runs = run_tools(scenario, holdout, config)
    stages["plan"] = time.perf_counter() - t0
    trace.log(
        "plan",
        "tools_done",
        scenario=scenario.model_dump(),
        tools=[r.model_dump() for r in tool_runs],
    )

    # ── [3] WRITE + [4] VALIDATE ───────────────────────────────────────────
    t0 = time.perf_counter()
    writer = get_writer(config, writer_provider)
    request = DraftRequest(
        intent=route.intent, question=question, scenario=scenario, tool_runs=tool_runs
    )
    final, report, attempts, degraded = write_brief(request, writer, config)
    stages["write_validate"] = time.perf_counter() - t0
    for i, attempt in enumerate(attempts):
        trace.log(
            "write",
            "draft",
            attempt=i,
            provider=attempt.provider,
            latency_s=round(attempt.latency_s, 3),
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
            passed=report.passed if i == len(attempts) - 1 else False,
        )
    trace.log(
        "validate",
        "report",
        passed=report.passed,
        degraded=degraded,
        checks=[c.model_dump() for c in report.checks],
        citations=[c.model_dump() for c in report.citations],
    )

    package = BriefPackage(
        question=question,
        intent=route.intent,
        fast_path="brief",
        prose=final.text,
        scenario=scenario,
        tool_runs=tool_runs,
        citations=report.citations,
        validation=report,
        degraded=degraded,
        provider=final.provider,
        draft_attempts=len(attempts),
        trace_id=trace_id,
        latency_s=time.perf_counter() - started,
        stage_latencies=stages,
    )
    trace.log("package", "done", fast_path="brief", degraded=degraded)
    return package


def _load_trace_dir() -> Path:  # pragma: no cover - convenience for notebooks
    return Path(load_config().trace.dir)
