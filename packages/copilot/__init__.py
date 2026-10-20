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

from packages.copilot.planner import HoldoutContext, extract_scenario, get_context, run_tools
from packages.copilot.providers import DecisionsClient, ProviderError
from packages.copilot.router import (
    build_questions,
    build_state,
    decide_fast_path,
    keyword_fallback_route,
    route_question,
)
from packages.copilot.schemas import (
    BriefPackage,
    CopilotConfig,
    DraftRequest,
    load_config,
)
from packages.copilot.trace import new_trace_id
from packages.copilot.writer import get_writer, write_brief
from packages.tracking import (
    annotate_trace,
    request_trace,
    set_span_outputs,
    trace_span,
)

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
    mlflow_trace: bool = False,
) -> BriefPackage:
    """Answer one marketer question end-to-end; never raises for bad input.

    Dependencies are injectable (config, gate client, writer provider,
    holdout context) so tests run the identical pipeline with fakes.
    A missing ``.env`` degrades gracefully: the keyword router replaces the
    Decisions API, and the template writer replaces a generative one.

    ``mlflow_trace`` opens one GenAI trace (gate → plan → write → validate)
    under the ``smartspend`` experiment. It is off by default so tests and
    library callers never touch the tracking store.
    """
    if not mlflow_trace:
        return _run_request(
            question,
            config=config,
            client=client,
            writer_provider=writer_provider,
            context=context,
        )

    with request_trace(question, writer_provider) as span:
        package = _run_request(
            question,
            config=config,
            client=client,
            writer_provider=writer_provider,
            context=context,
        )
        set_span_outputs(
            span,
            {
                "fast_path": package.fast_path,
                "intent": package.intent.value,
                "provider": package.provider,
                "trace_id": package.trace_id,
                "latency_s": round(package.latency_s, 3),
                "draft_attempts": package.draft_attempts,
                "degraded": package.degraded,
                "router": package.router,
                "decline_reason": package.decline_reason,
                "stage_latencies": {
                    key: round(value, 3)
                    for key, value in package.stage_latencies.items()
                },
            },
        )
        annotate_trace(
            trace_id=package.trace_id,
            request_preview=question,
            response_preview=package.prose,
            tags={
                "provider": package.provider,
                "fast_path": package.fast_path,
                "intent": package.intent.value,
                "llm.decision": package.router,
                "llm.response": (
                    package.provider if package.fast_path == "brief" else "none"
                ),
                **(
                    {"decline_reason": package.decline_reason}
                    if package.decline_reason
                    else {}
                ),
            },
        )
        return package


def _run_request(
    question: str,
    *,
    config: CopilotConfig | None = None,
    client: DecisionsClient | None = None,
    writer_provider: str | None = None,
    context: HoldoutContext | None = None,
) -> BriefPackage:
    """The pipeline body, shared by traced and untraced callers."""
    config = config if config is not None else load_config()
    trace_id = new_trace_id()
    started = time.perf_counter()
    stages: dict[str, float] = {}

    # ── [1] GATE ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if client is None:
        try:
            client = DecisionsClient.from_env()
        except ProviderError:
            client = None

    if client is not None:
        gate_state = build_state(question)
        gate_questions = build_questions()
        gate_name = "decision_llm"
        gate_inputs = {
            "model": getattr(client, "model", "unknown"),
            "state": gate_state,
            "questions": gate_questions,
        }
        gate_attributes = {
            "smartspend.llm_role": "decision",
            "llm.provider": "decisions",
            "llm.model": getattr(client, "model", "unknown"),
        }
    else:
        gate_state = None
        gate_questions = None
        gate_name = "decision_router_fallback"
        gate_inputs = {"question": question, "router": "keyword_fallback"}
        gate_attributes = {
            "smartspend.llm_role": "decision",
            "llm.provider": "keyword_fallback",
            "llm.model": "none",
        }

    with trace_span(
        gate_name,
        "LLM" if client is not None else "CHAIN",
        inputs=gate_inputs,
        attributes=gate_attributes,
    ) as span:
        route = (
            route_question(question, client, state=gate_state, questions=gate_questions)
            if client is not None
            else keyword_fallback_route(question)
        )
        fast_path = decide_fast_path(route, config)
        stages["gate"] = time.perf_counter() - t0
        set_span_outputs(
            span,
            {
                "intent": route.intent.value,
                "confidence": route.intent_confidence,
                "policy_risk": route.policy_risk,
                "router": route.router,
                "fast_path": fast_path,
                "missing_inputs": route.missing_inputs,
            },
        )

    # ── fast paths: no tools, no writer, no tokens ─────────────────────────
    if fast_path == "decline":
        return BriefPackage(
            question=question,
            intent=route.intent,
            fast_path="decline",
            prose=DECLINE_POLICY_TEXT,
            decline_reason="policy_risk",
            router=route.router,
            trace_id=trace_id,
            latency_s=time.perf_counter() - started,
            stage_latencies=stages,
        )
    if fast_path == "redirect":
        return BriefPackage(
            question=question,
            intent=route.intent,
            fast_path="redirect",
            prose=REDIRECT_TEXT,
            decline_reason="out_of_scope",
            router=route.router,
            trace_id=trace_id,
            latency_s=time.perf_counter() - started,
            stage_latencies=stages,
        )

    # ── [2] PLAN ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with trace_span("plan", "CHAIN", inputs={"intent": route.intent.value}) as span:
        scenario = extract_scenario(question, route.intent, config)
        if fast_path == "clarify":
            stages["plan"] = time.perf_counter() - t0
            clarify = _clarify_text(scenario.missing_fields)
            set_span_outputs(
                span,
                {"fast_path": "clarify", "missing": scenario.missing_fields},
            )
            return BriefPackage(
                question=question,
                intent=route.intent,
                fast_path="clarify",
                prose=clarify,
                clarify_question=clarify,
                scenario=scenario,
                router=route.router,
                trace_id=trace_id,
                latency_s=time.perf_counter() - started,
                stage_latencies=stages,
            )

        holdout = context if context is not None else get_context()
        tool_runs = run_tools(scenario, holdout, config)
        stages["plan"] = time.perf_counter() - t0
        set_span_outputs(
            span,
            {
                "scenario": scenario.model_dump(),
                "tools": [r.name for r in tool_runs],
            },
        )

    # ── [3] WRITE + [4] VALIDATE ───────────────────────────────────────────
    t0 = time.perf_counter()
    writer = get_writer(config, writer_provider)
    request = DraftRequest(
        intent=route.intent, question=question, scenario=scenario, tool_runs=tool_runs
    )
    with trace_span(
        "write_validate", "CHAIN", inputs={"provider": writer.name}
    ) as span:
        final, report, attempts, degraded = write_brief(request, writer, config)
        set_span_outputs(
            span,
            {
                "provider": final.provider,
                "attempts": len(attempts),
                "passed": report.passed,
                "degraded": degraded,
                "citations": len(report.citations),
            },
        )
    stages["write_validate"] = time.perf_counter() - t0

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
        router=route.router,
        draft_attempts=len(attempts),
        trace_id=trace_id,
        latency_s=time.perf_counter() - started,
        stage_latencies=stages,
    )
    return package
