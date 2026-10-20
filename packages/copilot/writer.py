"""The [3] WRITE stage — provider selection plus the draft→validate→retry loop.

The writer's contract is narration only: :func:`build_prompt` (in
``providers.py``) has already injected every number, and this module's loop
drafts, validates, and — on failure — re-drafts **once** with the validator's
error list attached (T5). A second failure never silently passes: the package
returns the prose with explicit ``[UNVERIFIED: …]`` flags and
``degraded=True``, so the user sees exactly what could not be verified.
"""

from __future__ import annotations

from typing import Any

from packages.copilot.providers import (
    ChatWriterBase,
    NeurometricWriter,
    OllamaWriter,
    ProviderError,
    TemplateWriter,
    WriterBase,
    build_prompt,
)
from packages.copilot.schemas import (
    CopilotConfig,
    DraftRequest,
    DraftResult,
    ValidationReport,
)
from packages.copilot.validator import validate
from packages.tracking import set_span_outputs, trace_span

#: Marker appended when the draft fails validation after the retry budget.
UNVERIFIED_MARKER = "[UNVERIFIED"


def get_writer(config: CopilotConfig, provider: str | None = None) -> WriterBase:
    """Instantiate the configured writer (``provider`` overrides for tests/ablation).

    Raises
    ------
    ValueError
        Unknown provider name — a typo must stop the request, not fall
        silently back to the template.
    ProviderError
        A generative provider is configured but its environment is missing.
    """
    name = provider or config.writer.provider
    if name == "template":
        return TemplateWriter()
    if name == "ollama":
        return OllamaWriter(
            model=config.writer.ollama_model,
            base_url=config.writer.ollama_base_url,
            timeout=config.writer.timeout_s,
        )
    if name == "neurometric":
        return NeurometricWriter.from_env(
            model=config.writer.neurometric_model, timeout=config.writer.timeout_s
        )
    raise ValueError(
        f"unknown writer provider {name!r} (expected template | ollama | neurometric)"
    )


def write_brief(
    request: DraftRequest,
    writer: WriterBase,
    config: CopilotConfig,
) -> tuple[DraftResult, ValidationReport, list[DraftResult], bool]:
    """Draft, validate, and retry once — the T1/T5 loop in one place.

    Returns
    -------
    (final, report, attempts, degraded)
        ``final`` is the draft that was returned (validated or flagged),
        ``attempts`` every draft made along the way (for the trace), and
        ``degraded`` whether ``[UNVERIFIED: …]`` flags were appended.
    """
    attempts: list[DraftResult] = []
    retry_errors: list[str] = []
    final: DraftResult | None = None
    report: ValidationReport | None = None

    for _round in range(1 + config.validator.max_rewrites + 1):
        draft_request = request.model_copy(update={"retry_errors": retry_errors})
        prompt = build_prompt(draft_request) if isinstance(writer, ChatWriterBase) else None
        model = getattr(writer, "model", None)
        span_name = (
            f"response_llm_attempt_{_round + 1}"
            if prompt is not None
            else f"response_writer_attempt_{_round + 1}"
        )
        span_inputs: dict[str, Any] = {
            "provider": writer.name,
            "retry_errors": retry_errors,
        }
        if prompt is not None:
            span_inputs["model"] = model
            span_inputs["prompt_version"] = config.prompts.prompt_version
            span_inputs["messages"] = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        with trace_span(
            span_name,
            "LLM" if prompt is not None else "CHAIN",
            inputs=span_inputs,
            attributes={
                "smartspend.llm_role": "response",
                "llm.provider": writer.name,
                "llm.model": model or "deterministic",
                "prompt.version": config.prompts.prompt_version,
            },
        ) as span:
            try:
                draft = writer.write(draft_request)
            except ProviderError as err:
                # A generative provider failing mid-flight degrades to the
                # template — an answer with a true provenance beats an exception.
                draft = TemplateWriter().write(draft_request)
                draft = draft.model_copy(
                    update={"provider": f"{writer.name}->template ({err.__class__.__name__})"}
                )
            set_span_outputs(
                span,
                {
                    "provider": draft.provider,
                    "model": model,
                    "chars": len(draft.text),
                    "input_tokens": draft.input_tokens,
                    "output_tokens": draft.output_tokens,
                    "latency_s": round(draft.latency_s, 3),
                },
            )
        attempts.append(draft)
        with trace_span(
            f"validate_attempt_{_round + 1}",
            "CHAIN",
            inputs={"chars": len(draft.text)},
        ) as span:
            report = validate(
                draft.text, request.tool_runs, request.scenario, config.validator
            )
            set_span_outputs(
                span,
                {
                    "passed": report.passed,
                    "citation_count": len(report.citations),
                    "citations": [c.model_dump() for c in report.citations],
                    "checks": [c.model_dump() for c in report.checks],
                    "failures": [c.rule for c in report.failures],
                },
            )
        final = draft
        if report.passed:
            break
        retry_errors = [f"{c.rule}: {c.detail}" for c in report.failures]
        if len(attempts) > config.validator.max_rewrites:
            break

    assert final is not None and report is not None  # loop always runs once
    degraded = not report.passed
    if degraded:
        flags = "\n\n" + "\n".join(
            f"{UNVERIFIED_MARKER}: {c.rule} — {c.detail}]" for c in report.failures
        )
        final = final.model_copy(update={"text": final.text + flags})
    return final, report, attempts, degraded
