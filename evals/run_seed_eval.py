"""Seed eval runner — the 15 Day-5 cases end-to-end, with a metrics report.

Not a test (it needs the live gate/writers); notebook 06 and Day 6 call it::

    uv run python evals/run_seed_eval.py                     # configured writer
    uv run python evals/run_seed_eval.py --provider template  # offline, no writer cost
    uv run python evals/run_seed_eval.py --json out.json

Reported: fast-path accuracy (all cases), intent accuracy (brief cases only —
for decline/redirect the routed intent is not the point), tool-call correctness
(brief cases only), validator pass@1 / pass@2 (brief cases only — fast paths
have no draft to grade), and p50/p95 latency. The gate is the live Decisions
API when the ``AZURE_*`` env is present, the keyword fallback otherwise —
recorded per case.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")  # CLI runs live whenever .env carries the AZURE_* names

from packages.copilot import run_request  # noqa: E402
from packages.copilot.schemas import load_config  # noqa: E402


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or ROOT / "evals" / "seed_cases.yaml"
    return yaml.safe_load(path.read_text())["cases"]


def run_seed_eval(
    provider: str | None = None,
    cases: list[dict[str, Any]] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Run every seed case through ``run_request`` and score against labels.

    ``context`` is injectable so tests can run the identical scoring on a
    synthetic holdout instead of the registered champion.
    """
    config = load_config()
    config.trace.enabled = False  # the eval prints its own report, not JSONL rows
    cases = cases if cases is not None else load_cases()

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        package = run_request(
            case["question"], config=config, writer_provider=provider, context=context
        )
        latency = time.perf_counter() - started

        expected_tools = case.get("tools") or []
        ran = {r.name for r in package.tool_runs if r.ok}
        is_brief = package.fast_path == "brief"
        row = {
            "id": case["id"],
            "fast_path": package.fast_path,
            "expected_fast_path": case["fast_path"],
            "fast_path_ok": package.fast_path == case["fast_path"],
            "intent": package.intent.value,
            "expected_intent": case["intent"],
            "is_brief": is_brief,
            "intent_ok": (
                package.intent.value == case["intent"] if is_brief else None
            ),
            "tools_ok": (
                set(expected_tools).issubset(ran) if expected_tools and is_brief else None
            ),
            "passed_first_draft": (
                bool(
                    package.validation
                    and package.validation.passed
                    and package.draft_attempts == 1
                )
                if is_brief
                else None
            ),
            "passed_final": (
                bool(package.validation and package.validation.passed) if is_brief else None
            ),
            "degraded": package.degraded,
            "attempts": package.draft_attempts,
            "latency_s": round(latency, 2),
            "router": package.router,
            "writer": package.provider,
        }
        rows.append(row)

    def _rate(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return round(sum(bool(v) for v in values) / len(values), 3) if values else None

    brief_rows = [r for r in rows if r["is_brief"]]
    n_briefs = len(brief_rows)
    latencies = sorted(r["latency_s"] for r in rows)
    summary = {
        "n_cases": len(rows),
        "n_briefs": n_briefs,
        "fast_path_accuracy": _rate("fast_path_ok"),
        "intent_accuracy_brief_cases": _rate("intent_ok"),
        "tool_call_correctness": _rate("tools_ok"),
        # pass@1: the configured writer's FIRST draft passed without degradation,
        # pass@2: the final draft (after one rewrite) passed — both graded on the
        # cases that actually produced a brief; fast paths have no draft to grade.
        "validator_pass_at_1": (
            round(
                sum(bool(r["passed_first_draft"]) and not r["degraded"] for r in brief_rows)
                / n_briefs,
                3,
            )
            if n_briefs
            else None
        ),
        "validator_pass_at_2": _rate("passed_final"),
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": latencies[max(int(0.95 * len(latencies)) - 1, 0)],
        "writer_provider": provider or load_config().writer.provider,
    }
    return {"summary": summary, "cases": rows}


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None, help="template | ollama | neurometric")
    parser.add_argument("--json", default=None, help="write the full report to this path")
    args = parser.parse_args()

    report = run_seed_eval(provider=args.provider)
    for row in report["cases"]:
        flags = []
        if not row["fast_path_ok"]:
            flags.append(f"EXPECTED {row['expected_fast_path']}")
        if row["intent_ok"] is False:
            flags.append(f"intent {row['expected_intent']}")
        if row["tools_ok"] is False:
            flags.append("missing tools")
        if row["degraded"]:
            flags.append("degraded")
        print(
            f"{row['id']:22s} {row['fast_path']:9s} {row['intent']:11s} "
            f"{row['latency_s']:6.2f}s {'; '.join(flags) or 'ok'}"
        )
    print("\nsummary:", json.dumps(report["summary"], indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
