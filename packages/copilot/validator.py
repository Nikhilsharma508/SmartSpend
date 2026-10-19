"""The [4] VALIDATE stage — deterministic grounding checks, never an LLM.

Four rules (the numbering is the one ``docs/concepts.md`` §7 uses):

- **V1 number-trace.** Every dollar amount, percentage/pp, unit-tagged count
  and signed effect estimate in the prose must match a number in the flattened
  tool outputs (or the scenario, or a declared constant) within a relative
  tolerance. Each verified number becomes a :class:`Citation` with its path.
- **V2 units.** USD only — the reporting currency decided on Day 1. A brief
  containing ₹/INR/€/£/"lakh"/"crore" fails.
- **V3 causal language.** A deny-list blocks overclaims ("will buy",
  "guaranteed", "proves", "because they are ..."), and when a tool output
  carries an interval the prose must contain interval phrasing.
- **V4 sections.** Per intent, every header from
  :data:`packages.copilot.schemas.BRIEF_SECTIONS` must be present.

The validator is the safety lever, not the model's intelligence: it checks
facts, so it must be code. Tolerance and retry budget come from
``configs/copilot.yaml``.
"""

from __future__ import annotations

import re

from packages.copilot.schemas import (
    BRIEF_SECTIONS,
    CampaignScenario,
    Citation,
    Intent,
    ToolRun,
    ValidationCheck,
    ValidationReport,
    ValidatorConfig,
)

# --- extraction patterns ---------------------------------------------------
# Money first, then percentages, then unit-tagged counts, then bare signed
# decimals; each pass masks what it consumed so "$-155.64" is never also
# read as a signed decimal.

_MONEY_RE = re.compile(r"([-−+]?)\$\s?(\d[\d,]*(?:\.\d+)?)")
_DOLLARS_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:dollars?|USD)\b")
_PCT_RE = re.compile(r"([-−+]?\d[\d,]*(?:\.\d+)?)\s*(?:%|percentage[\s-]?points?|\bpp\b)")
_COUNT_RE = re.compile(
    r"(\d[\d,]*)\s*(contacts?|customers?|conversions?|people|emails?|days?|visits?)\b",
    re.IGNORECASE,
)
_SIGNED_DEC_RE = re.compile(r"(?<![\w.$%])([-−+])(\d[\d,]*\.\d+)(?![\w%])")

_FOREIGN_RE = re.compile(r"₹|\binr\b|\beur\b|€|\bgbp\b|£|\blakh\b|\bcrore\b", re.IGNORECASE)

_DENY_PHRASES: tuple[str, ...] = (
    "will buy",
    "will convert",
    "will visit",
    "will purchase",
    "will definitely",
    "this customer will",
    "guaranteed",
    "guarantee",
    "proves",
    "proven that",
    "we know that",
    "because they are",
    "causes them to",
    "makes them buy",
    "100% certain",
    "rest assured",
    "no risk",
)

_INTERVAL_MARKERS: tuple[str, ...] = ("[", "interval", "confidence", "range", "uncertain")

#: Absolute floors per citation kind — prose rounds a little; it may not move.
_FLOORS = {"usd": 0.01, "pct": 0.05, "count": 0.5, "num": 0.0005}

#: Typographic characters LLM prose inherits from training data: dashes that
#: stand in for a minus sign, and no-break spaces that group digits ("2 443").
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DIGIT_GROUP_SEP = re.compile(r"(?<=\d)[\u00a0\u202f\u2009](?=\d)")


def _normalize(text: str) -> str:
    """Canonicalize typographic forms before number extraction.

    A model writing "‑$155.64" (U+2011 hyphen) or "2\u202f443 contacts" (narrow
    no-break space) cites the same tool number as "-$155.64" / "2443" — only
    the typography differs, so V1 must not fail on it.
    """
    table = str.maketrans({c: "-" for c in _DASHES})
    return _DIGIT_GROUP_SEP.sub("", text.translate(table))


def _has_section(low: str, section: str) -> bool:
    """True when ``section`` appears as a markdown heading at a line start."""
    pattern = rf"(?:^|\n)\s*(?:#+\s*|\*\*)\s*{re.escape(section)}\b"
    return re.search(pattern, low) is not None


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def extract_cited(text: str) -> list[tuple[float, str]]:
    """(value, kind) for every citable number in the prose, in order.

    Kinds: ``usd`` (money), ``pct`` (percent/percentage points), ``count``
    (unit-tagged integers), ``num`` (bare signed decimals — effect estimates).
    Typographic minus signs and digit-grouping no-break spaces are normalized
    first (see :func:`_normalize`).
    """
    text = _normalize(text)
    cited: list[tuple[float, str]] = []
    spans: list[tuple[int, int]] = []

    for match in _MONEY_RE.finditer(text):
        value = float(match.group(2).replace(",", ""))
        if match.group(1) in ("-", "−"):
            value = -value
        cited.append((value, "usd"))
        spans.append(match.span())
    work = _mask(text, spans)

    for match in _DOLLARS_RE.finditer(work):
        cited.append((float(match.group(1).replace(",", "")), "usd"))
    for match in _PCT_RE.finditer(work):
        raw = match.group(1).replace(",", "")
        value = -float(raw.lstrip("-−+")) if raw[0] in "-−" else float(raw)
        cited.append((value, "pct"))
    for match in _COUNT_RE.finditer(work):
        cited.append((float(match.group(1).replace(",", "")), "count"))
    for match in _SIGNED_DEC_RE.finditer(work):
        value = float(match.group(2).replace(",", ""))
        cited.append((-value if match.group(1) in ("-", "−") else value, "num"))
    return cited


def _flatten(obj: object, prefix: str, out: list[tuple[float, str]]) -> None:
    """Every numeric leaf of a tool-output dump, with its dotted path."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _flatten(value, f"{prefix}[{i}]", out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append((float(obj), prefix))


def _matches(
    value: float, allowed: list[tuple[float, str]], *, kind: str, rel: float
) -> str | None:
    """The path of the first allowed number matching ``value``, else None."""
    floor = _FLOORS[kind]
    for candidate, path in allowed:
        targets = (candidate, candidate * 100.0) if kind == "pct" else (candidate,)
        for target in targets:
            if abs(value - target) <= max(rel * abs(target), floor):
                return path
    return None


def validate(
    prose: str,
    tool_runs: list[ToolRun],
    scenario: CampaignScenario,
    config: ValidatorConfig,
) -> ValidationReport:
    """Run V1–V4 over one draft; the report carries citations and failures."""
    allowed: list[tuple[float, str]] = []
    has_interval_source = False
    for run in tool_runs:
        if run.ok and run.output is not None:
            _flatten(run.output, run.name, allowed)
            if any("ci_" in path for _, path in allowed):
                has_interval_source = True
    for field in ("budget_usd", "cost_per_contact_usd", "depth", "mde", "daily_eligible"):
        value = getattr(scenario, field)
        if value is not None:
            allowed.append((float(value), f"scenario.{field}"))
    for constant in config.allowed_constants:
        allowed.append((float(constant), "config.allowed_constant"))

    # V1 — number trace.
    citations: list[Citation] = []
    unverified: list[str] = []
    for value, kind in extract_cited(prose):
        path = _matches(value, allowed, kind=kind, rel=config.rel_tolerance)
        if path is None:
            unverified.append(f"{value:g} ({kind})")
        elif not any(c.value == value and c.source == path for c in citations):
            citations.append(Citation(value=value, source=path))
    v1 = ValidationCheck(
        rule="V1_number_trace",
        passed=not unverified,
        detail=(
            f"{len(citations)} number(s) traced to tool outputs"
            if not unverified
            else f"no source for: {', '.join(unverified[:8])}"
        ),
    )

    # V2 — units: USD only.
    foreign = sorted({m.group(0) for m in _FOREIGN_RE.finditer(prose)})
    v2 = ValidationCheck(
        rule="V2_units",
        passed=not foreign,
        detail="USD only" if not foreign else f"non-USD token(s): {foreign}",
    )

    # V3 — causal language + interval phrasing when the tools carry intervals.
    low = prose.lower()
    denied = [p for p in _DENY_PHRASES if p in low]
    interval_ok = not has_interval_source or any(m in low for m in _INTERVAL_MARKERS)
    v3 = ValidationCheck(
        rule="V3_causal_language",
        passed=not denied and interval_ok,
        detail=(
            "no overclaims; interval phrasing present"
            if not denied and interval_ok
            else "; ".join(
                ([f"denied phrase(s): {denied}"] if denied else [])
                + ([] if interval_ok else ["tools carry an interval but the prose shows none"])
            )
        ),
    )

    # V4 — required sections for this intent. A section counts only as a
    # heading at line start ("## Assumptions" / "**Assumptions**"); the word
    # appearing somewhere in prose does not satisfy the contract.
    missing = [s for s in BRIEF_SECTIONS[Intent(scenario.intent)] if not _has_section(low, s)]
    v4 = ValidationCheck(
        rule="V4_sections",
        passed=not missing,
        detail="all sections present" if not missing else f"missing section(s): {missing}",
    )

    return ValidationReport(checks=[v1, v2, v3, v4], citations=citations)
