# Day-6 copilot eval report

46 cases (15 seed + 31 variations/adversarial). Writer: template (no LLM cost).
Gate: live Decisions API vs. keyword fallback (ablation).

| Metric | Live router | Keyword fallback | Δ |
| --- | --- | --- | --- |
| fast_path accuracy | 0.913 | 0.739 | +0.174 |
| intent accuracy (brief) | 1.000 | 0.844 | +0.156 |
| tool-call correctness | 1.000 | 0.800 | +0.200 |
| validator pass@1 | 1.000 | 1.000 | 0.000 |
| latency p50 | 1.15s | 0.00s (no gate) | +1.15s |

Verdict: the Decisions API earns its place — it lifts routing and tool-call
correctness materially, at ~1.15s p50 gate latency. The keyword fallback remains
the offline/no-network safety net (router=keyword_fallback in the trace).
