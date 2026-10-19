// TS mirrors of the pydantic contracts the app consumes.
// Kept snake_case to match the API JSON byte-for-byte; when apps/api lands,
// openapi-generated types will replace this file (schema drift = type error).

export interface Assumptions {
  margin_rate: number
  cost_per_contact_usd: number
  value_per_conversion_usd: number
}

export interface ProfitPoint {
  depth: number
  n_targeted: number
  incremental_conversions: number
  incremental_conversions_se: number
  profit_usd: number
  profit_ci_low_usd: number
  profit_ci_high_usd: number
}

export interface ProfitCurve {
  points: ProfitPoint[]
  break_even_depth: number | null
  recommended_depth: number | null
  recommended_profit_usd: number | null
  assumptions: Assumptions
  note: string
}

export interface Scenario {
  strategy: string
  n_targeted: number
  incremental_conversions: number
  profit_usd: number
  profit_ci_low_usd: number
  profit_ci_high_usd: number
  feasible: boolean
}

export interface ScenarioComparison {
  depth: number
  budget_usd: number
  scenarios: Scenario[]
  champion_wins: boolean
  note: string
}

export interface Segment {
  segment_id: number
  rules: string[]
  n_eval: number
  share: number
  policy_arm: string
  uplift_visit: number
  uplift_visit_ci_low: number
  uplift_visit_ci_high: number
  uplift_conversion: number | null
}

export interface FairnessRow {
  group: string
  n_eval: number
  targeting_rate: number
  uplift_visit: number
  uplift_visit_ci_low: number
  uplift_visit_ci_high: number
  note: string
}

export interface SegmentProfile {
  segments: Segment[]
  fairness: FairnessRow[]
  note: string
}

export interface AbTestDesign {
  baseline_rate: number
  mde_absolute: number
  mde_relative: number
  alpha: number
  power: number
  n_per_arm: number
  n_total: number
  daily_eligible: number
  duration_days: number
  note: string
}

export interface ModelCard {
  name: string
  learner: string
  target: string
  trained_at: string
  train_rows: number
  data_sha256: string
  metrics: Record<string, number>
  assumptions: Assumptions
  fairness_note: string
  when_not_to_use: string[]
  limitations: string[]
}

export type IntentStr =
  | "plan"
  | "what_if"
  | "explain"
  | "test_design"
  | "out_of_scope"

export type FastPathStr = "brief" | "clarify" | "redirect" | "decline"

export interface ToolRun {
  name: string
  ok: boolean
  output: unknown
  error: string | null
}

export interface Citation {
  value: string
  source: string
}

export interface ValidationCheck {
  rule: string
  passed: boolean
  detail: string
}

export interface ValidationReport {
  checks: ValidationCheck[]
  citations: Citation[]
}

export interface ScenarioFields {
  intent: IntentStr
  question: string
  budget_usd: number | null
  cost_per_contact_usd: number | null
  depth: number | null
  mde: number | null
  daily_eligible: number | null
  assumptions_used: boolean
  missing_fields: string[]
}

export interface CopilotPackage {
  question: string
  intent: IntentStr
  fast_path: FastPathStr
  prose: string
  clarify_question: string | null
  decline_reason: string | null
  scenario: ScenarioFields | null
  tool_runs: ToolRun[]
  citations: Citation[]
  validation: ValidationReport | null
  degraded: boolean
  provider: string
  router: "decisions" | "keyword_fallback"
  draft_attempts: number
  trace_id: string
  latency_s: number
  stage_latencies: Record<string, number>
}

export interface CopilotDemo extends CopilotPackage {
  id: string
  expected_intent: string | null
  expected_fast_path: string | null
  note: string | null
}

export interface FixturesMeta {
  generated: string
  generator: string
  mock: boolean
}

export interface Fixtures {
  meta: FixturesMeta
  context: {
    n_test: number
    baseline_visit_rate: number
    feature_names: string[]
    champion_random_state: number | null
    data_sha256: string | null
  }
  assumptions: Assumptions
  profit_curve: ProfitCurve
  scenarios: ScenarioComparison
  segment_profile: SegmentProfile
  ab_designs: AbTestDesign[]
  model_card: ModelCard
  copilot_demos: CopilotDemo[]
}
