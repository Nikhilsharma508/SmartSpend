// TS mirrors of the pydantic contracts the app consumes.
// Kept snake_case to match the API JSON byte-for-byte; apps/api serves them
// live (openapi-generated types can replace this file; drift = type error).

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
  label: string
  target: string
  trained_at: string
  train_rows: number
  data_sha256: string
  metrics: Record<string, number>
  assumptions: Assumptions
  tier: string
  score_kind: string
  supports_intervals: boolean
  research_only: boolean
  use_case: string
  fairness_note: string
  when_not_to_use: string[]
  limitations: string[]
}

/** One row of the model catalog (GET /api/v1/models). */
export interface ModelDescriptor {
  name: string
  label: string
  learner: string
  tier: string
  score_kind: string
  supports_intervals: boolean
  research_only: boolean
  use_case: string
  warning: string | null
  registered: boolean
  selectable: boolean
  added_to_allow_list_by_mode: boolean
  trained_at: string | null
  seed42_metrics: Record<string, number> | null
  meta_error: string | null
}

export interface ModelCatalog {
  default: string
  research_mode: boolean
  models: ModelDescriptor[]
  score_meanings: Record<string, string>
}

/** One row of the model comparison (GET /api/v1/models/compare). */
export interface ModelComparisonRow {
  name: string
  label: string
  learner: string
  tier: string
  score_kind: string
  supports_intervals: boolean
  research_only: boolean
  use_case: string
  warning: string | null
  qini_auc: number
  uplift_at_k: number
  incremental_conversions_at_k: number
  profit_at_k_usd: number
  profit_per_contact_usd: number
  n_test: number
  trained_at: string
  benchmark_warning: string | null
}

export interface ModelComparison {
  rows: ModelComparisonRow[]
  note: string
  score_meanings: Record<string, string>
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
  assumptions_used: string[]
  missing_fields: string[]
}

export type ModelSelectorStr = "gate" | "manual" | "fallback" | "default"

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
  model_name: string | null
  model_label: string | null
  model_learner: string | null
  model_tier: string | null
  score_kind: string | null
  model_selector: ModelSelectorStr | null
  model_confidence: number | null
  research_mode: boolean
  draft_attempts: number
  trace_id: string
  latency_s: number
  stage_latencies: Record<string, number>
}

export interface OverviewMeta {
  generated: string
  generator: string
}

export interface Overview {
  meta: OverviewMeta
  context: {
    n_test: number
    baseline_visit_rate: number
    feature_names: string[]
    model_name: string
    model_label: string
    learner: string | null
    score_kind: string | null
    tier: string
    supports_intervals: boolean
    research_only: boolean
    model_random_state: number | null
    data_sha256: string | null
  }
  assumptions: Assumptions
  profit_curve: ProfitCurve
  scenarios: ScenarioComparison
  segment_profile: SegmentProfile
  ab_designs: AbTestDesign[]
  model_card: ModelCard
}
