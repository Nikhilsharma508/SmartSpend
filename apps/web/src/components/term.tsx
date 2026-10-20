import type { ReactNode } from "react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

/**
 * Plain-English renderings for technical nouns the UI cannot avoid. Used by
 * <Term>: keep them short, honest and causal-care compliant — describe what a
 * number *is*, never promise what it will do.
 */
export const PLAIN_TERMS = {
  uplift:
    "Uplift = the extra response caused by the email (treated customers minus similar untreated ones). It is not the same thing as the response rate.",
  depth:
    "Depth = the share of the contact list you target, starting from the customers the model ranks most persuadable.",
  mde: "MDE = minimum detectable effect — the smallest true lift a test could notice. Wanting to notice smaller lifts means running a bigger test.",
  holdout:
    "Holdout = the part of the data the model never trained on (`n_test` rows here). Numbers come from those untouched customers, which is what makes them honest.",
  seed42:
    "seed 42 = the random-split number used everywhere in this app. Same seed → same split → the same page reproduces the same numbers to the cent.",
  qini:
    "Qini AUC = how much better this model ranks people for contact than contacting them in random order. 0 means no better than random.",
  ci: "95% CI (confidence interval) = the band where the true value plausibly sits. A wide band is an uncertain estimate, not a wrong one.",
  registeredModel:
    "Registered model = a trained artifact saved to disk with its metadata (who reads it, what a score means). The app serves it as-is; it never retrains.",
  researchMode:
    "Research Mode = the deliberate unlock of benchmark-verified weak models (S/T/X learners, causal forest). Recorded on every request; never for a live campaign.",
  scoreKind:
    "Score kind = what the model's ranking actually measures — a response probability ('who responds if emailed') or an incremental effect ('who changes because of the email'). Mixing them up is the classic uplift mistake.",
} as const

export type TermKey = keyof typeof PLAIN_TERMS

/** Dotted-underlined jargon with the plain-English rendering on hover/focus. */
export function Term({
  k,
  children,
  className,
}: {
  k: TermKey
  children?: ReactNode
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className={cn(
            "cursor-help border-b border-dotted border-muted-foreground/50 outline-none focus-visible:border-primary",
            className,
          )}
        >
          {children ?? k}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-80">
        {PLAIN_TERMS[k]}
      </TooltipContent>
    </Tooltip>
  )
}

/**
 * The decision-tool names (they appear in traces and the API): honest
 * provenance in a muted mono chip, with the plain-English job as the tooltip.
 */
const TOOL_DOCS = {
  optimize_budget:
    "Walks the contact list from most to least persuadable and prices profit at every cut. Its peak is the recommended depth.",
  compare_scenarios:
    "Prices the standard strategies (contact everyone, contact nobody, contact the model's top picks) at one depth, in dollars, so they can be compared.",
  get_segment_profile:
    "Describes the groups the model found — the rule that defines each one, its size, and how much visit lift it carries.",
  design_ab_test:
    "Sizes an A/B test: customers per arm and run days, given the smallest lift you care about and the baseline rate.",
  get_model_card:
    "The model's identity card — what it predicts, when it was trained, and when it should not be used.",
} as const

export type ToolName = keyof typeof TOOL_DOCS

export function ToolChip({ name, className }: { name: ToolName; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <code
          tabIndex={0}
          className={cn(
            "cursor-help rounded border border-border/60 bg-muted/60 px-1 py-px font-mono text-[10px] text-muted-foreground outline-none focus-visible:border-primary",
            className,
          )}
        >
          {name}
        </code>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-80">
        {TOOL_DOCS[name]}
      </TooltipContent>
    </Tooltip>
  )
}
