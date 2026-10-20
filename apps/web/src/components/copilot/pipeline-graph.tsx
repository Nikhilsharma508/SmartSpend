import { motion } from "motion/react"
import {
  Calculator,
  ListOrdered,
  PenLine,
  Route,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react"
import type { CopilotPackage } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { seconds } from "@/lib/format"
import { cn } from "@/lib/utils"

/**
 * The copilot pipeline as a graph (the rail counterpart to the collapsible
 * text): one node per stage, plain-language caption plus the technical engine
 * as provenance. While a question is running, the active node pulses; the
 * per-stage timings come from the last run's real `stage_latencies` — the API
 * times gate / plan together and writes write+validate as one key, which is
 * shown as such. Nothing here is invented; an unfilled piece stays empty.
 */

/** One node in the copilot pipeline graph. */
interface GraphStage {
  id: string
  name: string
  icon: LucideIcon
  /** One plain sentence — what this stage does for the user. */
  plain: string
  /** Technical provenance chip — what actually executes. */
  engine: string
}

const PIPELINE_STAGES: GraphStage[] = [
  {
    id: "gate",
    name: "Gate",
    icon: Route,
    plain: "sorts out your question — what you are asking, any policy risks, which model should answer",
    engine: "LLM classifier · Decisions API",
  },
  {
    id: "plan",
    name: "Plan",
    icon: ListOrdered,
    plain: "collects what you stated (budget, cost, test size) and flags what was assumed",
    engine: "Python rules · no LLM",
  },
  {
    id: "tools",
    name: "Tools",
    icon: Calculator,
    plain: "compute every number from the model's holdout — budget, scenarios, segments, test size",
    engine: "decision tools · deterministic",
  },
  {
    id: "write",
    name: "Write",
    icon: PenLine,
    plain: "turns tool outputs into prose — the writer narrates, it never does arithmetic",
    engine: "LLM writer · template floor",
  },
  {
    id: "validate",
    name: "Validate",
    icon: ShieldCheck,
    plain: "traces every printed number back to a tool output before the brief reaches you",
    engine: "Python validator · no LLM",
  },
]

export interface LastRunSummary {
  verdict: CopilotPackage["fast_path"]
  modelLabel: string | null
  latencyS: number
  toolRuns: number
  stageLatencies: Record<string, number>
}

/** Rail-facing graph: nodes, timings, and the active-stage pulse. */
export function PipelineGraph({
  activeStage,
  lastRun,
}: {
  /** 0–4 while a question runs; null when idle. */
  activeStage: number | null
  lastRun: LastRunSummary | null
}) {
  const latencies = lastRun?.stageLatencies ?? {}
  return (
    <div className="relative px-4 pb-4 pt-2">
      {/* the rail line the stage chips sit on */}
      <span
        aria-hidden
        className="absolute bottom-14 left-[27px] top-4 w-px bg-gradient-to-b from-border/70 via-border/50 to-border/70"
      />
      <ul className="space-y-1">
        {PIPELINE_STAGES.map((stage, i) => {
          const active = activeStage === i
          const done = activeStage !== null && i < activeStage
          return (
            <li key={stage.id} className="relative flex gap-2.5 py-1.5 pl-0">
              <motion.span
                aria-hidden
                className={cn(
                  "relative z-10 mx-0 flex size-9 shrink-0 items-center justify-center rounded-lg border bg-card",
                  active && "border-primary/70 ring-2 ring-primary/25",
                  done && !active && "border-emerald-400/50 bg-emerald-400/10",
                )}
                animate={active ? { scale: [1, 1.08, 1] } : { scale: 1 }}
                transition={
                  active
                    ? { repeat: Infinity, duration: 1.4, ease: "easeInOut" }
                    : { type: "spring", stiffness: 300, damping: 22 }
                }
              >
                <stage.icon
                  className={cn(
                    "size-4",
                    active
                      ? "text-primary"
                      : done
                        ? "text-emerald-500 dark:text-emerald-400"
                        : "text-muted-foreground",
                  )}
                />
              </motion.span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-x-2">
                  <span className={cn("text-[12px] font-semibold", active && "text-primary")}>
                    {i + 1} · {stage.name}
                  </span>
                  <span className="truncate font-mono text-[9.5px] text-muted-foreground/80">
                    {stage.engine}
                  </span>
                </div>
                <p className="text-[10.5px] leading-snug text-muted-foreground">{stage.plain}</p>
                <span className="mt-0.5 flex flex-wrap gap-1">
                  {stage.id === "tools" && lastRun && (
                    <StageChip>
                      {lastRun.toolRuns} tool run{lastRun.toolRuns === 1 ? "" : "s"}
                    </StageChip>
                  )}
                  {i < 2 && latencies[stage.id] !== undefined && (
                    <StageChip>{seconds(latencies[stage.id]!)}</StageChip>
                  )}
                  {stage.id === "validate" && latencies["write_validate"] !== undefined && (
                    <StageChip>write + validate {seconds(latencies["write_validate"]!)}</StageChip>
                  )}
                </span>
              </div>
            </li>
          )
        })}
      </ul>

      {/* last run, glanceably — the trace id stays on the per-answer card */}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-border/50 bg-muted/30 px-3 py-2 text-[10.5px] text-muted-foreground">
        {lastRun ? (
          <>
            <Badge variant="outline" className="text-[9px] normal-case">
              last run · {lastRun.verdict}
            </Badge>
            {lastRun.modelLabel && <span className="truncate">model: {lastRun.modelLabel}</span>}
            <span className="tabular-nums">{seconds(lastRun.latencyS)} total</span>
          </>
        ) : (
          <span>no run yet — ask a question and the stages fill in with real timings.</span>
        )}
      </div>
    </div>
  )
}

function StageChip({ children }: { children: import("react").ReactNode }) {
  return (
    <span className="rounded border border-border/50 bg-muted/60 px-1 py-px font-mono text-[9px] text-muted-foreground">
      {children}
    </span>
  )
}
