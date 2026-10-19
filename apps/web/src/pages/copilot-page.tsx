import { useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  Bot,
  CheckCircle2,
  ListOrdered,
  Plug,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react"
import AIPrompt from "@/components/kokonutui/ai-prompt"
import AILoadingState, { type AgentStep } from "@/components/kokonutui/ai-loading"
import AITextLoading from "@/components/kokonutui/ai-text-loading"
import { StatusIsland, type IslandTask } from "@/components/kokonutui/status-island"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type {
  CopilotDemo,
  FastPathStr,
  ValidationCheck,
} from "@/api/types"
import { MEASURED, fixtures, findDemo } from "@/api/mock/fixtures"
import { seconds, usd } from "@/lib/format"
import { cn } from "@/lib/utils"

/* ------------------------------------------------------------------ */

type Msg =
  | { kind: "user"; text: string }
  | { kind: "assistant"; demo: CopilotDemo }
  | { kind: "notfound"; question: string }

const FAST_PATH_STYLES: Record<FastPathStr, string> = {
  brief: "bg-primary/15 text-primary border-primary/40",
  clarify: "border-amber-400/60 bg-amber-400/10 text-amber-500",
  decline: "border-red-400/50 bg-red-500/10 text-red-400",
  redirect: "border-border bg-muted text-muted-foreground",
}

const stageOrder = (d: CopilotDemo): string[] => {
  const stages = ["gate"]
  if (d.tool_runs.length > 0) stages.push("tools")
  if (d.fast_path === "brief") stages.push("write", "validate")
  return stages
}

const stageLabels: Record<string, string> = {
  gate: "routing the question",
  tools: "running decision tools",
  write: "narrating the brief",
  validate: "tracing every number",
}

function buildSteps(d: CopilotDemo, stage: number): AgentStep[] {
  const order = stageOrder(d)
  const done = (name: string): AgentStep => {
    switch (name) {
      case "gate":
        return {
          id: "gate",
          label: "Gate — routing the question",
          status: "done",
          detail: [
            `intent: ${d.intent} · fast path: ${d.fast_path}`,
            "router: keyword fallback (offline playback)",
          ],
        }
      case "tools":
        return {
          id: "tools",
          label: "Plan — decision tools",
          status: "done",
          detail: d.tool_runs.map((t) => `${t.name} — ok`),
        }
      case "write":
        return {
          id: "write",
          label: "Write — template floor",
          status: "done",
          detail: [`provider: ${d.provider}`, `drafts: ${d.draft_attempts}`],
        }
      case "validate":
        return {
          id: "validate",
          label: "Validate — number trace",
          status: "done",
          detail: d.validation
            ? [
                `${d.validation.checks.filter((c) => c.passed).length}/${d.validation.checks.length} checks passed — V1 numbers, V2 units, V3 causal rubric, V4 sections`,
              ]
            : [],
        }
      default:
        return { id: name, label: name, status: "done" }
    }
  }
  const running = (name: string): AgentStep => ({ ...done(name), status: "running" })
  return order.map((name, i) => {
    if (i < stage) return done(name)
    return name === order[stage] ? running(name) : { ...done(name), status: "pending" }
  })
}

/** Markdown-lite renderer for the template writer's prose. */
function ProseLite({ text }: { text: string }) {
  const nodes: React.ReactNode[] = []
  let bullets: string[] = []
  const flush = () => {
    if (bullets.length) {
      nodes.push(
        <ul key={`ul-${nodes.length}`} className="mb-2 ml-4 list-disc space-y-1">
          {bullets.map((b, i) => (
            <li key={i}>{inline(b)}</li>
          ))}
        </ul>,
      )
      bullets = []
    }
  }
  for (const line of text.split("\n")) {
    const t = line.trim()
    if (!t) {
      flush()
      continue
    }
    if (t.startsWith("- ")) {
      bullets.push(t.slice(2))
      continue
    }
    if (/^#{2,3}\s/.test(t)) {
      flush()
      nodes.push(
        <h3
          key={`h-${nodes.length}`}
          className="mb-1.5 mt-4 flex items-center gap-1.5 text-[13px] font-semibold first:mt-0"
        >
          <Sparkles className="size-3 text-[var(--color-profit)]" />
          {t.replace(/^#{2,3}\s/, "")}
        </h3>,
      )
      continue
    }
    flush()
    nodes.push(<p key={`p-${nodes.length}`}>{inline(t)}</p>)
  }
  flush()
  return <div className="space-y-1 text-[13px] leading-relaxed text-foreground/90">{nodes}</div>
}

function inline(text: string): React.ReactNode {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-foreground">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    ),
  )
}

function VerdictRow({ d }: { d: CopilotDemo }) {
  const mismatch =
    d.expected_fast_path !== null &&
    (d.expected_fast_path !== d.fast_path || d.expected_intent !== d.intent)
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge variant="secondary" className="text-[10px] capitalize">
        {d.intent.replace("_", " ")}
      </Badge>
      <Badge className={cn("border text-[10px] capitalize", FAST_PATH_STYLES[d.fast_path])}>
        {d.fast_path}
      </Badge>
      {mismatch && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant="outline" className="gap-1 border-amber-400/60 text-[10px] text-amber-500">
              <ListOrdered className="size-3" />
              offline router miss
            </Badge>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-80">
            Live labels expect {d.expected_intent}/{d.expected_fast_path}. The offline
            keyword router is the documented fallback floor; the live Decisions API
            answered these correctly ({MEASURED.seed_fast_path_accuracy}).
          </TooltipContent>
        </Tooltip>
      )}
      <Badge variant="outline" className="gap-1 font-mono text-[10px] normal-case text-muted-foreground">
        {seconds(d.latency_s)} · {d.router.replace("_", " ")}
      </Badge>
    </div>
  )
}

function ValidationBlock({ checks }: { checks: ValidationCheck[] }) {
  const pass = checks.filter((c) => c.passed).length
  return (
    <div className="space-y-1.5 rounded-lg border border-border/50 bg-muted/30 p-3">
      <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        <ShieldCheck className="size-3.5" />
        validation — {pass}/{checks.length} passed
      </div>
      {checks.map((c) => (
        <div key={c.rule} className="flex items-start gap-2 text-[12px]">
          {c.passed ? (
            <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-[var(--color-profit)]" />
          ) : (
            <XCircle className="mt-0.5 size-3.5 shrink-0 text-red-400" />
          )}
          <div>
            <span className="font-medium">{c.rule}</span>{" "}
            <span className="text-muted-foreground">— {c.detail}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function AssistantCard({ demo }: { demo: CopilotDemo }) {
  const s = demo.scenario
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.995 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
        <CardContent className="space-y-3 pt-5">
          <div className="flex items-start justify-between gap-3">
            <VerdictRow d={demo} />
            <Bot className="size-4 shrink-0 text-muted-foreground" />
          </div>

          {s && (
            <div className="flex flex-wrap gap-1.5">
              {s.budget_usd !== null && (
                <Badge variant="outline" className="font-mono text-[10px]">
                  budget {usd(s.budget_usd)}
                </Badge>
              )}
              {s.cost_per_contact_usd !== null && (
                <Badge variant="outline" className="font-mono text-[10px]">
                  cost/contact ${s.cost_per_contact_usd}
                </Badge>
              )}
              {s.mde !== null && (
                <Badge variant="outline" className="font-mono text-[10px]">
                  MDE {(s.mde * 100).toFixed(1)}pp
                </Badge>
              )}
              {s.daily_eligible !== null && (
                <Badge variant="outline" className="font-mono text-[10px]">
                  {s.daily_eligible.toLocaleString("en-US")}/day eligible
                </Badge>
              )}
            </div>
          )}

          {demo.prose && <ProseLite text={demo.prose} />}

          {demo.fast_path === "clarify" && demo.clarify_question && (
            <div className="rounded-lg border border-amber-400/40 bg-amber-400/10 p-3 text-[13px] text-amber-600 dark:text-amber-300">
              {demo.clarify_question}
            </div>
          )}

          {demo.validation && demo.fast_path === "brief" && (
            <ValidationBlock checks={demo.validation.checks} />
          )}

          {demo.citations.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {demo.citations.map((c, i) => (
                <Tooltip key={i}>
                  <TooltipTrigger asChild>
                    <Badge
                      variant="outline"
                      className="max-w-72 cursor-default truncate font-mono text-[10px] font-normal text-muted-foreground"
                    >
                      {c.value}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="top">{c.source}</TooltipContent>
                </Tooltip>
              ))}
            </div>
          )}

          <div className="flex items-center gap-2 border-t border-border/40 pt-2 font-mono text-[10px] text-muted-foreground">
            trace {demo.trace_id} · {demo.provider} · {demo.draft_attempts} draft
            {demo.draft_attempts > 1 ? "s" : ""}
            {demo.degraded && (
              <Badge className="border border-red-400/50 bg-red-500/10 text-[9px] text-red-400">
                degraded
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

/* ------------------------------------------------------------------ */

/**
 * Question taxonomy — every verdict shape the copilot can produce, with the
 * verbatim recorded examples that exercise it (ids index copilot_demos so the
 * text is the real seed question, never paraphrased).
 */
const QUESTION_KINDS: {
  key: string
  label: string
  verdict: FastPathStr
  note: string
  demoIds: string[]
}[] = [
  {
    key: "plan",
    label: "Plan a campaign",
    verdict: "brief",
    note: "budget and/or cost per contact stated → depth + expected profit",
    demoIds: ["plan_budget_500", "plan_budget_2000", "plan_no_budget"],
  },
  {
    key: "what_if",
    label: "What-if",
    verdict: "brief",
    note: "re-prices the plan under new assumptions",
    demoIds: ["whatif_cost_rise", "whatif_budget_double"],
  },
  {
    key: "explain",
    label: "Explain the model",
    verdict: "brief",
    note: "who gets targeted and what each segment earns",
    demoIds: ["explain_persuadables", "explain_fairness"],
  },
  {
    key: "test",
    label: "Size a test",
    verdict: "brief",
    note: "MDE and daily volume in → customers per arm out",
    demoIds: ["test_duration", "test_mde"],
  },
  {
    key: "decline",
    label: "Policy decline",
    verdict: "decline",
    note: "targeting on protected or proxy attributes is refused",
    demoIds: ["policy_gender", "policy_proxy_zip"],
  },
  {
    key: "clarify",
    label: "Clarify",
    verdict: "clarify",
    note: "nothing stated → one question back, no brief (live-gate behavior)",
    demoIds: ["campaign_clarify", "ambiguous_campaign"],
  },
  {
    key: "redirect",
    label: "Off-scope",
    verdict: "redirect",
    note: "poems, weather — pointed back to the menu",
    demoIds: ["oos_poem", "oos_weather"],
  },
]

const DOT_COLOR: Record<FastPathStr, string> = {
  brief: "bg-[var(--color-profit)]",
  clarify: "bg-amber-400",
  decline: "bg-red-400",
  redirect: "bg-zinc-400",
}

export default function CopilotPage() {
  const [messages, setMessages] = useState<Msg[]>([
    { kind: "assistant", demo: fixtures.copilot_demos[0]! },
  ])
  const [pendingDemo, setPendingDemo] = useState<CopilotDemo | null>(null)
  const [stage, setStage] = useState(0)
  const [openKind, setOpenKind] = useState<string | null>("plan")
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const resolvePending = useRef<(() => void) | null>(null)
  const cancelled = useRef(false)

  const run = (question: string): Promise<void> =>
    new Promise((resolve) => {
      if (pendingDemo) return resolve()
      const demo = findDemo(question)
      setMessages((m) => [...m, { kind: "user", text: question }])
      if (!demo) {
        setMessages((m) => [...m, { kind: "notfound", question }])
        return resolve()
      }
      cancelled.current = false
      resolvePending.current = resolve
      setPendingDemo(demo)
      setStage(0)
    })

  // advance the staged playback
  useEffect(() => {
    if (!pendingDemo) return
    const order = stageOrder(pendingDemo)
    if (stage >= order.length) {
      const id = setTimeout(() => {
        if (!cancelled.current) {
          setMessages((m) => [...m, { kind: "assistant", demo: pendingDemo }])
        }
        setPendingDemo(null)
        resolvePending.current?.()
        resolvePending.current = null
      }, 300)
      return () => clearTimeout(id)
    }
    const latencies = pendingDemo.stage_latencies
    const measured = (latencies[order[stage]!] ?? 0) * 1000
    const duration = Math.max(450, Math.min(measured, 2500))
    const id = setTimeout(() => setStage((s) => s + 1), duration)
    return () => clearTimeout(id)
  }, [pendingDemo, stage])

  // keep the transcript pinned to the latest entry
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages.length, stage, pendingDemo])

  const islandTasks: IslandTask[] = useMemo(() => {
    if (pendingDemo) {
      const order = stageOrder(pendingDemo)
      return [
        {
          id: "copilot-run",
          label: pendingDemo.id,
          detail: stageLabels[order[Math.min(stage, order.length - 1)]!],
          status: "running",
          progress: stage / order.length,
        },
      ]
    }
    const last = [...messages].reverse().find((m): m is Extract<Msg, { kind: "assistant" }> => m.kind === "assistant")
    if (!last || last.demo.id === fixtures.copilot_demos[0]!.id) return []
    return [
      {
        id: "copilot-run",
        label: last.demo.id,
        detail: `${last.demo.fast_path} · ${seconds(last.demo.latency_s)}`,
        status: "done",
        action: { label: "trace", onClick: () => undefined },
      },
    ]
  }, [pendingDemo, stage, messages])

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_19rem]">
      {/* transcript + prompt column */}
      <div className="flex min-h-[calc(100dvh-13rem)] flex-col gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Campaign copilot</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Gate → plan → write → validate, per request. Playback replays the 15
            recorded seed-eval runs; every number the copilot prints was traced to a
            tool output by the validator.
          </p>
        </div>

        <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto pr-1">
          {messages.map((m, i) => (
            <div key={i}>
              {m.kind === "user" && (
                <div className="flex justify-end">
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-[13px] text-primary-foreground"
                  >
                    {m.text}
                  </motion.div>
                </div>
              )}
              {m.kind === "assistant" && <AssistantCard demo={m.demo} />}
              {m.kind === "notfound" && (
                <Card className="border-dashed border-border/70 bg-muted/20">
                  <CardContent className="pt-5 text-[13px] text-muted-foreground">
                    Not in the offline corpus. The FastAPI service (the next Day-6
                    step) answers live questions; for now try one of the recorded
                    runs — the chips below cover every verdict shape.
                  </CardContent>
                </Card>
              )}
            </div>
          ))}

          {pendingDemo && (
            <div className="space-y-3">
              <AILoadingState steps={buildSteps(pendingDemo, stage)} collapseWhenDone={false} />
              <AnimatePresence>
                {stageOrder(pendingDemo)[stage] === "write" && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <AITextLoading
                      texts={[
                        "narrating champion profit…",
                        "inserting verified numbers…",
                        "checking causal claims…",
                      ]}
                      className="ml-1 text-[12px] text-muted-foreground"
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>

        {/* question taxonomy — what can I ask */}
        <div className="rounded-2xl border border-border/70 bg-card/70 p-3 backdrop-blur-sm">
          <p className="mb-2 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            <ListOrdered className="size-3" />
            what can I ask — every verdict shape
          </p>
          <div className="flex flex-wrap gap-1.5">
            {QUESTION_KINDS.map((k) => {
              const active = openKind === k.key
              return (
                <button
                  key={k.key}
                  onClick={() => setOpenKind(active ? null : k.key)}
                  disabled={pendingDemo !== null}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors disabled:opacity-50",
                    active
                      ? "border-primary/60 bg-primary/10 text-primary"
                      : "border-border/70 bg-muted/30 text-muted-foreground hover:text-foreground",
                  )}
                >
                  <span className={cn("size-1.5 rounded-full", DOT_COLOR[k.verdict])} />
                  {k.label}
                </button>
              )
            })}
          </div>
          <AnimatePresence initial={false}>
            {openKind &&
              (() => {
                const kind = QUESTION_KINDS.find((k) => k.key === openKind)!
                return (
                  <motion.div
                    key={kind.key}
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
                    className="mt-2.5 space-y-1"
                  >
                    <p className="text-[11px] text-muted-foreground">{kind.note}</p>
                    {kind.demoIds.map((id) => {
                      const demo = fixtures.copilot_demos.find((d) => d.id === id)
                      if (!demo) return null
                      return (
                        <button
                          key={id}
                          onClick={() => run(demo.question)}
                          disabled={pendingDemo !== null}
                          className={cn(
                            "flex w-full items-center gap-2 rounded-lg border px-2.5 py-1.5 text-left text-[12px] transition-colors hover:border-primary/50 disabled:opacity-50",
                            demo.fast_path !== demo.expected_fast_path
                              ? "border-amber-400/40 bg-amber-400/5"
                              : "border-border/50 bg-background/40",
                          )}
                        >
                          <span className={cn("size-1.5 shrink-0 rounded-full", DOT_COLOR[demo.fast_path])} />
                          <span className="flex-1 truncate text-foreground/90">{demo.question}</span>
                          <Badge variant="outline" className="shrink-0 text-[10px] capitalize text-muted-foreground">
                            → {demo.fast_path}
                          </Badge>
                        </button>
                      )
                    })}
                  </motion.div>
                )
              })()}
          </AnimatePresence>
        </div>

        {/* prompt */}
        <div className="rounded-2xl border border-border/70 bg-card/70 p-2 backdrop-blur-sm">
          <p className="px-2 pb-1 text-[10px] text-muted-foreground">
            playback uses the recorded template writer; writer choice matters live
          </p>
          <AIPrompt
            models={MEASURED.writers.map((w) => ({ name: w.name, description: w.description }))}
            defaultModel={MEASURED.writers[1]?.name}
            placeholder="Ask the copilot — a budget, a margin, a test size…"
            headerText=""
            onSubmit={(value) => run(value)}
          />
        </div>
      </div>

      {/* system card sidebar */}
      <div className="space-y-4">
        <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Plug className="size-4 text-muted-foreground" />
              <CardTitle className="text-sm">System card — live service</CardTitle>
            </div>
            <CardDescription>measured with the Decisions API on, Day 5</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2.5 text-[12px]">
            <Row k="gate (Decisions API)" v={`${MEASURED.gate_latency_s} s`} />
            <Row k="fast path band" v={MEASURED.fast_path_band_s} />
            <Row k="seed fast-path accuracy" v={MEASURED.seed_fast_path_accuracy} />
            <Row k="brief latency p50 / p95" v={`${MEASURED.p50_s} / ${MEASURED.p95_s} s`} />
            <Row k="context boot" v={MEASURED.context_boot_s + " s"} />
            <Row k="first explain" v={MEASURED.first_explain_s} />
            <div className="space-y-1.5 pt-2">
              {MEASURED.writers.map((w) => (
                <div key={w.name} className="rounded-md bg-muted/40 px-2.5 py-2">
                  <div className="font-mono text-[11px] font-medium">{w.name}</div>
                  <div className="mt-0.5 text-[11px] text-muted-foreground">{w.description}</div>
                </div>
              ))}
            </div>
            <p className="pt-1 font-mono text-[10px] text-muted-foreground">{MEASURED.source}</p>
          </CardContent>
        </Card>

        <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Recorded corpus</CardTitle>
            <CardDescription>15 seed-eval runs, offline router + template</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-1.5">
            {fixtures.copilot_demos.map((d) => (
              <button
                key={d.id}
                onClick={() => run(d.question)}
                disabled={pendingDemo !== null}
                className={cn(
                  "rounded-md border px-2 py-1 font-mono text-[10px] transition-colors hover:border-primary/50 disabled:opacity-50",
                  d.fast_path !== d.expected_fast_path
                    ? "border-amber-400/40 text-amber-600 dark:text-amber-300"
                    : "border-border/70 text-muted-foreground",
                )}
              >
                {d.id}
              </button>
            ))}
            <p className="pt-1 text-[10px] text-muted-foreground">
              amber = offline router missed the live label (documented keyword-floor
              gaps: clarifications and one proxy-zip case).
            </p>
          </CardContent>
        </Card>
      </div>

      <StatusIsland
        tasks={islandTasks}
        className="fixed bottom-6 right-6 z-50"
        dismissAfter={1200}
      />
    </div>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[11px] text-muted-foreground">{k}</span>
      <span className="text-right font-medium tabular-nums">{v}</span>
    </div>
  )
}
