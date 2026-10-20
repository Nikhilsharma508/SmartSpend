import { useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  Bot,
  Calculator,
  CheckCircle2,
  ChevronDown,
  ListOrdered,
  Network,
  PenLine,
  Plug,
  Route,
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
import { askCopilot, type WriterProvider } from "@/api/client"
import type { CopilotPackage, FastPathStr, ValidationCheck } from "@/api/types"
import { seconds, usd } from "@/lib/format"
import { cn } from "@/lib/utils"

/* ------------------------------------------------------------------ */

type Msg =
  | { kind: "user"; text: string }
  | { kind: "assistant"; pkg: CopilotPackage }
  | { kind: "error"; text: string }

const FAST_PATH_STYLES: Record<FastPathStr, string> = {
  brief: "bg-primary/15 text-primary border-primary/40",
  clarify: "border-amber-400/60 bg-amber-400/10 text-amber-500",
  decline: "border-red-400/50 bg-red-500/10 text-red-400",
  redirect: "border-border bg-muted text-muted-foreground",
}

const PIPELINE_STAGES = ["gate", "tools", "write", "validate"] as const

const stageLabels: Record<string, string> = {
  gate: "routing the question",
  tools: "running decision tools",
  write: "narrating the brief",
  validate: "tracing every number",
}

const stageDetails: Record<string, string> = {
  gate: "intent + policy · missing · confidence",
  tools: "profit curve · budget · segments · power",
  write: "the writer sees tool outputs only",
  validate: "V1 numbers · V2 units · V3 causal · V4 sections",
}

function pendingSteps(stage: number): AgentStep[] {
  return PIPELINE_STAGES.map((name, i) => ({
    id: name,
    label: `${name[0]!.toUpperCase()}${name.slice(1)} — ${stageLabels[name]}`,
    status: i < stage ? "done" : i === stage ? "running" : "pending",
    detail: [stageDetails[name]],
  }))
}

/* ------------------------------------------------------------------ */

/** Markdown-lite renderer for the writer's prose (headings, bullets, bold). */
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

/* ------------------------------------------------------------------ */

function VerdictRow({ pkg }: { pkg: CopilotPackage }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge variant="secondary" className="text-[10px] capitalize">
        {pkg.intent.replace("_", " ")}
      </Badge>
      <Badge className={cn("border text-[10px] capitalize", FAST_PATH_STYLES[pkg.fast_path])}>
        {pkg.fast_path}
      </Badge>
      <Badge variant="outline" className="gap-1 font-mono text-[10px] normal-case text-muted-foreground">
        {seconds(pkg.latency_s)} · {pkg.router.replace("_", " ")}
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

function AssistantCard({ pkg }: { pkg: CopilotPackage }) {
  const s = pkg.scenario
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.995 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
    >
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
        <CardContent className="space-y-3 pt-5">
          <div className="flex items-start justify-between gap-3">
            <VerdictRow pkg={pkg} />
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

          {pkg.prose && <ProseLite text={pkg.prose} />}

          {pkg.fast_path === "clarify" && pkg.clarify_question && (
            <div className="rounded-lg border border-amber-400/40 bg-amber-400/10 p-3 text-[13px] text-amber-600 dark:text-amber-300">
              {pkg.clarify_question}
            </div>
          )}

          {pkg.validation && pkg.fast_path === "brief" && (
            <ValidationBlock checks={pkg.validation.checks} />
          )}

          {pkg.citations.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {pkg.citations.map((c, i) => (
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
            trace {pkg.trace_id} · {pkg.provider} · {pkg.draft_attempts} draft
            {pkg.draft_attempts > 1 ? "s" : ""}
            {pkg.degraded && (
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

/** The pipeline, spelled out — the LLM only routes and writes prose. */
const ARCHITECTURE = [
  {
    icon: Route,
    name: "Gate — route the question",
    engine: "LLM classifier · Decisions API",
    copy: "Classifies intent and returns three probabilities: policy risk, missing inputs and confidence. Thresholds live in configs/copilot.yaml. If the API is unreachable, a deterministic keyword router takes over and says so in the trace.",
  },
  {
    icon: ListOrdered,
    name: "Plan — extract the scenario",
    engine: "Python rules · no LLM",
    copy: "Parses budget, cost per contact, MDE and daily volume. Every default is recorded in assumptions_used. Invalid or negative money becomes one clarifying question instead of a guess.",
  },
  {
    icon: Calculator,
    name: "Tools — compute the numbers",
    engine: "packages.decision · deterministic",
    copy: "optimize_budget, compare_scenarios, get_segment_profile, design_ab_test and get_model_card run on the champion's holdout. The LLM never computes or rounds a number.",
  },
  {
    icon: PenLine,
    name: "Write — narrate the brief",
    engine: "LLM writer · template floor",
    copy: "The writer receives only the tool outputs and turns them into prose. Pick the provider below: template (no network), Ollama (local gpt-oss:120b-cloud), or the Azure writer. A failing LLM degrades to the template, never to invented numbers.",
  },
  {
    icon: ShieldCheck,
    name: "Validate — trace every number",
    engine: "Python validator · no LLM",
    copy: "V1 matches every printed number against a tool output (1% tolerance), V2 forbids non-USD units, V3 checks causal claims, V4 checks the required sections. One rewrite is allowed, then the answer is flagged UNVERIFIED.",
  },
]

function ArchitecturePanel() {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-2xl border border-border/70 bg-card/70 backdrop-blur-sm">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-4 py-3 text-left"
      >
        <Network className="size-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium">LLM architecture — how an answer is built</span>
        <span className="ml-auto hidden text-[11px] text-muted-foreground sm:inline">
          gate · plan · tools · write · validate
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-3 border-t border-border/50 px-4 py-3">
              {ARCHITECTURE.map((stage, i) => (
                <div key={stage.name} className="flex gap-3">
                  <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-muted/40">
                    <stage.icon className="size-3.5 text-muted-foreground" />
                  </div>
                  <div className="min-w-0 text-[12px]">
                    <div className="flex flex-wrap items-baseline gap-x-2">
                      <span className="font-medium">
                        {i + 1} · {stage.name}
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {stage.engine}
                      </span>
                    </div>
                    <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">
                      {stage.copy}
                    </p>
                  </div>
                </div>
              ))}
              <p className="rounded-lg border border-border/50 bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground">
                Division of labour: the LLM does two jobs — <strong>routing</strong> and{" "}
                <strong>prose</strong>. Money, segments and validation are code. That is why
                the validator can trace every number in the answer back to a tool run.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ------------------------------------------------------------------ */

const QUESTION_KINDS: {
  key: string
  label: string
  verdict: FastPathStr
  note: string
  examples: string[]
}[] = [
  {
    key: "plan",
    label: "Plan a campaign",
    verdict: "brief",
    note: "budget and/or cost per contact stated → depth + expected profit",
    examples: [
      "I have $500 to spend on contacts at $0.10 each — how deep should I target and what should I expect to earn?",
      "Plan a $2,000 email campaign for me — who should get it?",
    ],
  },
  {
    key: "what_if",
    label: "What-if",
    verdict: "brief",
    note: "re-prices the plan under new assumptions",
    examples: ["What if contact cost rises to $0.60?", "What if we doubled the budget to $1,000?"],
  },
  {
    key: "explain",
    label: "Explain the model",
    verdict: "brief",
    note: "who gets targeted and what each segment earns",
    examples: [
      "Who are the persuadables and who should we avoid targeting?",
      "Explain the segments the model found, and how targeting rates differ by zip code area.",
    ],
  },
  {
    key: "test",
    label: "Size a test",
    verdict: "brief",
    note: "MDE and daily volume in → customers per arm out",
    examples: [
      "We see 2,000 customers a day — how long should we test a new offer?",
      "How many customers do we need per arm to detect a 1 percentage point lift on visits?",
    ],
  },
  {
    key: "decline",
    label: "Policy decline",
    verdict: "decline",
    note: "targeting on protected or proxy attributes is refused",
    examples: [
      "Only target women aged 25-34 for this offer.",
      "Exclude everyone from low-income zip codes.",
    ],
  },
  {
    key: "clarify",
    label: "Clarify",
    verdict: "clarify",
    note: "nothing stated → one question back, no brief",
    examples: ["Should we run a campaign?", "Help me with marketing."],
  },
  {
    key: "redirect",
    label: "Off-scope",
    verdict: "redirect",
    note: "poems, weather — pointed back to the menu",
    examples: ["Write me a poem about spring.", "What's the weather tomorrow?"],
  },
]

const DOT_COLOR: Record<FastPathStr, string> = {
  brief: "bg-[var(--color-profit)]",
  clarify: "bg-amber-400",
  decline: "bg-red-400",
  redirect: "bg-zinc-400",
}

const WRITERS: { name: WriterProvider; description: string }[] = [
  { name: "ollama", description: "Local Ollama · gpt-oss:120b-cloud (the configured writer)" },
  { name: "template", description: "Deterministic house template — no network, always works" },
  { name: "neurometric", description: "Azure writer — needs its environment variables" },
]

export default function CopilotPage() {
  const [messages, setMessages] = useState<Msg[]>([])
  const [pending, setPending] = useState<{ question: string; stage: number } | null>(null)
  const [openKind, setOpenKind] = useState<string | null>("plan")
  const [provider, setProvider] = useState<WriterProvider>("ollama")
  const scrollRef = useRef<HTMLDivElement | null>(null)

  const lastPkg = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i]
      if (m?.kind === "assistant") return m.pkg
    }
    return null
  }, [messages])

  // keep the transcript pinned to the latest entry
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages.length, pending])

  // the pipeline stages are known even before the response lands
  useEffect(() => {
    if (!pending || pending.stage >= PIPELINE_STAGES.length - 1) return
    const id = setTimeout(
      () => setPending((p) => (p ? { ...p, stage: p.stage + 1 } : p)),
      pending.stage === 0 ? 700 : 1400,
    )
    return () => clearTimeout(id)
  }, [pending])

  const run = async (
    question: string,
    writer: string = provider,
    signal?: AbortSignal,
  ): Promise<void> => {
    const q = question.trim()
    if (!q || pending) return
    setMessages((m) => [...m, { kind: "user", text: q }])
    setPending({ question: q, stage: 0 })
    try {
      const pkg = await askCopilot(q, writer as WriterProvider, signal)
      setMessages((m) => [...m, { kind: "assistant", pkg }])
    } catch (err) {
      const e = err as Error
      setMessages((m) => [
        ...m,
        {
          kind: "error",
          text:
            e.name === "AbortError"
              ? "Request stopped."
              : `The copilot request failed: ${e.message}`,
        },
      ])
    } finally {
      setPending(null)
    }
  }

  const islandTasks: IslandTask[] = pending
    ? [
        {
          id: "copilot-run",
          label: pending.question,
          detail: stageLabels[PIPELINE_STAGES[Math.min(pending.stage, 3)]!],
          status: "running",
          progress: pending.stage / PIPELINE_STAGES.length,
        },
      ]
    : lastPkg
      ? [
          {
            id: "copilot-run",
            label: lastPkg.fast_path,
            detail: `${lastPkg.intent} · ${seconds(lastPkg.latency_s)}`,
            status: "done",
          },
        ]
      : []

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_19rem]">
      {/* transcript + prompt column */}
      <div className="flex min-h-[calc(100dvh-13rem)] flex-col gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Campaign copilot</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Gate → plan → write → validate, per request. Every answer is produced by the
            live pipeline and every number is traced back to a tool output by the
            validator.
          </p>
        </div>

        <ArchitecturePanel />

        <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto pr-1">
          {messages.length === 0 && !pending && (
            <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 px-6 py-10 text-center">
              <Bot className="size-5 text-muted-foreground" />
              <p className="text-sm font-medium">Ask about a budget, a test size, or the model.</p>
              <p className="max-w-md text-xs text-muted-foreground">
                The answer comes from the live pipeline; the panel above shows where the LLM
                sits and what code computes.
              </p>
            </div>
          )}

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
              {m.kind === "assistant" && <AssistantCard pkg={m.pkg} />}
              {m.kind === "error" && (
                <Card className="border-destructive/40 bg-destructive/5">
                  <CardContent className="pt-5 text-[13px] text-destructive">
                    {m.text}
                  </CardContent>
                </Card>
              )}
            </div>
          ))}

          {pending && (
            <div className="space-y-3">
              <AILoadingState steps={pendingSteps(pending.stage)} collapseWhenDone={false} />
              <AnimatePresence>
                {pending.stage >= 2 && (
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
                  disabled={pending !== null}
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
                    {kind.examples.map((q) => (
                      <button
                        key={q}
                        onClick={() => void run(q)}
                        disabled={pending !== null}
                        className="flex w-full items-center gap-2 rounded-lg border border-border/50 bg-background/40 px-2.5 py-1.5 text-left text-[12px] transition-colors hover:border-primary/50 disabled:opacity-50"
                      >
                        <span className={cn("size-1.5 shrink-0 rounded-full", DOT_COLOR[kind.verdict])} />
                        <span className="flex-1 text-foreground/90">{q}</span>
                        <Badge
                          variant="outline"
                          className="shrink-0 text-[10px] capitalize text-muted-foreground"
                        >
                          → {kind.verdict}
                        </Badge>
                      </button>
                    ))}
                  </motion.div>
                )
              })()}
          </AnimatePresence>
        </div>

        {/* prompt */}
        <div className="rounded-2xl border border-border/70 bg-card/70 p-2 backdrop-blur-sm">
          <p className="px-2 pb-1 text-[10px] text-muted-foreground">
            writer provider is sent with the request; a failing LLM falls back to the
            template, never to invented numbers
          </p>
          <AIPrompt
            models={WRITERS.map((w) => ({ name: w.name, description: w.description }))}
            defaultModel={provider}
            placeholder="Ask the copilot — a budget, a margin, a test size…"
            headerText=""
            onSubmit={(value, model, { signal }) => {
              setProvider(model as WriterProvider)
              return run(value, model, signal)
            }}
          />
        </div>
      </div>

      {/* live run card */}
      <div className="space-y-4">
        <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Plug className="size-4 text-muted-foreground" />
              <CardTitle className="text-sm">Last live run</CardTitle>
            </div>
            <CardDescription>
              {lastPkg
                ? `trace ${lastPkg.trace_id}`
                : "Ask a question — the gate, tool runs and validator verdict land here."}
            </CardDescription>
          </CardHeader>
          {lastPkg && (
            <CardContent className="space-y-2.5 text-[12px]">
              <Row k="intent → fast path" v={`${lastPkg.intent} → ${lastPkg.fast_path}`} />
              <Row k="router" v={lastPkg.router.replace("_", " ")} />
              <Row k="writer" v={lastPkg.provider} />
              <Row k="latency" v={seconds(lastPkg.latency_s)} />
              <Row k="drafts" v={String(lastPkg.draft_attempts)} />
              <Row k="tool runs" v={String(lastPkg.tool_runs.length)} />
              <Row
                k="validation"
                v={
                  lastPkg.validation
                    ? `${lastPkg.validation.checks.filter((c) => c.passed).length}/${lastPkg.validation.checks.length} passed`
                    : "no draft to grade"
                }
              />
              <Row k="citations" v={String(lastPkg.citations.length)} />
              {lastPkg.degraded && (
                <p className="pt-1 text-[11px] text-amber-500">
                  Degraded: at least one validator check failed and was flagged
                  UNVERIFIED.
                </p>
              )}
            </CardContent>
          )}
        </Card>

        <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Live, not a replay</CardTitle>
            <CardDescription>what changed when the API came online</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1.5 text-[11px] text-muted-foreground">
            <p>• Every question runs the real pipeline; nothing is matched to a recording.</p>
            <p>
              • The header badge polls <span className="font-mono">/health</span>; if the API
              is down the UI says so and offers no fallback numbers.
            </p>
            <p>
              • The planner, segments and test-design pages read{" "}
              <span className="font-mono">/api/v1/overview</span>, computed from the same
              champion.
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
