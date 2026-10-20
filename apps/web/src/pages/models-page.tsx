import { motion } from "motion/react"
import { Check, FlaskConical, Info, Lock, Scale, Unlock } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { LoadError } from "@/components/load-error"
import { Term } from "@/components/term"
import { useModelComparison } from "@/hooks/use-model-comparison"
import { useModels } from "@/hooks/use-models"
import { applyModelChoice, toggleResearchMode } from "@/lib/model-actions"
import { intFmt, usd } from "@/lib/format"
import { cn } from "@/lib/utils"
import { useModelSettings } from "@/state/model-settings"

function TierBadge({ tier }: { tier: string }) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "text-[10px]",
        tier === "champion" && "border-dashed",
        tier === "challenger" && "border-sky-400/40 bg-sky-400/10 text-sky-500 dark:text-sky-300",
        tier === "research" && "border-amber-400/40 bg-amber-400/10 text-amber-500 dark:text-amber-300",
      )}
    >
      {tier}
    </Badge>
  )
}

export default function ModelsPage() {
  const { data, isPending, isError, error } = useModels()
  const comparison = useModelComparison()
  const { model, researchMode, setResearchMode, pickModel } = useModelSettings()

  if (isPending) {
    return (
      <div className="h-96 animate-pulse rounded-xl border border-border/60 bg-muted/40" />
    )
  }
  if (isError || !data) return <LoadError error={error} />

  const meanings = data.score_meanings

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
            <FlaskConical className="size-5 text-muted-foreground" />
            Model lab
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            {data.models.length} ranking models, each trained once on the same seed-42
            holdout. Pick one and the planner, segments and test-design all start
            answering with it — or flip it from anywhere with the header pill. The
            copilot picks its own model per question (its gate suggests,{" "}
            <Term k="registeredModel">registered</Term> code enforces the rules).
            Single-seed numbers audit the artifact on file; the honest envelope is the
            10-seed benchmark in{" "}
            <span className="font-mono text-[12px]">docs/concepts.md</span>.
          </p>
        </div>
        <Button
          variant={researchMode ? "default" : "outline"}
          size="sm"
          className={cn("gap-2", researchMode && "bg-amber-500/90 text-black hover:bg-amber-500")}
          aria-pressed={researchMode}
          onClick={() =>
            toggleResearchMode(!researchMode, {
              models: data.models,
              model,
              pickModel,
              setResearchMode,
            })
          }
        >
          {researchMode ? <Unlock className="size-3.5" /> : <Lock className="size-3.5" />}
          Research Mode {researchMode ? "on" : "off"}
        </Button>
      </div>

      {/* selected-model strip */}
      <motion.div layout className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {data.models.map((m) => {
          const active = m.name === model
          const locked = !m.selectable
          return (
            <Card
              key={m.name}
              className={cn(
                "border-border/60 bg-card/70 backdrop-blur-sm transition-colors",
                active && "border-primary/60 ring-1 ring-primary/40",
                !m.registered && "opacity-60",
              )}
            >
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium">
                    {m.label}
                    {active && (
                      <Badge className="gap-1 bg-primary/15 text-[10px] text-primary">
                        <Check className="size-3" /> in use
                      </Badge>
                    )}
                  </CardTitle>
                  <TierBadge tier={m.tier} />
                </div>
                <CardDescription className="text-[11px]">
                  {m.name} · {m.learner} · {m.registered ? `trained ${m.trained_at?.slice(0, 10)}` : "not registered"}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2.5 text-xs">
                <p className="leading-relaxed text-muted-foreground">
                  {meanings[m.score_kind] ?? m.score_kind}
                </p>
                <p className="text-[11px] leading-relaxed text-muted-foreground/80">{m.use_case}</p>
                {m.warning && (
                  <p className="rounded-md bg-amber-400/10 px-2 py-1.5 text-[11px] leading-relaxed text-amber-600 dark:text-amber-300">
                    {m.warning}
                  </p>
                )}
                {m.seed42_metrics && (
                  <div className="flex flex-wrap gap-1.5 font-mono text-[10.5px] text-muted-foreground">
                    <span className="rounded-md bg-muted/70 px-1.5 py-0.5">
                      qini {m.seed42_metrics.qini_auc?.toFixed(4)}
                    </span>
                    <span className="rounded-md bg-muted/70 px-1.5 py-0.5">
                      profit@k {usd(m.seed42_metrics.profit_at_k_usd ?? 0)}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between gap-2 pt-1">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    {m.registered === false ? "unavailable" : locked ? "research-only" : "available"}
                  </span>
                  <Button
                    size="sm"
                    variant={active ? "secondary" : "default"}
                    disabled={!m.registered}
                    onClick={() => {
                      if (locked) {
                        if (!researchMode) {
                          toast.error("Research-only model", {
                            description:
                              "Turn Research Mode on (the lab switch above or the header pill) to select S/T/X or the causal forest.",
                          })
                          return
                        }
                      }
                      applyModelChoice(m, { pickModel })
                    }}
                  >
                    {active ? "Selected" : "Use in planner"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </motion.div>

      <p className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Info className="size-3" />
        Note: Research Mode is a deliberate unlock, recorded on every copilot request
        and MLflow trace — it never changes the numbers the tools compute, only who is
        allowed to ask.
      </p>

      {/* comparison table */}
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Scale className="size-4 text-muted-foreground" />
            <CardTitle className="text-base">Same split, one table</CardTitle>
          </div>
          <CardDescription>{comparison.data?.note ?? "Scoring…"}</CardDescription>
        </CardHeader>
        <CardContent>
          {comparison.isPending && (
            <div className="h-40 animate-pulse rounded-lg bg-muted/40" />
          )}
          {comparison.data && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Score meaning</TableHead>
                  <TableHead className="text-right">Qini AUC</TableHead>
                  <TableHead className="text-right">extra conv @ k</TableHead>
                  <TableHead className="text-right">Profit @ k</TableHead>
                  <TableHead>10-seed reading</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {comparison.data.rows.map((row) => (
                  <TableRow key={row.name} className={cn(row.name === model && "bg-primary/5")}>
                    <TableCell>
                      <span className="flex flex-wrap items-center gap-2 font-medium">
                        {row.label}
                        <TierBadge tier={row.tier} />
                      </span>
                      <span className="text-[11px] text-muted-foreground">{row.name}</span>
                    </TableCell>
                    <TableCell className="max-w-[16rem] whitespace-normal text-[11px] leading-relaxed text-muted-foreground">
                      {(comparison.data.score_meanings[row.score_kind] ?? row.score_kind)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {row.qini_auc.toFixed(4)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {row.incremental_conversions_at_k.toFixed(2)}
                    </TableCell>
                    <TableCell
                      className={cn(
                        "text-right font-semibold tabular-nums",
                        row.profit_at_k_usd >= 0
                          ? "text-[var(--color-profit)]"
                          : "text-[var(--color-loss)]",
                      )}
                    >
                      {usd(row.profit_at_k_usd)}
                    </TableCell>
                    <TableCell className="max-w-[18rem] whitespace-normal text-[11px] leading-relaxed text-muted-foreground">
                      {row.benchmark_warning}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
          <p className="mt-3 text-[11px] text-muted-foreground">
            Profit/Δconv at k = 30% of the seed-42 test holdout
            {comparison.data?.rows[0] && ` (n_test ${intFmt(comparison.data.rows[0].n_test)})`}.
            T-learner's negative profit is the honest one: its top-k buys contacts the
            uplift math cannot pay for.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
