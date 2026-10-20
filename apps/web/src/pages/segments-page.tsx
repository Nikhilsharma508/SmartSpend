import { motion } from "motion/react"
import { AlertTriangle, Cpu, Scale } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { SegmentUpliftChart } from "@/components/charts/segment-uplift-chart"
import type { Segment } from "@/api/types"
import { SwitchVeil } from "@/components/switch-veil"
import { Term } from "@/components/term"
import { LoadError } from "@/components/load-error"
import { useOverview } from "@/hooks/use-overview"
import { usePreserveScrollDuringSwap } from "@/hooks/use-preserve-scroll"
import { intFmt, rateFmt } from "@/lib/format"
import { cn } from "@/lib/utils"

/** The model card's raw metric keys, readably. Unknown keys fall back raw. */
const METRIC_LABELS: Record<string, string> = {
  qini_auc: "ranking quality (Qini)",
  qini: "ranking quality (Qini)",
  auuc: "ranking quality (AUUC)",
  auc: "response ranking (AUC)",
  mae: "mean abs. error",
  rmse: "root mean sq. error",
  spearman: "rank correlation (Spearman)",
  calibration: "calibration error",
}

function UpliftBar({
  value,
  low,
  high,
  max,
}: {
  value: number
  low: number
  high: number
  max: number
}) {
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`
  return (
    <div className="relative h-2.5 w-full overflow-visible rounded-full bg-muted">
      <motion.div
        className="absolute inset-y-0 left-0 rounded-full bg-[var(--color-profit)]/80"
        initial={{ width: 0 }}
        animate={{ width: pct(value) }}
        transition={{ type: "spring", stiffness: 90, damping: 20, delay: 0.15 }}
      />
      {/* ± CI whisker anchored at the point estimate */}
      <motion.div
        className="absolute top-1/2 h-3.5 -translate-y-1/2 rounded-[2px] border-x-[2px] border-foreground/40 bg-foreground/5"
        initial={{ opacity: 0 }}
        animate={{ left: pct(low), width: `calc(${pct(high)} - ${pct(low)})` }}
        transition={{ duration: 0.5, delay: 0.35 }}
      />
    </div>
  )
}

function SegmentCard({
  seg,
  maxUplift,
  index,
}: {
  seg: Segment
  maxUplift: number
  index: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -3 }}
    >
      <Card className="h-full border-border/60 bg-card/70 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold">Segment {seg.segment_id}</CardTitle>
            <Badge
              variant="outline"
              className="text-[10px] uppercase tracking-wide text-muted-foreground"
            >
              {seg.policy_arm}
            </Badge>
          </div>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {seg.rules.map((r) => (
              <span
                key={r}
                className="rounded-md bg-muted/70 px-1.5 py-0.5 font-mono text-[10.5px] text-foreground/80"
              >
                {r}
              </span>
            ))}
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-xs">
          <div className="flex items-center justify-between text-muted-foreground">
            <span>
              {intFmt(seg.n_eval)} customers · {rateFmt(seg.share, 1)} of list
            </span>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-baseline justify-between">
              <span className="text-[11px] text-muted-foreground">
                <Term k="uplift">visit uplift</Term>
              </span>
              <span className="font-semibold tabular-nums">{rateFmt(seg.uplift_visit, 1)}</span>
            </div>
            <UpliftBar
              value={seg.uplift_visit}
              low={seg.uplift_visit_ci_low}
              high={seg.uplift_visit_ci_high}
              max={maxUplift}
            />
            <div className="flex justify-between tabular-nums text-[10.5px] text-muted-foreground">
              <span>
                <Term k="ci">95% band</Term> {rateFmt(seg.uplift_visit_ci_low, 1)}
              </span>
              <span>{rateFmt(seg.uplift_visit_ci_high, 1)}</span>
            </div>
          </div>
          {seg.uplift_conversion === null && (
            <p className="rounded-md bg-muted/50 px-2 py-1 text-[11px] text-muted-foreground">
              Conversion uplift can't be estimated here — this segment held too few
              conversions; read the visit lift only.
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default function SegmentsPage() {
  const { data: fx, isPending, isError, error, isPlaceholderData, switchingLabel } =
    useOverview()
  // a model swap must not move the reader
  usePreserveScrollDuringSwap(isPlaceholderData)

  if (isPending) {
    return (
      <div className="h-96 animate-pulse rounded-xl border border-border/60 bg-muted/40" />
    )
  }

  if (isError || !fx) return <LoadError error={error} />

  const profile = fx.segment_profile
  const maxUplift = Math.max(...profile.segments.map((s) => s.uplift_visit_ci_high))
  const evalTotal = profile.fairness.reduce((acc, r) => acc + r.n_eval, 0)

  return (
    <div className="relative space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Segments</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{profile.note}</p>
      </div>

      <SegmentUpliftChart segments={profile.segments} className="w-full" />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {profile.segments.map((seg, i) => (
          <SegmentCard key={seg.segment_id} seg={seg} maxUplift={maxUplift} index={i} />
        ))}
      </div>

      {/* fairness audit */}
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Scale className="size-4 text-muted-foreground" />
            <CardTitle className="text-base">Fairness audit</CardTitle>
          </div>
          <CardDescription>
            Who gets targeted vs who is on the list, per observable group — measured,
            not aspirational.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {profile.fairness.map((row) => {
            const share = row.n_eval / evalTotal
            const skew = row.targeting_rate - share
            return (
              <motion.div
                key={row.group}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="grid grid-cols-[7rem_1fr] gap-x-6 gap-y-2 rounded-lg border border-border/50 p-3 sm:grid-cols-[10rem_1fr]"
              >
                <div className="text-sm font-medium">{row.group}</div>
                <div className="space-y-2">
                  <PairBar
                    label="targeting rate"
                    value={row.targeting_rate}
                    accent="var(--color-profit)"
                  />
                  <PairBar label="population share" value={share} accent="var(--chart-2)" />
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
                    <span>
                      {intFmt(row.n_eval)} in eval · uplift {rateFmt(row.uplift_visit, 1)}{" "}
                      (CI {rateFmt(row.uplift_visit_ci_low, 1)} –{" "}
                      {rateFmt(row.uplift_visit_ci_high, 1)})
                    </span>
                    {Math.abs(skew) > 0.02 && (
                      <Badge
                        variant="outline"
                        className="text-[10px] text-amber-500"
                      >
                        {skew > 0 ? "+" : "−"}
                        {Math.abs(skew * 100).toFixed(1)}pp skew
                      </Badge>
                    )}
                  </div>
                  <p className="text-[11px] leading-relaxed text-muted-foreground/90">
                    {row.note}
                  </p>
                </div>
              </motion.div>
            )
          })}
        </CardContent>
      </Card>

      {/* model card */}
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
        <CardHeader>
          <div className="flex flex-wrap items-center gap-2">
            <Cpu className="size-4 text-muted-foreground" />
            <CardTitle className="text-base">Model card</CardTitle>
            <Badge variant="outline" className="text-[10px]">
              {fx.model_card.name}
            </Badge>
            <Badge
              variant="outline"
              className={cn(
                "text-[10px]",
                fx.model_card.tier === "champion"
                  ? "border-dashed"
                  : "border-amber-400/40 bg-amber-400/10 text-amber-500 dark:text-amber-300",
              )}
            >
              {fx.model_card.tier}
            </Badge>
            {fx.model_card.score_kind && (
              <Badge variant="secondary" className="text-[10px]">
                {fx.model_card.score_kind}
              </Badge>
            )}
          </div>
          <CardDescription>
            {fx.model_card.label || fx.model_card.learner} · {fx.model_card.learner} · target{" "}
            {fx.model_card.target} · trained{" "}
            {fx.model_card.trained_at.slice(0, 10)} on {intFmt(fx.model_card.train_rows)} rows
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2 text-xs">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              metrics
            </p>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(fx.model_card.metrics).map(([k, v]) => {
                const bare = k.replace(/_(test|train)$/g, "")
                return (
                  <span
                    key={k}
                    className="rounded-md bg-muted/70 px-2 py-1 font-mono text-[10.5px]"
                    title={k}
                  >
                    {METRIC_LABELS[bare] ?? bare}{" "}
                    <span className="font-semibold">{v.toFixed(3)}</span>
                  </span>
                )
              })}
            </div>
            <p className="pt-1 leading-relaxed text-muted-foreground">{fx.model_card.fairness_note}</p>
          </div>
          <div className="space-y-3 text-xs">
            <div>
              <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                <AlertTriangle className="size-3 text-amber-400" /> when not to use
              </p>
              <ul className="mt-1.5 list-disc space-y-1 pl-4 text-muted-foreground">
                {fx.model_card.when_not_to_use.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                limitations
              </p>
              <ul className="mt-1.5 list-disc space-y-1 pl-4 text-muted-foreground">
                {fx.model_card.limitations.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* keep the outgoing page mounted while the next model's numbers load */}
      <SwitchVeil show={isPlaceholderData} label={switchingLabel} />
    </div>
  )
}

function PairBar({
  label,
  value,
  accent,
}: {
  label: string
  value: number
  accent: string
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 text-[11px] text-muted-foreground">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
        <motion.div
          className="h-full rounded-full"
          style={{ background: accent, opacity: 0.85 }}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ type: "spring", stiffness: 80, damping: 20 }}
        />
      </div>
      <span className="w-12 shrink-0 text-right tabular-nums text-[11px]">
        {rateFmt(value, 1)}
      </span>
    </div>
  )
}
