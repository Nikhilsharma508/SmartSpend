import { useState } from "react"
import { motion } from "motion/react"
import { FlaskConical } from "lucide-react"
import { KpiCard } from "@/components/kpi-card"
import { Badge } from "@/components/ui/badge"
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
import { useFixtures } from "@/hooks/use-fixtures"
import { intFmt, rateFmt } from "@/lib/format"
import { cn } from "@/lib/utils"

const MDE_LABELS: Record<string, string> = {
  "0.005": "±0.5pp (50 bp)",
  "0.01": "±1.0pp (100 bp)",
  "0.02": "±2.0pp (200 bp)",
  "0.03": "±3.0pp (300 bp)",
}

export default function TestDesignPage() {
  const { data: fx, isPending } = useFixtures()
  const [chosenMde, setChosenMde] = useState(0.01)

  if (isPending || !fx) {
    return <div className="h-96 animate-pulse rounded-xl border border-border/60 bg-muted/40" />
  }

  const designs = fx.ab_designs
  const design = designs.find((d) => Math.abs(d.mde_absolute - chosenMde) < 1e-9) ?? designs[1]!

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Test design</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{design.note}</p>
      </div>

      {/* MDE picker */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 flex items-center gap-1.5 text-xs text-muted-foreground">
          <FlaskConical className="size-3.5" /> minimum detectable effect (
          {rateFmt(design.alpha, 0)} alpha · {Math.round(design.power * 100)}% power)
        </span>
        <div className="flex rounded-full border border-border/60 bg-muted/40 p-1">
          {designs.map((d) => {
            const key = String(d.mde_absolute)
            const active = Math.abs(d.mde_absolute - chosenMde) < 1e-9
            return (
              <button
                key={key}
                onClick={() => setChosenMde(d.mde_absolute)}
                className={cn(
                  "relative rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors",
                  active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {active && (
                  <motion.span
                    layoutId="mde-pill"
                    className="absolute inset-0 rounded-full bg-primary"
                    transition={{ type: "spring", stiffness: 420, damping: 32 }}
                  />
                )}
                <span className="relative z-10">{MDE_LABELS[key]}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* KPIs */}
      <div className="grid gap-4 sm:grid-cols-3">
        <KpiCard
          label="customers / arm"
          value={design.n_per_arm}
          format={(n) => intFmt(n)}
          sub={`both arms · total ${intFmt(design.n_total)}`}
          accentClass="bg-primary"
          hint="Two-proportion test sizing at the chosen MDE against the baseline visit rate."
        />
        <KpiCard
          label="duration"
          value={design.duration_days}
          format={(n) => `${Math.ceil(n)} days`}
          sub={`at ${intFmt(design.daily_eligible)} eligible/day`}
          accentClass="bg-amber-400"
          hint="Randomization unit is the customer; allocation 50/50."
        />
        <KpiCard
          label="baseline visit rate"
          value={design.baseline_rate}
          format={(n) => rateFmt(n, 2)}
          sub={`MDE ${(design.mde_relative * 100).toFixed(0)}% relative`}
          accentClass="bg-[var(--color-profit)]"
          hint={`Champion-model customers (seed-42 holdout), alpha ${design.alpha}, power ${design.power}.`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* all four designs */}
        <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-base">The grid the tool produces</CardTitle>
            <CardDescription>
              design_ab_test rows at alpha 0.05, power 0.8 — pick per business stakes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>MDE</TableHead>
                  <TableHead className="text-right">per arm</TableHead>
                  <TableHead className="text-right">total</TableHead>
                  <TableHead className="text-right">days</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {designs.map((d) => {
                  const active = Math.abs(d.mde_absolute - chosenMde) < 1e-9
                  return (
                    <TableRow
                      key={d.mde_absolute}
                      className={cn(active && "bg-primary/5")}
                      onClick={() => setChosenMde(d.mde_absolute)}
                    >
                      <TableCell className="font-medium">
                        <span className="flex items-center gap-2">
                          {MDE_LABELS[String(d.mde_absolute)]}
                          <Badge variant="outline" className="text-[10px]">
                            {(d.mde_relative * 100).toFixed(0)}%
                          </Badge>
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{intFmt(d.n_per_arm)}</TableCell>
                      <TableCell className="text-right tabular-nums">{intFmt(d.n_total)}</TableCell>
                      <TableCell className="text-right tabular-nums">{Math.ceil(d.duration_days)}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* reading guide */}
        <Card className="border-border/60 bg-card/70 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-base">How to read this</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-xs leading-relaxed text-muted-foreground">
            <p>
              To detect a <span className="font-medium text-foreground">tiny</span> lift
              (0.5pp against a {rateFmt(designs[0]!.baseline_rate, 1)} baseline) the test
              needs {intFmt(designs[0]!.n_total)} customers — weeks at current volume. A{" "}
              <span className="font-medium text-foreground">big</span> lift (3pp) lands in a
              few days, but you learn almost nothing if the lift is real-but-small.
            </p>
            <p>
              Day 3's evaluation used n≈{intFmt(fx.context.n_test)} per effect plane —
              that is why the visit-uplift intervals are tight (±~1pp) while the
              conversion plane is unusable. This page shows the same arithmetic applied{" "}
              <span className="italic">before</span> a new campaign, not after the fact.
            </p>
            <p className="rounded-lg border border-border/50 bg-muted/40 p-3">
              {design.note}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
