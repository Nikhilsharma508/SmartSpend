import { useMemo } from "react"
import { motion } from "motion/react"
import { BarChart } from "@/components/charts/bar-chart"
import { Bar } from "@/components/charts/bar"
import { BarXAxis } from "@/components/charts/bar-x-axis"
import { Grid } from "@/components/charts/grid"
import { ChartTooltip } from "@/components/charts/tooltip"
import type { Segment } from "@/api/types"

/**
 * Mean visit-rate uplift per policy-tree segment, from `get_segment_profile`.
 * Bars are the tool's point estimates (percentage points of visit rate);
 * the ± CI lives in the Segments page cards, where there is room for it.
 */
interface Props {
  segments: Segment[]
  className?: string
}

export function SegmentUpliftChart({ segments, className }: Props) {
  const data = useMemo(
    () =>
      [...segments]
        .sort((a, b) => b.uplift_visit - a.uplift_visit)
        .map((s) => ({
          name: `S${s.segment_id}`,
          uplift: +(s.uplift_visit * 100).toFixed(2),
        })),
    [segments],
  )

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      <BarChart data={data} xDataKey="name" className="h-[300px] w-full">
        <Grid horizontal />
        <Bar dataKey="uplift" fill="var(--chart-line-primary)" />
        <BarXAxis showAllLabels />
        <ChartTooltip />
      </BarChart>
    </motion.div>
  )
}
