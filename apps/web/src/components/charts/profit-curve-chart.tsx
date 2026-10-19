import { useMemo, useRef, useState } from "react"
import { motion } from "motion/react"
import type { ProfitCurve, ProfitPoint } from "@/api/types"
import { interpPoint, intFmt, rateFmt, usd, usdSigned } from "@/lib/format"

/**
 * The planner's hero chart: the tool's profit curve over targeting depth.
 *
 * Numbers come straight from `optimize_budget` (the fixture dump); the only
 * client-side geometry is scale math. The chosen depth is draggable —
 * pointer drag, then a spring snap to the nearest grid point on release.
 */
const W = 960
const H = 340
const PAD = { top: 28, right: 34, bottom: 46, left: 72 }
const PLOT_W = W - PAD.left - PAD.right
const PLOT_H = H - PAD.top - PAD.bottom
const DEPTH_MAX = 0.5
const ENTER_EASE = [0.22, 1, 0.36, 1] as const

interface Props {
  curve: ProfitCurve
  chosenDepth: number
  onChooseDepth: (depth: number) => void
}

export function ProfitCurveChart({ curve, chosenDepth, onChooseDepth }: Props) {
  const points = curve.points
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [dragging, setDragging] = useState(false)
  const [hover, setHover] = useState<number | null>(null)

  const { xOf, yOf, zeroY, yMin, yMax } = useMemo(() => {
    const lows = points.map((p) => p.profit_ci_low_usd)
    const highs = points.map((p) => p.profit_ci_high_usd)
    const yMin = Math.min(0, ...lows)
    const yMax = Math.max(...highs) * 1.06
    const xOf = (d: number) => PAD.left + (d / DEPTH_MAX) * PLOT_W
    const yOf = (p: number) => PAD.top + PLOT_H * (1 - (p - yMin) / (yMax - yMin))
    return { xOf, yOf, zeroY: yOf(0), yMin, yMax }
  }, [points])

  const { linePath, bandPath, areaPath, ciPaths, breakEvenPath } = useMemo(() => {
    const fwd = points
      .map((p, i) => `${i === 0 ? "M" : "L"}${xOf(p.depth).toFixed(1)} ${yOf(p.profit_usd).toFixed(1)}`)
      .join(" ")
    const upper = points
      .map((p, i) => `${i === 0 ? "M" : "L"}${xOf(p.depth).toFixed(1)} ${yOf(p.profit_ci_high_usd).toFixed(1)}`)
      .join(" ")
    const lowerRev = [...points]
      .reverse()
      .map((p) => `L${xOf(p.depth).toFixed(1)} ${yOf(p.profit_ci_low_usd).toFixed(1)}`)
      .join(" ")
    const band = `${upper} ${lowerRev} Z`
    const first = points[0]!
    const last = points[points.length - 1]!
    const area = `${fwd} L${xOf(last.depth).toFixed(1)} ${zeroY.toFixed(1)} L${xOf(first.depth).toFixed(1)} ${zeroY.toFixed(1)} Z`
    const ci = points.map((p) => {
      const x = xOf(p.depth)
      return `M${x.toFixed(1)} ${yOf(p.profit_ci_high_usd).toFixed(1)} V${yOf(p.profit_ci_low_usd).toFixed(1)}`
    })
    const breakEven =
      curve.break_even_depth === null
        ? ""
        : `M${xOf(curve.break_even_depth).toFixed(1)} ${(PAD.top - 6).toFixed(1)} V${(PAD.top + PLOT_H).toFixed(1)}`
    return { linePath: fwd, bandPath: band, areaPath: area, ciPaths: ci, breakEvenPath: breakEven }
  }, [points, xOf, yOf, zeroY, curve.break_even_depth])

  const yTicks = useMemo(() => {
    const out: { v: number; y: number }[] = []
    const n = 5
    for (let i = 0; i <= n; i++) {
      const v = yMin + ((yMax - yMin) * i) / n
      out.push({ v, y: yOf(v) })
    }
    return out
  }, [yMin, yMax, yOf])

  const xTicks = useMemo(() => Array.from({ length: 11 }, (_, i) => i * 0.05), [])

  const depthFromClientX = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return chosenDepth
    const px = ((clientX - rect.left) / rect.width) * W
    return Math.max(0, Math.min(DEPTH_MAX, ((px - PAD.left) / PLOT_W) * DEPTH_MAX))
  }

  const nearest = (depth: number): ProfitPoint => {
    let best = points[0]!
    for (const p of points) {
      if (Math.abs(p.depth - depth) < Math.abs(best.depth - depth)) best = p
    }
    return best
  }

  const chosen = interpPoint(points, chosenDepth)
  const rec = curve.recommended_depth
  const hoverPoint = hover === null ? null : (points[hover] ?? null)
  const readout = hoverPoint ?? chosen.point

  function readoutRow(label: string, value: string) {
    return (
      <div key={label} className="flex items-baseline justify-between gap-6">
        <span className="text-[11px] text-muted-foreground">{label}</span>
        <span className="text-[12px] font-medium tabular-nums">{value}</span>
      </div>
    )
  }

  return (
    <div className="relative">
      {/* hover/drag readout */}
      <div className="pointer-events-none absolute top-3 right-3 z-10 w-60 space-y-1 rounded-lg border border-border/70 bg-popover/90 p-3 shadow-sm backdrop-blur">
        <div className="mb-2 flex items-center gap-2 text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
          depth {rateFmt(readout.depth, 0)}
          {hoverPoint === null && !chosen.exact && (
            <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] tracking-normal normal-case text-muted-foreground">
              interpolated
            </span>
          )}
        </div>
        {readoutRow("targeted", `${intFmt(readout.n_targeted)} contacts`)}
        {readoutRow("Δ conversions", readout.incremental_conversions.toFixed(2))}
        {readoutRow("profit", usdSigned(readout.profit_usd))}
        {readoutRow("95% CI", `${usd(readout.profit_ci_low_usd)} – ${usd(readout.profit_ci_high_usd)}`)}
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className={dragging ? "w-full cursor-grabbing select-none" : "w-full cursor-grab select-none"}
        onPointerMove={(e) => {
          const d = depthFromClientX(e.clientX)
          let i = points.findIndex((p) => p.depth >= d)
          if (i < 0) i = points.length - 1
          setHover(i)
        }}
        onPointerLeave={() => {
          setHover(null)
          if (dragging) {
            setDragging(false)
            onChooseDepth(nearest(chosenDepth).depth)
          }
        }}
        style={{ touchAction: "none" }}
      >
        <defs>
          <linearGradient id="profitBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-profit)" stopOpacity="0.9" />
            <stop offset="100%" stopColor="var(--color-profit)" stopOpacity="0.5" />
          </linearGradient>
          <clipPath id="clipPositive">
            <rect
              x={PAD.left}
              y={PAD.top - 14}
              width={PLOT_W}
              height={Math.max(0, zeroY - PAD.top + 15)}
            />
          </clipPath>
          <clipPath id="clipNegative">
            <rect
              x={PAD.left}
              y={zeroY}
              width={PLOT_W}
              height={Math.max(0, PAD.top + PLOT_H - zeroY + 8)}
            />
          </clipPath>
        </defs>

        {/* y grid */}
        {yTicks.map((t, i) => (
          <g key={`yt-${i}`}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={t.y}
              y2={t.y}
              stroke="var(--chart-grid)"
              strokeDasharray={Math.abs(t.v) < 1e-9 ? undefined : "3 6"}
              opacity={Math.abs(t.v) < 1e-9 ? 0 : 1}
            />
            <text
              x={PAD.left - 10}
              y={t.y + 4}
              textAnchor="end"
              className="fill-muted-foreground text-[11px] tabular-nums"
            >
              {usd(t.v)}
            </text>
          </g>
        ))}

        {/* x labels */}
        {xTicks.map((d) => (
          <text
            key={`xt-${d}`}
            x={xOf(d)}
            y={PAD.top + PLOT_H + 20}
            textAnchor="middle"
            className="fill-muted-foreground text-[11px] tabular-nums"
          >
            {(d * 100).toFixed(0)}%
          </text>
        ))}
        <text
          x={W - PAD.right}
          y={H - 6}
          textAnchor="end"
          className="fill-muted-foreground text-[11px]"
        >
          targeting depth (top-k ranked by predicted uplift) →
        </text>

        {/* CI band: green above zero, red below, split by clip */}
        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.15, ease: ENTER_EASE }}
        >
          <path d={bandPath} fill="url(#profitBand)" clipPath="url(#clipPositive)" opacity={0.22} />
          <path d={bandPath} fill="var(--color-loss)" clipPath="url(#clipNegative)" opacity={0.26} />
          {/* per-point CI whiskers */}
          {ciPaths.map((d, i) => (
            <path key={i} d={d} stroke="var(--chart-foreground)" strokeOpacity="0.18" />
          ))}
        </motion.g>

        {/* positive-side fill under the curve */}
        <motion.path
          d={areaPath}
          fill="url(#profitBand)"
          clipPath="url(#clipPositive)"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.14 }}
          transition={{ duration: 0.9, delay: 0.25, ease: ENTER_EASE }}
        />

        {/* the profit curve itself, drawn on; clipped per polarity */}
        <motion.path
          d={linePath}
          fill="none"
          clipPath="url(#clipPositive)"
          stroke="var(--color-profit)"
          strokeWidth={2.4}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.15, ease: ENTER_EASE, delay: 0.1 }}
        />
        <motion.path
          d={linePath}
          fill="none"
          clipPath="url(#clipNegative)"
          stroke="var(--color-loss)"
          strokeWidth={2.4}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.15, ease: ENTER_EASE, delay: 0.1 }}
        />

        {/* break-even marker */}
        {curve.break_even_depth !== null && (
          <g>
            <motion.path
              d={breakEvenPath}
              stroke="var(--color-loss)"
              strokeDasharray="4 4"
              opacity={0.65}
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.65 }}
              transition={{ delay: 1, duration: 0.4 }}
            />
            <text
              x={xOf(curve.break_even_depth) + 6}
              y={PAD.top + 10}
              className="fill-muted-foreground text-[11px]"
            >
              break-even {(curve.break_even_depth * 100).toFixed(0)}%
            </text>
          </g>
        )}

        {/* recommended marker */}
        {rec !== null && (
          <g>
            <motion.circle
              cx={xOf(rec)}
              cy={yOf(curve.recommended_profit_usd ?? 0)}
              r={5}
              fill="var(--color-profit)"
              animate={{ opacity: [1, 0.5, 1], scale: [1, 1.2, 1] }}
              transition={{ repeat: Infinity, duration: 2.6, ease: "easeInOut" }}
            />
            <text
              x={xOf(rec) - 10}
              y={yOf(curve.recommended_profit_usd ?? 0) - 14}
              textAnchor="end"
              className="fill-foreground text-[11px] font-medium"
            >
              recommended {(rec * 100).toFixed(0)}% · {usd(curve.recommended_profit_usd ?? 0)}
            </text>
          </g>
        )}

        {/* chosen depth — draggable */}
        <motion.g
          initial={false}
          animate={{ x: xOf(chosenDepth) }}
          transition={dragging ? { duration: 0 } : { type: "spring", stiffness: 260, damping: 26 }}
        >
          <line
            x1={0}
            x2={0}
            y1={PAD.top}
            y2={PAD.top + PLOT_H}
            stroke="var(--chart-crosshair)"
            strokeDasharray="5 5"
            opacity={dragging ? 0.9 : 0.55}
          />
          <motion.circle
            cy={yOf(chosen.point.profit_usd)}
            r={dragging ? 6.5 : 5}
            fill="var(--background)"
            stroke="var(--chart-crosshair)"
            strokeWidth={2}
            animate={dragging ? { r: 6.5 } : { r: 5 }}
          />
          {/* wide invisible drag strip */}
          <rect
            x={-16}
            y={PAD.top}
            width={32}
            height={PLOT_H}
            fill="transparent"
            style={{ cursor: "ew-resize" }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              setDragging(true)
            }}
            onPointerMove={(e) => {
              if (!dragging) return
              const d = depthFromClientX(e.clientX)
              if (Math.abs(d - chosenDepth) > 0.0005) onChooseDepth(d)
            }}
            onPointerUp={() => {
              setDragging(false)
              onChooseDepth(nearest(chosenDepth).depth)
            }}
            onPointerCancel={() => {
              setDragging(false)
            }}
          />
        </motion.g>

        {/* hover crosshair */}
        {hoverPoint && !dragging && (
          <motion.g animate={{ x: xOf(hoverPoint.depth) }} transition={{ duration: 0.06 }}>
            <line
              x1={0}
              x2={0}
              y1={PAD.top}
              y2={PAD.top + PLOT_H}
              stroke="var(--chart-crosshair)"
              strokeOpacity="0.25"
            />
          </motion.g>
        )}
      </svg>
    </div>
  )
}
