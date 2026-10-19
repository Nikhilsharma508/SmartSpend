import { useEffect, useRef, useState, type ReactNode } from "react"
import { animate, motion } from "motion/react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

const EASE_OUT = [0.22, 1, 0.36, 1] as const

interface KpiCardProps {
  label: string
  value: number
  /** Stable formatting, e.g. n => usd(n). */
  format: (n: number) => string
  sub?: ReactNode
  /** Optional tooltip explaining where the number comes from. */
  hint?: string
  /** Tailwind classes for the accent dot. */
  accentClass?: string
  className?: string
}

/** Motion-rendered KPI tile with a count-up on value change. */
export function KpiCard({
  label,
  value,
  format,
  sub,
  hint,
  accentClass = "bg-primary",
  className,
}: KpiCardProps) {
  const [display, setDisplay] = useState(() => format(0))
  const formatRef = useRef(format)
  const previous = useRef(0)

  useEffect(() => {
    formatRef.current = format
  })

  // spring-free count-up: animated in the effect, stopped on unmount/retarget
  useEffect(() => {
    const from = previous.current
    previous.current = value
    const controls = animate(from, value, {
      duration: 0.75,
      ease: EASE_OUT,
      onUpdate: (v) => setDisplay(formatRef.current(v)),
    })
    return () => controls.stop()
  }, [value])

  const tile = (
    <motion.div
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 320, damping: 24 }}
      className={cn(
        "group relative overflow-hidden rounded-xl border border-border/60 bg-card/80 p-4 backdrop-blur-sm",
        "shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset]",
        className,
      )}
    >
      <div className="flex items-center gap-2">
        <span className={cn("size-1.5 rounded-full", accentClass)} />
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
      </div>
      <div className="mt-2 font-semibold tabular-nums text-2xl leading-tight tracking-tight">
        {display}
      </div>
      {sub ? <div className="mt-1 text-xs text-muted-foreground">{sub}</div> : null}
      {/* hover sheen */}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/6 to-transparent opacity-0 transition-opacity group-hover:opacity-100"
      />
    </motion.div>
  )

  return hint ? (
    <Tooltip>
      <TooltipTrigger asChild>
        <div>{tile}</div>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-72">
        {hint}
      </TooltipContent>
    </Tooltip>
  ) : (
    tile
  )
}
