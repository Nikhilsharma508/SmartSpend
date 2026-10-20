import { useId } from "react"
import { cn } from "@/lib/utils"

export interface RangeSliderProps {
  value: number
  min: number
  max: number
  step: number
  onValueChange: (value: number) => void
  /** Small caption above the track. */
  label: string
  /** The big readout — the number, formatted for the reader. */
  valueText: string
  /** Optional tick captions under the track (left → right). */
  ticks?: string[]
  className?: string
}

/**
 * A quiet, readable range control: a filled track, a knob, a caption and a
 * value. The native `input[type=range]` sits invisible on top, so dragging,
 * arrow keys and screen readers all work without a widget framework.
 */
export function RangeSlider({
  value,
  min,
  max,
  step,
  onValueChange,
  label,
  valueText,
  ticks,
  className,
}: RangeSliderProps) {
  const id = useId()
  const pct = max > min ? Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)) : 0
  return (
    <div className={cn("space-y-2.5", className)}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-[11px] uppercase tracking-wider text-muted-foreground">
          {label}
        </label>
        <span className="text-lg font-semibold leading-none tabular-nums">{valueText}</span>
      </div>
      <div className="group relative h-5 rounded-full focus-within:outline-none">
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-muted" />
        <div
          className="absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-primary transition-[width] duration-150"
          style={{ width: `${pct}%` }}
        />
        <span
          aria-hidden
          className="pointer-events-none absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-background shadow-sm transition-transform group-hover:scale-110"
          style={{ left: `${pct}%` }}
        />
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onValueChange(Number(e.target.value))}
          aria-label={label}
          aria-valuetext={valueText}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
        />
      </div>
      {ticks && (
        <div className="flex items-baseline justify-between text-[10.5px] text-muted-foreground">
          {ticks.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      )}
    </div>
  )
}
