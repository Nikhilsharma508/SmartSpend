import type { ProfitPoint } from "@/api/types"

export const intFmt = (n: number) => Math.round(n).toLocaleString("en-US")

export const usd = (n: number, digits = 0) =>
  `${n < 0 ? "−" : ""}$${Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`

export const usdSigned = (n: number, digits = 0) =>
  `${n >= 0 ? "+" : "−"}$${Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`

/** 0.34 → "+34.0%" or "0.071 → +7.1%"; callers add the +/- context. */
export const rateFmt = (n: number, digits = 1) =>
  `${(n * 100).toFixed(digits)}%`

export const bpFmt = (n: number) =>
  `${n >= 0 ? "+" : "−"}${(Math.abs(n) * 10_000).toFixed(0)} bp`

/** Linear interpolation over the tool's depth grid — labelled where used. */
export function interpPoint(
  points: ProfitPoint[],
  depth: number,
): { point: ProfitPoint; exact: boolean } {
  if (points.length === 0) throw new Error("empty curve")
  const first = points[0]!
  if (depth <= first.depth) return { point: first, exact: true }
  const last = points[points.length - 1]!
  if (depth >= last.depth) return { point: last, exact: true }
  let lo = first
  let hi = last
  for (const p of points) {
    if (p.depth === depth) return { point: p, exact: true }
    if (p.depth < depth && p.depth > lo.depth) lo = p
    if (p.depth > depth && p.depth < hi.depth) hi = p
  }
  const t = (depth - lo.depth) / (hi.depth - lo.depth)
  const mix = (f: (p: ProfitPoint) => number) => f(lo) * (1 - t) + f(hi) * t
  return {
    point: {
      depth,
      n_targeted: Math.round(mix(p => p.n_targeted)),
      incremental_conversions: mix(p => p.incremental_conversions),
      incremental_conversions_se: mix(p => p.incremental_conversions_se),
      profit_usd: mix(p => p.profit_usd),
      profit_ci_low_usd: mix(p => p.profit_ci_low_usd),
      profit_ci_high_usd: mix(p => p.profit_ci_high_usd),
    },
    exact: false,
  }
}

export const seconds = (n: number) =>
  n >= 1 ? `${n.toFixed(2)} s` : `${Math.round(n * 1000)} ms`
