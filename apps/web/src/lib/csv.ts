/** RFC-4180-ish CSV writer: quotes fields containing delimiters/quotes/newlines. */
export function buildCsv(rows: Record<string, string | number | boolean | null>[]): string {
  if (rows.length === 0) return ""
  const cols = Object.keys(rows[0]!)
  const escape = (v: string | number | boolean | null): string => {
    const s = v === null ? "" : String(v)
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [
    cols.join(","),
    ...rows.map((r) => cols.map((c) => escape(r[c])).join(",")),
  ].join("\n")
}

export function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
