/** The UI has no fallback data: if FastAPI is down, say so plainly. */
export function LoadError({ error }: { error: unknown }) {
  const message = (error as Error | null)?.message ?? "unknown error"
  return (
    <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm">
      <p className="font-medium">The SmartSpend API did not answer.</p>
      <p className="mt-1 text-muted-foreground">
        {message} — start the stack with <span className="font-mono">npm run start</span>.
        The UI has no recorded fallback.
      </p>
    </div>
  )
}
