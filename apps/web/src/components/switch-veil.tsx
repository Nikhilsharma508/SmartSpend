import { Loader2 } from "lucide-react"

/**
 * Covers a page while the *next* model's payload is loading.
 *
 * A model switch must not look like a page reload: the outgoing content stays
 * mounted (so the scroll position and the page height survive), while an
 * opaque veil hides the outgoing numbers — the alternative (a skeleton) would
 * collapse the page and throw the reader back to the top, and showing the old
 * model's numbers under the new model's name would be a lie.
 */
export function SwitchVeil({ show, label }: { show: boolean; label: string }) {
  if (!show) return null
  return (
    <div
      data-switch-veil
      aria-live="polite"
      className="absolute inset-0 z-20 flex items-start justify-center bg-background/92 pt-24 backdrop-blur-[2px]"
    >
      <div className="flex items-center gap-2 rounded-full border border-border/60 bg-card px-3 py-1.5 text-xs text-muted-foreground shadow-sm">
        <Loader2 className="size-3.5 animate-spin" />
        loading {label}…
      </div>
    </div>
  )
}
