import type { CopilotPackage, Overview } from "@/api/types"

const JSON_HEADERS = { Accept: "application/json" }

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: JSON_HEADERS })
  if (!res.ok) throw new Error(`SmartSpend API responded ${res.status}`)
  return (await res.json()) as T
}

/** Everything the planner, segments and test-design pages render. */
export function fetchOverview(): Promise<Overview> {
  return getJson<Overview>("/api/v1/overview")
}

export interface ApiHealth {
  status: string
  service: string
}

/** Liveness probe for the header badge. */
export function fetchHealth(): Promise<ApiHealth> {
  return getJson<ApiHealth>("/health")
}

export type WriterProvider = "template" | "ollama" | "neurometric"

/** One copilot request — the whole gate → plan → write → validate pipeline. */
export async function askCopilot(
  question: string,
  provider?: WriterProvider | null,
  signal?: AbortSignal,
): Promise<CopilotPackage> {
  const res = await fetch("/api/v1/copilot", {
    method: "POST",
    headers: { ...JSON_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({ question, provider: provider ?? null }),
    signal,
  })
  if (!res.ok) {
    let detail = `SmartSpend API responded ${res.status}`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (body?.detail) detail = String(body.detail)
    } catch {
      // keep the status-code message
    }
    throw new Error(detail)
  }
  return (await res.json()) as CopilotPackage
}
