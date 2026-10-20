import type {
  CopilotPackage,
  ModelCatalog,
  ModelComparison,
  Overview,
} from "@/api/types"

const JSON_HEADERS = { Accept: "application/json" }

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: JSON_HEADERS })
  if (!res.ok) throw new Error(`SmartSpend API responded ${res.status}`)
  return (await res.json()) as T
}

/** Everything the planner, segments and test-design pages render. */
export function fetchOverview(model: string, researchMode: boolean): Promise<Overview> {
  const params = new URLSearchParams({ model, research_mode: String(researchMode) })
  return getJson<Overview>(`/api/v1/overview?${params}`)
}

export interface ApiHealth {
  status: string
  service: string
}

/** Liveness probe for the header badge. */
export function fetchHealth(): Promise<ApiHealth> {
  return getJson<ApiHealth>("/health")
}

/** The model lab's snapshot of the registered catalog. */
export function fetchModels(researchMode: boolean): Promise<ModelCatalog> {
  const params = new URLSearchParams({ research_mode: String(researchMode) })
  return getJson<ModelCatalog>(`/api/v1/models?${params}`)
}

/** Every registered model scored once on the same test rows. */
export function fetchModelComparison(): Promise<ModelComparison> {
  return getJson<ModelComparison>("/api/v1/models/compare")
}

export type WriterProvider = "template" | "ollama" | "neurometric"

/** One copilot request — the whole gate → plan → write → validate pipeline. */
export async function askCopilot(
  question: string,
  provider?: WriterProvider | null,
  options?: { model?: string | null; researchMode?: boolean },
  signal?: AbortSignal,
): Promise<CopilotPackage> {
  const res = await fetch("/api/v1/copilot", {
    method: "POST",
    headers: { ...JSON_HEADERS, "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      question,
      provider: provider || null,
      model: options?.model || null,
      research_mode: Boolean(options?.researchMode),
    }),
  })
  if (!res.ok) throw new Error(`SmartSpend API responded ${res.status}`)
  return (await res.json()) as CopilotPackage
}
