import type { Fixtures } from "@/api/types"
import { fixtures } from "@/api/mock/fixtures"

/**
 * Until apps/api exists the app plays back recorded fixtures. Flipping
 * `USE_MOCK` to false is the whole switch — the FastAPI service will serve the
 * same aggregate under /api/fixtures (see apps/api scaffolding note in
 * AGENTS.md), so no page code changes.
 */
export const USE_MOCK = false

export async function fetchFixtures(): Promise<Fixtures> {
  if (USE_MOCK) return fixtures
  const res = await fetch("/api/fixtures", {
    headers: { Accept: "application/json" },
  })
  if (!res.ok) throw new Error(`SmartSpend API responded ${res.status}`)
  return (await res.json()) as Fixtures
}
