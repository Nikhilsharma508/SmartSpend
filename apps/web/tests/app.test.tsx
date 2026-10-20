import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import { TooltipProvider } from "@/components/ui/tooltip"
import App from "@/App"
import overview from "./overview.json"

// A small handcrafted catalog/comparison for the Model Lab pages; the numbers
// are irrelevant here (unit test), the shapes are the contract.
const modelsCatalog = {
  default: "champion",
  research_mode: false,
  models: [
    {
      name: "champion",
      label: "Response baseline",
      learner: "response",
      tier: "champion",
      score_kind: "response_probability",
      supports_intervals: false,
      research_only: false,
      use_case: "budget plans on response likelihood",
      warning: null,
      registered: true,
      selectable: true,
      added_to_allow_list_by_mode: false,
      trained_at: "2026-10-20T12:00:00",
      seed42_metrics: { qini_auc: 0.0199, profit_at_k_usd: 857.6 },
      meta_error: null,
    },
    {
      name: "dr-learner",
      label: "DR learner",
      learner: "dr_learner",
      tier: "challenger",
      score_kind: "uplift_effect",
      supports_intervals: false,
      research_only: false,
      use_case: "incremental effect per customer",
      warning: null,
      registered: true,
      selectable: true,
      added_to_allow_list_by_mode: false,
      trained_at: "2026-10-20T12:00:00",
      seed42_metrics: { qini_auc: 0.0081, profit_at_k_usd: 723.98 },
      meta_error: null,
    },
    {
      name: "s-learner",
      label: "S learner",
      learner: "s_learner",
      tier: "research",
      score_kind: "uplift_effect",
      supports_intervals: false,
      research_only: true,
      use_case: "research only",
      warning: "weak on the benchmark",
      registered: true,
      selectable: false,
      added_to_allow_list_by_mode: false,
      trained_at: "2026-10-20T12:00:00",
      seed42_metrics: { qini_auc: 0.0069, profit_at_k_usd: 194.04 },
      meta_error: null,
    },
  ],
  score_meanings: {
    response_probability:
      "likelihood this customer converts if contacted (P(convert | treated)).",
    uplift_effect: "estimated incremental effect of contact (treated − control).",
  },
}

const modelsComparison = {
  rows: modelsCatalog.models.map((m) => ({
    name: m.name,
    label: m.label,
    learner: m.learner,
    tier: m.tier,
    score_kind: m.score_kind,
    supports_intervals: m.supports_intervals,
    research_only: m.research_only,
    use_case: m.use_case,
    warning: m.warning,
    qini_auc: m.seed42_metrics.qini_auc,
    uplift_at_k: 0.005,
    incremental_conversions_at_k: 42.3,
    profit_at_k_usd: m.seed42_metrics.profit_at_k_usd,
    profit_per_contact_usd: 0.1,
    n_test: 8523,
    trained_at: m.trained_at as string,
    benchmark_warning: "see the 10-seed benchmark in docs/concepts.md",
  })),
  note: "Every registered model scored once on the same seed-42 test rows.",
  score_meanings: modelsCatalog.score_meanings,
}

// The pages fetch /api/v1/overview and /health; there is no in-app fallback.
const json = (body: unknown) => ({ ok: true, json: async () => body })

globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
  const url = String(input)
  if (url.includes("/health")) return json({ status: "ok", service: "SmartSpend" })
  if (url.includes("/api/v1/models/compare")) return json(modelsComparison)
  if (url.includes("/api/v1/models")) {
    return json({ ...modelsCatalog, research_mode: url.includes("research_mode=true") })
  }
  return json(overview)
}) as unknown as typeof fetch


class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver
  // No-op WAAPI so Motion's canceled-animation `.finished` promises can't
  // reject as unhandled errors under happy-dom.
  Element.prototype.animate = (function () {
    return (() => ({
      cancel: () => undefined,
      finish: () => undefined,
      pause: () => undefined,
      play: () => undefined,
      onfinish: null,
      finished: Promise.resolve(),
    })) as unknown as typeof Element.prototype.animate
  })()
  window.matchMedia =
    window.matchMedia ??
    ((query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList)
})

afterEach(() => cleanup())

function renderAt(path: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
      <QueryClientProvider client={client}>
        <TooltipProvider delayDuration={200}>
          <MemoryRouter initialEntries={[path]}>
            <App />
          </MemoryRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>,
  )
}

describe("app routes render from the live API payload", () => {
  it("planner shows the tool's KPIs, chart card and export", async () => {
    renderAt("/")
    // nav link plus the page heading, plus the tool's KPIs
    expect(await screen.findByRole("heading", { name: "Planner" })).toBeTruthy()
    expect(screen.getAllByRole("link", { name: "Planner" }).length).toBeGreaterThanOrEqual(1)
    expect(await screen.findByText("Recommended depth")).toBeTruthy()
    expect(screen.getByText("Break-even depth")).toBeTruthy()
    expect(screen.getByText("Scenario comparison")).toBeTruthy()
    expect(await screen.findByText("Download CSV")).toBeTruthy()
  })

  it("segments shows rule cards, fairness audit and the model card", async () => {
    renderAt("/segments")
    expect(await screen.findByText("Fairness audit")).toBeTruthy()
    expect(screen.getByText("Model card")).toBeTruthy()
    expect(screen.getByText("Segment 1")).toBeTruthy()
  })

  it("test design shows the MDE grid", async () => {
    renderAt("/test-design")
    expect(await screen.findByText("All four test sizes")).toBeTruthy()
    expect(screen.getByText("How to read this")).toBeTruthy()
  })

  it("copilot asks the live pipeline and explains its architecture", async () => {
    renderAt("/copilot")
    expect(await screen.findByText("Campaign copilot")).toBeTruthy()
    expect(
      screen.getByText("LLM architecture — how an answer is built"),
    ).toBeTruthy()
    // the rail cards were removed; the pipeline graph card is the rail now
    expect(screen.getByText("the LLM does two jobs — routing and prose; every number is code")).toBeTruthy()
    // the offline-playback UI is gone
    expect(screen.queryByText(/offline playback/i)).toBeNull()
    expect(screen.queryByText("Recorded corpus")).toBeNull()
  })

  it("model lab lists the catalog and locks research-only models normally", async () => {
    renderAt("/models")
    expect(await screen.findByText("Model lab")).toBeTruthy()
    // labels appear both in cards and the comparison table
    expect(screen.getAllByText("Response baseline").length).toBeGreaterThan(0)
    expect(screen.getAllByText("DR learner").length).toBeGreaterThan(0)
    // research-only tier is listed but locked (not selectable) by default
    expect(screen.getAllByText("research").length).toBeGreaterThan(0)
    expect(screen.getAllByText("research-only").length).toBeGreaterThan(0)
    // research mode toggle present, off by default
    expect(screen.getByRole("button", { name: /Research Mode off/ })).toBeTruthy()
    // comparison table also renders (same mock serves /models/compare)
    expect(await screen.findByText("Same split, one table")).toBeTruthy()
  })
})
