import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import { TooltipProvider } from "@/components/ui/tooltip"
import App from "@/App"
import { fixtures } from "@/api/mock/fixtures"

globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => fixtures,
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

describe("app routes render from fixtures", () => {
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
    expect(await screen.findByText("The grid the tool produces")).toBeTruthy()
    expect(screen.getByText("How to read this")).toBeTruthy()
  })

  it("copilot replays a recorded run", async () => {
    renderAt("/copilot")
    expect(await screen.findByText("Campaign copilot")).toBeTruthy()
    expect(await screen.findByText("System card — live service")).toBeTruthy()
    expect(screen.getByText("Recorded corpus")).toBeTruthy()
    // the seeded playback of the first seed case + its corpus chip
    expect(screen.getAllByText("plan_budget_500").length).toBeGreaterThanOrEqual(1)
  })
})
