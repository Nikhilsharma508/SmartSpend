import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import { TooltipProvider } from "@/components/ui/tooltip"
import App from "@/App"
import overview from "./overview.json"

// The pages fetch /api/v1/overview and /health; there is no in-app fallback.
const json = (body: unknown) => ({ ok: true, json: async () => body })

globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
  const url = String(input)
  if (url.includes("/health")) return json({ status: "ok", service: "SmartSpend" })
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
    expect(await screen.findByText("The grid the tool produces")).toBeTruthy()
    expect(screen.getByText("How to read this")).toBeTruthy()
  })

  it("copilot asks the live pipeline and explains its architecture", async () => {
    renderAt("/copilot")
    expect(await screen.findByText("Campaign copilot")).toBeTruthy()
    expect(
      screen.getByText("LLM architecture — how an answer is built"),
    ).toBeTruthy()
    expect(screen.getByText("Last live run")).toBeTruthy()
    // the offline-playback UI is gone
    expect(screen.queryByText(/offline playback/i)).toBeNull()
    expect(screen.queryByText("Recorded corpus")).toBeNull()
  })
})
