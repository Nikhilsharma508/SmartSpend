import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react"

/** Which ranking model answers, and whether research-only models are unlocked.
 *
 * Persisted in localStorage (it is a preference, not data): every page that
 * computes something live (planner, segments, copilot pin) reads the same
 * values, so switching a model in the Model Lab switches what the whole app
 * computes on the next fetch.
 */

const MODEL_KEY = "smartspend-model"
const RESEARCH_KEY = "smartspend-research-mode"

function storedModel(): string {
  const fromStorage = localStorage.getItem(MODEL_KEY)
  if (!fromStorage || !/^[a-z0-9][a-z0-9-]*$/.test(fromStorage)) return "champion"
  // A research-only pick is only restored when Research Mode survived the
  // reload (both keys are written together). In normal mode only champion /
  // dr are legal anyway. If the artifact was since deleted, the pages surface
  // the load error honestly instead of silently guessing another model.
  if (storedResearch()) return fromStorage
  if (fromStorage === "champion" || fromStorage === "dr-learner") return fromStorage
  return "champion"
}

function storedResearch(): boolean {
  return localStorage.getItem(RESEARCH_KEY) === "1"
}

interface ModelSettings {
  model: string
  researchMode: boolean
  setModel: (model: string) => void
  setResearchMode: (on: boolean) => void
  /** Research-only tiers turn the mode on as part of the pick. */
  pickModel: (model: string, researchOnly: boolean) => void
}

const ModelSettingsContext = createContext<ModelSettings | null>(null)

export function ModelSettingsProvider({ children }: { children: ReactNode }) {
  const [model, setModelState] = useState(storedModel)
  const [researchMode, setResearchState] = useState(storedResearch)

  const setModel = useCallback(
    (next: string) => {
      // Normal mode allows only the champion/DR pins; research-only picks go
      // through pickModel, which turns the mode on as part of the choice.
      if (next !== "champion" && next !== "dr-learner" && !researchMode) return
      setModelState(next)
      localStorage.setItem(MODEL_KEY, next)
    },
    [researchMode],
  )

  const pickModel = useCallback((next: string, researchOnly: boolean) => {
    if (researchOnly) {
      setResearchState(true)
      localStorage.setItem(RESEARCH_KEY, "1")
    }
    setModelState(next)
    localStorage.setItem(MODEL_KEY, next)
  }, [])

  const setResearchMode = useCallback((on: boolean) => {
    setResearchState(on)
    localStorage.setItem(RESEARCH_KEY, on ? "1" : "0")
  }, [])

  const value = useMemo(
    () => ({ model, researchMode, setModel, setResearchMode, pickModel }),
    [model, researchMode, setModel, setResearchMode, pickModel],
  )
  return <ModelSettingsContext.Provider value={value}>{children}</ModelSettingsContext.Provider>
}

export function useModelSettings(): ModelSettings {
  const ctx = useContext(ModelSettingsContext)
  if (!ctx) throw new Error("useModelSettings must be used inside ModelSettingsProvider")
  return ctx
}
