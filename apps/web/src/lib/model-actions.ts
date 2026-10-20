import { toast } from "sonner"
import type { ModelDescriptor } from "@/api/types"

/** Shared pick/research-mode behaviour so the header pill, the Model lab and
 * any future picker apply one set of guards and one tone of voice. The API
 * would 422 anything these guards prevent, so the UI keeps itself legal. */

export interface ModelActions {
  pickModel: (model: string, researchOnly: boolean) => void
  setResearchMode: (on: boolean) => void
}

/** Select a catalog model as the app-wide answering model. */
export function applyModelChoice(m: ModelDescriptor, { pickModel }: Pick<ModelActions, "pickModel">) {
  if (m.research_only) {
    pickModel(m.name, true)
    toast(`${m.label} selected — Research Mode`, {
      description:
        "This model is benchmark-verified weak and unlocked for exploration only. Never drive a live campaign with it.",
    })
    return
  }
  pickModel(m.name, false)
  toast.success(`${m.label} now answers`, {
    description: "Planner, segments and test-design recompute from this model.",
  })
}

/** Flip Research Mode; if it is being turned off while a research-only model
 * is held, snap back to the champion first (the lab's original guard). */
export function toggleResearchMode(
  on: boolean,
  { models, model, pickModel, setResearchMode }: ModelActions & {
    models: ModelDescriptor[]
    model: string
  },
) {
  if (!on) {
    if (models.some((m) => m.research_only && m.name === model)) {
      pickModel("champion", false)
    }
    setResearchMode(false)
    toast("Research Mode off", {
      description: "Answers come from the champion and the DR challenger again; the weak tiers are locked.",
    })
    return
  }
  setResearchMode(true)
  toast("Research Mode on", {
    description:
      "S/T/X learners and the causal forest are now selectable — benchmark-verified weak, for exploration only.",
  })
}
