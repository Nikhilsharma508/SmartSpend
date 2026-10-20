import { useQuery } from "@tanstack/react-query"
import { fetchModels } from "@/api/client"
import { useModelSettings } from "@/state/model-settings"

/** The model catalog snapshot (registered artifacts + registration status). */
export function useModels(researchModeOverride?: boolean) {
  const { researchMode } = useModelSettings()
  const mode = researchModeOverride ?? researchMode
  return useQuery({
    queryKey: ["models", mode],
    queryFn: () => fetchModels(mode),
    staleTime: Infinity,
  })
}
