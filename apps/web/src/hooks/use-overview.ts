import { useQuery } from "@tanstack/react-query"
import { fetchOverview } from "@/api/client"
import { useModels } from "@/hooks/use-models"
import { useModelSettings } from "@/state/model-settings"

/** Process-wide dashboard context, computed live by FastAPI for the *selected*
 * model and cached server-side per model after the first request.
 *
 * `placeholderData` keeps the outgoing payload mounted while the next model's
 * payload is fetched: the page does not collapse to a skeleton (which would
 * reset the scroll position) — instead the pages veil the old numbers until
 * the new ones arrive. `switchingLabel` names the model being loaded. */
export function useOverview() {
  const { model, researchMode } = useModelSettings()
  const { data: catalog } = useModels()
  const query = useQuery({
    queryKey: ["overview", model, researchMode],
    queryFn: () => fetchOverview(model, researchMode),
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  })
  const switchingLabel = catalog?.models.find((m) => m.name === model)?.label ?? model
  return { ...query, switchingLabel }
}
