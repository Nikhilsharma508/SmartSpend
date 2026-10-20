import { useQuery } from "@tanstack/react-query"
import { fetchOverview } from "@/api/client"

/** Process-wide dashboard context: champion evaluation outputs, computed live
 * by FastAPI (cached server-side after the first request). */
export function useOverview() {
  return useQuery({
    queryKey: ["overview"],
    queryFn: fetchOverview,
    staleTime: Infinity,
  })
}
