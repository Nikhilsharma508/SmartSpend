import { useQuery } from "@tanstack/react-query"
import { fetchFixtures } from "@/api/client"

/** Process-wide context: the champion + its split + the day's tool outputs.
 * Fits packages.copilot.planner.get_context()'s one-boot-per-process shape. */
export function useFixtures() {
  return useQuery({
    queryKey: ["fixtures"],
    queryFn: fetchFixtures,
    staleTime: Infinity,
  })
}
