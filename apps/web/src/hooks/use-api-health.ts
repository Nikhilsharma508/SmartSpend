import { useQuery } from "@tanstack/react-query"
import { fetchHealth } from "@/api/client"

/** Polls the FastAPI service so the header badge reflects reality. */
export function useApiHealth() {
  return useQuery({
    queryKey: ["api-health"],
    queryFn: fetchHealth,
    retry: false,
    refetchInterval: 15_000,
    staleTime: 5_000,
  })
}
