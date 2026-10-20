import { useQuery } from "@tanstack/react-query"
import { fetchModelComparison } from "@/api/client"

/** Every registered model scored once on the same test rows (the lab table). */
export function useModelComparison() {
  return useQuery({
    queryKey: ["models-compare"],
    queryFn: fetchModelComparison,
    staleTime: Infinity,
  })
}
