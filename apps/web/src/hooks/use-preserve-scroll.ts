import { useEffect, useRef } from "react"

/**
 * Keep the reader where they are while a model's payload is swapped in.
 *
 * The page stays mounted during the swap, so the scroll offset is kept
 * explicitly: the browser's own scroll anchoring would let the position drift
 * by the height of whatever changed above the fold. If the reader scrolls
 * during the swap, their choice wins and nothing is restored.
 */
export function usePreserveScrollDuringSwap(swapping: boolean) {
  const saved = useRef<number | null>(null)
  const userMoved = useRef(false)
  const anchorStyle = useRef<string | null>(null)

  useEffect(() => {
    if (swapping) {
      saved.current = window.scrollY
      userMoved.current = false
      // the browser's scroll anchoring would move the offset when the content
      // above the fold changes height — the explicit restore below is the
      // authority while the swap runs
      anchorStyle.current = document.body.style.overflowAnchor
      document.body.style.overflowAnchor = "none"
      const mark = () => {
        userMoved.current = true
      }
      window.addEventListener("wheel", mark, { passive: true })
      window.addEventListener("touchmove", mark, { passive: true })
      window.addEventListener("keydown", mark)
      return () => {
        window.removeEventListener("wheel", mark)
        window.removeEventListener("touchmove", mark)
        window.removeEventListener("keydown", mark)
      }
    }
    if (anchorStyle.current !== null) {
      document.body.style.overflowAnchor = anchorStyle.current
      anchorStyle.current = null
    }
    if (saved.current !== null && !userMoved.current) {
      const root = document.documentElement
      const previous = root.style.scrollBehavior
      root.style.scrollBehavior = "auto"
      window.scrollTo(0, saved.current)
      root.style.scrollBehavior = previous
    }
    saved.current = null
  }, [swapping])
}
