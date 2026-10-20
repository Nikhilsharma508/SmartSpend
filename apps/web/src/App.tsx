import { Route, Routes, useLocation } from "react-router-dom"
import { AnimatePresence, MotionConfig, motion } from "motion/react"
import { AppShell } from "@/components/layout/app-shell"
import PlannerPage from "@/pages/planner-page"
import SegmentsPage from "@/pages/segments-page"
import TestDesignPage from "@/pages/test-design-page"
import CopilotPage from "@/pages/copilot-page"
import ModelsPage from "@/pages/models-page"
import { ModelSettingsProvider } from "@/state/model-settings"

const PAGE_EASE = [0.22, 1, 0.36, 1] as const

/**
 * Tab-to-tab navigation is lateral — it carries no spatial depth — so the
 * swap is a soft crossfade (with a barely-there settle) rather than a
 * directional slide. Directional slides are reserved for hierarchical
 * navigation (list → detail), which this app does not have.
 */
export default function App() {
  const location = useLocation()
  return (
    <ModelSettingsProvider>
      <MotionConfig reducedMotion="user">
        <AppShell>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, scale: 0.996 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18, ease: PAGE_EASE }}
            >
              <Routes location={location}>
                <Route path="/" element={<PlannerPage />} />
                <Route path="/segments" element={<SegmentsPage />} />
                <Route path="/test-design" element={<TestDesignPage />} />
                <Route path="/copilot" element={<CopilotPage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="*" element={<PlannerPage />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </AppShell>
      </MotionConfig>
    </ModelSettingsProvider>
  )
}
