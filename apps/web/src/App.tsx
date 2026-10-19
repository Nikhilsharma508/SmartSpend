import { Route, Routes, useLocation } from "react-router-dom"
import { AnimatePresence, motion } from "motion/react"
import { AppShell } from "@/components/layout/app-shell"
import PlannerPage from "@/pages/planner-page"
import SegmentsPage from "@/pages/segments-page"
import TestDesignPage from "@/pages/test-design-page"
import CopilotPage from "@/pages/copilot-page"

const PAGE_EASE = [0.22, 1, 0.36, 1] as const

export default function App() {
  const location = useLocation()
  return (
    <AppShell>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.24, ease: PAGE_EASE }}
        >
          <Routes location={location}>
            <Route path="/" element={<PlannerPage />} />
            <Route path="/segments" element={<SegmentsPage />} />
            <Route path="/test-design" element={<TestDesignPage />} />
            <Route path="/copilot" element={<CopilotPage />} />
            <Route path="*" element={<PlannerPage />} />
          </Routes>
        </motion.div>
      </AnimatePresence>
    </AppShell>
  )
}
