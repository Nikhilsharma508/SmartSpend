import { useEffect, useState, type ReactNode } from "react"
import { NavLink, Link, useLocation } from "react-router-dom"
import { AnimatePresence, motion } from "motion/react"
import { useTheme } from "next-themes"
import { Moon, Sun } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { MEASURED } from "@/api/mock/fixtures"
import { cn } from "@/lib/utils"

const TABS = [
  { to: "/", label: "Planner" },
  { to: "/segments", label: "Segments" },
  { to: "/test-design", label: "Test design" },
  { to: "/copilot", label: "Copilot" },
]

const OFFLINE_NOTE = `Offline playback of recorded tool runs (${MEASURED.source}). The FastAPI service goes live next.`

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  // async so the effect body itself never calls setState synchronously
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    const id = setTimeout(() => setMounted(true), 0)
    return () => clearTimeout(id)
  }, [])
  const dark = mounted && resolvedTheme === "dark"
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(dark ? "light" : "dark")}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={dark ? "moon" : "sun"}
          initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
          animate={{ rotate: 0, opacity: 1, scale: 1 }}
          exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
          transition={{ type: "spring", stiffness: 300, damping: 22 }}
          className="flex"
        >
          {dark ? <Moon className="size-4" /> : <Sun className="size-4" />}
        </motion.span>
      </AnimatePresence>
    </Button>
  )
}

function NavPill() {
  const location = useLocation()
  const [canAnimate, setCanAnimate] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => setCanAnimate(true))
    return () => cancelAnimationFrame(id)
  }, [])
  return (
    <nav className="mx-auto flex items-center gap-1 rounded-full border border-border/60 bg-muted/40 p-1">
      {TABS.map((tab) => {
        const isActive =
          tab.to === "/" ? location.pathname === "/" : location.pathname.startsWith(tab.to)
        return (
          <NavLink
            key={tab.to}
            to={tab.to}
            end={tab.to === "/"}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              "relative flex whitespace-nowrap rounded-full px-3.5 py-1.5 text-[13px] font-medium transition-colors",
              isActive
                ? "text-primary-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {isActive && canAnimate && (
              <motion.span
                layoutId="nav-pill"
                className="absolute inset-0 rounded-full bg-primary"
                transition={{ type: "spring", stiffness: 420, damping: 32 }}
              />
            )}
            {isActive && !canAnimate && (
              <span className="absolute inset-0 rounded-full bg-primary" />
            )}
            <span className="relative z-10">{tab.label}</span>
          </NavLink>
        )
      })}
    </nav>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-dvh">
      {/* subtle masked grid backdrop */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 opacity-40 dark:opacity-60"
        style={{
          backgroundImage:
            "linear-gradient(var(--chart-grid) 1px, transparent 1px), linear-gradient(90deg, var(--chart-grid) 1px, transparent 1px)",
          backgroundSize: "44px 44px",
          maskImage:
            "radial-gradient(ellipse 90% 60% at 50% 0%, black 30%, transparent 75%)",
          WebkitMaskImage:
            "radial-gradient(ellipse 90% 60% at 50% 0%, black 30%, transparent 75%)",
        }}
      />

      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-6">
          <Link to="/" className="group flex items-center gap-2.5">
            <motion.span
              className="relative flex size-5 items-center justify-center rounded-md bg-primary"
              whileHover={{ rotate: -8, scale: 1.08 }}
              transition={{ type: "spring", stiffness: 380, damping: 18 }}
            >
              <motion.span
                className="size-1.5 rounded-full"
                style={{ background: "var(--color-profit)" }}
                animate={{ opacity: [1, 0.35, 1] }}
                transition={{ repeat: Infinity, duration: 2.2, ease: "easeInOut" }}
              />
            </motion.span>
            <span className="text-sm font-semibold tracking-tight">SmartSpend</span>
            <span className="hidden text-xs text-muted-foreground sm:inline">
              · campaign planner
            </span>
          </Link>

          <NavPill />

          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className="gap-1.5 border-dashed text-[11px] text-muted-foreground"
                >
                  <motion.span
                    className="size-1.5 rounded-full bg-amber-400"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ repeat: Infinity, duration: 1.8 }}
                  />
                  offline playback
                </Badge>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-72">
                {OFFLINE_NOTE}
              </TooltipContent>
            </Tooltip>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="relative mx-auto max-w-6xl px-6 pb-24 pt-8">{children}</main>

      <footer className="relative mx-auto max-w-6xl px-6 pb-10 text-xs text-muted-foreground">
        Profit in USD at the Day-1 assumptions (30% margin · $0.10/contact ·
        $34.91/conversion). Numbers come from the registered champion on its
        seed-42 test split (n = 8,523) — the tools' own outputs.
      </footer>
    </div>
  )
}
