import { useEffect, useState, type ReactNode } from "react"
import { NavLink, Link, useLocation } from "react-router-dom"
import { AnimatePresence, motion } from "motion/react"
import { useTheme } from "next-themes"
import { Check, ChevronDown, Lock, Moon, Sun, Unlock } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useApiHealth } from "@/hooks/use-api-health"
import { useModels } from "@/hooks/use-models"
import { applyModelChoice, toggleResearchMode } from "@/lib/model-actions"
import { cn } from "@/lib/utils"
import { useModelSettings } from "@/state/model-settings"

const TABS = [
  { to: "/", label: "Planner" },
  { to: "/segments", label: "Segments" },
  { to: "/test-design", label: "Test design" },
  { to: "/models", label: "Models" },
  { to: "/copilot", label: "Copilot" },
]

/* ------------------------------------------------------------------ */
/* theme: auto follows local time, then manual light / dark            */

type ThemeMode = "auto" | "light" | "dark"

const THEME_KEY = "smartspend-theme-mode"
const AUTO_DARK_FROM = 19 // 19:00 local
const AUTO_DARK_UNTIL = 7 // 07:00 local

function autoTheme(now = new Date()): "light" | "dark" {
  const hour = now.getHours()
  return hour >= AUTO_DARK_FROM || hour < AUTO_DARK_UNTIL ? "dark" : "light"
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const [mode, setMode] = useState<ThemeMode>(() => {
    const stored = localStorage.getItem(THEME_KEY)
    return stored === "light" || stored === "dark" || stored === "auto" ? stored : "auto"
  })

  useEffect(() => {
    if (mode !== "auto") {
      setTheme(mode)
      return
    }
    const apply = () => setTheme(autoTheme())
    apply()
    // cross the day/night boundary without a reload
    const id = setInterval(apply, 60_000)
    return () => clearInterval(id)
  }, [mode, setTheme])

  const dark = resolvedTheme === "dark"
  const next: ThemeMode = mode === "auto" ? "light" : mode === "light" ? "dark" : "auto"

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Theme: ${mode === "auto" ? `auto (${dark ? "night" : "day"})` : mode}. Switch to ${next}.`}
          onClick={() => {
            setMode(next)
            localStorage.setItem(THEME_KEY, next)
          }}
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
      </TooltipTrigger>
      <TooltipContent side="bottom">
        Theme: {mode === "auto" ? `auto (${dark ? "night" : "day"})` : mode} · auto follows
        local time (dark 19:00–07:00)
      </TooltipContent>
    </Tooltip>
  )
}

/* ------------------------------------------------------------------ */

function ApiBadge() {
  const { isPending, isError } = useApiHealth()
  const state = isPending ? "connecting" : isError ? "offline" : "live"
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className={cn(
            "gap-1.5 text-[11px]",
            state === "live"
              ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400"
              : state === "offline"
                ? "border-red-400/50 text-red-500"
                : "border-dashed text-muted-foreground",
          )}
        >
          <motion.span
            className={cn(
              "size-1.5 rounded-full",
              state === "live"
                ? "bg-emerald-500"
                : state === "offline"
                  ? "bg-red-500"
                  : "bg-muted-foreground",
            )}
            animate={
              state === "connecting"
                ? { opacity: [1, 0.3, 1] }
                : state === "offline"
                  ? { opacity: [1, 0.4, 1] }
                  : undefined
            }
            transition={{ repeat: Infinity, duration: 1.8 }}
          />
          {state === "live" ? "live api" : state === "offline" ? "api offline" : "connecting"}
        </Badge>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-72">
        {state === "live"
          ? "FastAPI :8000 is healthy. Every number on this page is computed by the decision tools for this request."
          : state === "offline"
            ? "The FastAPI service is not reachable. Start the stack with `npm run start`; the UI has no fallback data."
            : "Checking the FastAPI service on :8000…"}
      </TooltipContent>
    </Tooltip>
  )
}

const TIER_DOT: Record<string, string> = {
  champion: "bg-primary",
  challenger: "bg-sky-400",
  research: "bg-amber-400",
}

/** The answering model, visible on every page. Switching it here recomputes
 * the numbers on the page you are already looking at — no lab round-trip. */
function ModelPill() {
  const { model, researchMode, pickModel, setResearchMode } = useModelSettings()
  const { data } = useModels()
  const current = data?.models.find((m) => m.name === model)
  const actions = { pickModel, setResearchMode }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          aria-label={`Answering model: ${current?.label ?? model} — open to switch`}
          title="The selected model answers every page — switch it here and this page recomputes"
          className="h-7 gap-2 rounded-full border-border/60 px-3 text-[11px] font-medium"
        >
          <span className={cn("size-1.5 rounded-full", TIER_DOT[current?.tier ?? "champion"])} />
          <span className={cn("max-w-36 truncate", researchMode && "text-amber-600 dark:text-amber-300")}>
            {current?.label ?? model}
          </span>
          {researchMode ? (
            <Unlock className="size-3 text-amber-500" />
          ) : (
            <Lock className="size-3 text-muted-foreground/70" />
          )}
          <ChevronDown className="size-3 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        <DropdownMenuLabel className="text-[11px] uppercase tracking-wider">
          Answering model
        </DropdownMenuLabel>
        {data
          ? data.models.map((m) => {
              const active = m.name === model
              const unavailable = !m.registered || (!m.selectable && !researchMode)
              return (
                <DropdownMenuItem
                  key={m.name}
                  disabled={unavailable}
                  onSelect={() => applyModelChoice(m, { pickModel })}
                  className="items-start gap-2.5 py-2"
                >
                  <span className={cn("mt-1.5 size-1.5 shrink-0 rounded-full", TIER_DOT[m.tier])} />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2 text-[12px] font-medium">
                      {m.label}
                      {active && <Check className="size-3 text-primary" />}
                    </span>
                    <span className="block truncate text-[10px] text-muted-foreground">
                      {m.use_case}
                    </span>
                  </span>
                  {m.research_only && <Lock className="mt-0.5 size-3 shrink-0 text-amber-500/80" />}
                </DropdownMenuItem>
              )
            })
          : (
            <p className="px-2 py-3 text-[11px] text-muted-foreground">loading the model catalog…</p>
          )}
        <DropdownMenuSeparator />
        <DropdownMenuCheckboxItem
          checked={researchMode}
          onCheckedChange={(v) =>
            toggleResearchMode(v === true, { models: data?.models ?? [], model, ...actions })
          }
          className={cn("gap-2", researchMode && "text-amber-600 dark:text-amber-300")}
        >
          Research Mode
        </DropdownMenuCheckboxItem>
        <p className="px-2 pb-2 pt-1 text-[10px] leading-relaxed text-muted-foreground">
          Unlocks the benchmark-weak tiers (S/T/X learners, causal forest) for
          exploration. Locked rows stay listed but untouchable until it is on.
        </p>
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link
            to="/models"
            className="text-[12px] text-muted-foreground hover:text-foreground"
          >
            Model lab — all models, side by side →
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
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
            <ModelPill />
            <ApiBadge />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="relative mx-auto max-w-6xl px-6 pb-24 pt-8">{children}</main>

      <footer className="relative mx-auto max-w-6xl px-6 pb-10 text-xs text-muted-foreground">
        Profit in USD at the Day-1 assumptions (30% margin · $0.10/contact ·
        $34.91/conversion). Numbers are computed live from the model in the
        header pill (registered artifact, seed-42 test split) — the tools' own
        outputs.
      </footer>
    </div>
  )
}
