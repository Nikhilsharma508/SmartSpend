import { fileURLToPath, URL } from "node:url"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vitest/config"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    // Fail loudly instead of silently hopping to 5174/5175 — the documented
    // URL (and anything bookmarked) must stay 5173. `scripts/start_app.sh`
    // frees a stale SmartSpend listener before starting.
    strictPort: true,
    // Proxy straight to the IPv4 loopback uvicorn binds by default. `localhost`
    // can resolve to ::1 first on macOS, which added connect latency.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  test: {
    environment: "happy-dom",
  },
})
