import js from "@eslint/js"
import globals from "globals"
import reactHooks from "eslint-plugin-react-hooks"
import reactRefresh from "eslint-plugin-react-refresh"
import tseslint from "typescript-eslint"

export default tseslint.config(
  { ignores: ["dist", "coverage", "node_modules"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": "warn",
    },
  },
  // Registry-vendored code (shadcn / KokonutUI / Bklit UI) is generated
  // upstream with its own lint budget: measuring refs during render, setState
  // in effects, and animation-path `any`s are their idiom. Keep the two chart
  // files we authored ourselves under the full rule set.
  {
    files: [
      "src/components/charts/**/*.{ts,tsx}",
      "src/components/kokonutui/**/*.{ts,tsx}",
      "src/components/shimmering-text.tsx",
      "src/hooks/use-auto-resize-textarea.ts",
      "src/lib/profit-loss-segments.ts",
    ],
    ignores: [
      "src/components/charts/profit-curve-chart.tsx",
      "src/components/charts/segment-uplift-chart.tsx",
    ],
    rules: {
      "react-hooks/refs": "off",
      "react-hooks/set-state-in-effect": "off",
      "react-hooks/immutability": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "no-useless-assignment": "off",
    },
  },
  // Fixtures and the Vitest smoke test may use loose object literals.
  {
    files: ["src/api/mock/**/*.ts", "src/api/mock/**/*.tsx", "tests/**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
)
