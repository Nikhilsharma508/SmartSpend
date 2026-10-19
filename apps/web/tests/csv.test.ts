import { describe, expect, it } from "vitest"
import { buildCsv } from "@/lib/csv"

describe("buildCsv", () => {
  it("writes headers and simple rows", () => {
    const csv = buildCsv([{ a: 1, b: "x" }, { a: 2, b: "y" }])
    expect(csv).toBe("a,b\n1,x\n2,y")
  })

  it("quotes fields containing quote characters", () => {
    const csv = buildCsv([
      { rules: "history <= 100.4 AND visits > 0", note: "say \"hi\", ok" },
    ])
    // comma in the note forces quoting around it; the rules field has no delimiter
    expect(csv.split("\n")).toEqual([
      "rules,note",
      'history <= 100.4 AND visits > 0,"say ""hi"", ok"',
    ])
  })

  it("emits null as empty string", () => {
    expect(buildCsv([{ a: null }])).toBe("a\n")
  })

  it("returns empty string for no rows", () => {
    expect(buildCsv([])).toBe("")
  })
})
