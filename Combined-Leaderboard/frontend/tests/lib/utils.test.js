import { describe, expect, it } from "vitest";
import { cn, modelType, safeNext } from "@/lib/utils";


describe("safeNext", () => {
  it("accepts internal paths and rejects redirect-like values", () => {
    expect(safeNext("/profile?tab=account")).toBe("/profile?tab=account");
    expect(safeNext("//untrusted.example/path")).toBe("/submit");
    expect(safeNext("/\\untrusted.example/path")).toBe("/submit");
    expect(safeNext("https://untrusted.example/path")).toBe("/submit");
    expect(safeNext("/profile\nmalformed")).toBe("/submit");
  });
});

describe("modelType", () => {
  it("does not present organization or family metadata as model access", () => {
    expect(modelType({ organization: "Example Lab", family: "Example" })).toBe("N/A");
    expect(modelType({ org: "Example Lab" })).toBe("N/A");
    expect(modelType({ access: "open_weights" })).toBe("Open Weights");
  });
});

describe("cn", () => {
  it("keeps the outline style next to outline width and colour (Tailwind 3 semantics)", () => {
    const ring = "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand";
    expect(cn("grid", ring).split(" ")).toEqual(["grid", ...ring.split(" ")]);
  });

  it("still merges genuinely conflicting outline classes", () => {
    expect(cn("outline-2", "outline-4")).toBe("outline-4");
    expect(cn("outline", "outline-dashed")).toBe("outline-dashed");
  });
});
