import { describe, expect, it } from "vitest";
import { modelType, safeNext } from "@/lib/utils";


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
