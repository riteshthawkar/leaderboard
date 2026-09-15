import { describe, expect, it } from "vitest";
import { buildContentSecurityPolicy } from "../../build-security.js";

describe("deployment CSP", () => {
  it("limits cross-origin connections to the configured API", () => {
    const policy = buildContentSecurityPolicy("https://api.example.test");
    expect(policy).toContain("connect-src 'self' https://api.example.test;");
    expect(policy).not.toContain("connect-src 'self' https: ");
    expect(policy).not.toContain("localhost");
    expect(policy).not.toContain("unsafe-eval");
    expect(policy).toContain("object-src 'none'");
  });
  it("keeps same-origin and static builds same-origin only", () => {
    expect(buildContentSecurityPolicy()).toContain("connect-src 'self';");
  });
  it.each(["https://api.example.test/path", "https://user:password@example.test", "javascript:alert(1)"])("rejects non-origin input", (value) => {
    expect(() => buildContentSecurityPolicy(value)).toThrow();
  });
});
