import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiUrl, downloadFile, errorMessage, getJSON } from "@/lib/api";

describe("API client error handling", () => {
  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("builds API URLs without duplicating slashes", () => {
    expect(apiUrl("/api/readiness")).toMatch(/\/api\/readiness$/);
  });

  it("preserves actionable server error details", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: "Two sample outputs are missing: sample-2 and sample-9.",
      code: "missing_samples",
      field_errors: { file: "Add every required sample." },
      request_id: "request-123",
      retryable: false,
    }), {
      status: 422,
      headers: { "Content-Type": "application/json" },
    })));

    await expect(getJSON("/api/test")).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      code: "missing_samples",
      requestId: "request-123",
      fieldErrors: { file: "Add every required sample." },
      message: "Two sample outputs are missing: sample-2 and sample-9.",
    });
  });

  it("includes request references for retryable failures", () => {
    const error = new ApiError("Submission scoring failed.", {
      status: 503,
      requestId: "request-456",
      retryable: true,
    });

    expect(errorMessage(error)).toContain("Request reference: request-456.");
  });

  it("uses the CSRF token for protected POST downloads", async () => {
    localStorage.setItem("lb_csrf_token", "csrf-token");
    const fetchMock = vi.fn().mockResolvedValue(new Response("archive", {
      status: 200,
      headers: { "Content-Disposition": 'attachment; filename="backup.zip"' },
    }));
    vi.stubGlobal("fetch", fetchMock);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:test"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    const filename = await downloadFile(
      "/api/admin/backups/download",
      "backup.zip",
      { method: "POST" },
    );

    expect(filename).toBe("backup.zip");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/admin\/backups\/download$/),
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        headers: { "X-CSRF-Token": "csrf-token" },
      }),
    );
  });
});
