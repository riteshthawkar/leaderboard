import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("bearer API transport", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("VITE_AUTH_TRANSPORT", "bearer");
    localStorage.clear();
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("stores only the refresh token in session storage and sends explicit authorization", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        email: "member@example.com",
        auth_transport: "bearer",
        token_type: "Bearer",
        access_token: "access-1",
        refresh_token: "refresh-1",
        expires_in: 900,
      }))
      .mockResolvedValueOnce(jsonResponse({ models: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const api = await import("@/lib/api");

    await api.postJSON("/api/auth/login", {
      email: "member@example.com",
      password: "violet telescope cedar glacier",
    });
    await api.getJSON("/api/models/mine");

    expect(sessionStorage.getItem("lb_refresh_token_v1")).toBe("refresh-1");
    expect(localStorage.getItem("lb_refresh_token_v1")).toBeNull();
    const loginOptions = fetchMock.mock.calls[0][1];
    const protectedOptions = fetchMock.mock.calls[1][1];
    expect(loginOptions.credentials).toBe("omit");
    expect(loginOptions.headers.get("X-Auth-Transport")).toBe("bearer");
    expect(loginOptions.headers.get("Authorization")).toBeNull();
    expect(protectedOptions.credentials).toBe("omit");
    expect(protectedOptions.headers.get("Authorization")).toBe("Bearer access-1");
  });

  it("rotates once after a 401 and retries the original request", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        email: "member@example.com",
        auth_transport: "bearer",
        token_type: "Bearer",
        access_token: "access-old",
        refresh_token: "refresh-old",
        expires_in: 900,
      }))
      .mockResolvedValueOnce(jsonResponse({
        error: "Access token expired.",
        code: "auth_required",
      }, 401))
      .mockResolvedValueOnce(jsonResponse({
        email: "member@example.com",
        auth_transport: "bearer",
        token_type: "Bearer",
        access_token: "access-new",
        refresh_token: "refresh-new",
        expires_in: 900,
      }))
      .mockResolvedValueOnce(jsonResponse({ models: [{ id: "model-1" }] }));
    vi.stubGlobal("fetch", fetchMock);
    const api = await import("@/lib/api");

    await api.postJSON("/api/auth/login", {
      email: "member@example.com",
      password: "violet telescope cedar glacier",
    });
    const result = await api.getJSON("/api/models/mine");

    expect(result.models).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    const refreshCall = fetchMock.mock.calls[2];
    expect(refreshCall[0]).toMatch(/\/api\/auth\/token\/refresh$/);
    expect(JSON.parse(refreshCall[1].body)).toEqual({ refresh_token: "refresh-old" });
    expect(fetchMock.mock.calls[3][1].headers.get("Authorization")).toBe("Bearer access-new");
    expect(sessionStorage.getItem("lb_refresh_token_v1")).toBe("refresh-new");
  });

  it("restores a reloaded tab from its rotating refresh credential", async () => {
    sessionStorage.setItem("lb_refresh_token_v1", "refresh-existing");
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        email: "member@example.com",
        auth_transport: "bearer",
        token_type: "Bearer",
        access_token: "access-restored",
        refresh_token: "refresh-restored",
        expires_in: 900,
      }))
      .mockResolvedValueOnce(jsonResponse({
        authenticated: true,
        email: "member@example.com",
        quota: null,
        email_verified: true,
        auth_provider: "microsoft",
      }));
    vi.stubGlobal("fetch", fetchMock);
    const api = await import("@/lib/api");

    const user = await api.fetchMe();

    expect(user.email).toBe("member@example.com");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      refresh_token: "refresh-existing",
    });
    expect(fetchMock.mock.calls[1][1].headers.get("Authorization")).toBe("Bearer access-restored");
  });

  it("uses bearer credentials for protected POST downloads on Pages", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        email: "member@example.com",
        auth_transport: "bearer",
        token_type: "Bearer",
        access_token: "download-access",
        refresh_token: "download-refresh",
        expires_in: 900,
      }))
      .mockResolvedValueOnce(new Response("archive", {
        headers: { "Content-Disposition": 'attachment; filename="backup.zip"' },
      }));
    vi.stubGlobal("fetch", fetchMock);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true, value: vi.fn(() => "blob:test"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true, value: vi.fn(),
    });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const api = await import("@/lib/api");
    await api.postJSON("/api/auth/login", {
      email: "member@example.com", password: "violet telescope cedar glacier",
    });
    expect(await api.downloadFile("/api/admin/backups/download", "backup.zip", {
      method: "POST",
    })).toBe("backup.zip");
    const options = fetchMock.mock.calls[1][1];
    expect(options.credentials).toBe("omit");
    expect(options.headers.get("Authorization")).toBe("Bearer download-access");
    expect(options.headers.get("X-Auth-Transport")).toBe("bearer");
  });
});
