import { snapshots } from "@/data/snapshot";

// When built with VITE_STATIC=1, the app runs with no backend: API reads are
// served from a frozen snapshot and write actions (submit / sign-in) are disabled.
export const IS_STATIC_DEMO = import.meta.env.VITE_STATIC === "1";
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");
const CSRF_STORAGE_KEY = "lb_csrf_token";
const READ_TIMEOUT_MS = 20_000;
const WRITE_TIMEOUT_MS = 30_000;
const UPLOAD_TIMEOUT_MS = 180_000;

function pathOnly(url) {
  const value = String(url || "/");
  if (!/^https?:\/\//i.test(value)) return value.split("?")[0];
  try {
    const parsed = new URL(value);
    return parsed.pathname;
  } catch {
    return value;
  }
}

export function apiUrl(path) {
  const value = String(path || "");
  if (/^https?:\/\//i.test(value)) return value;
  const normalized = value.startsWith("/") ? value : `/${value}`;
  return API_BASE_URL ? `${API_BASE_URL}${normalized}` : normalized;
}

export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "ApiError";
    this.status = options.status || 0;
    this.code = options.code || "request_failed";
    this.data = options.data || {};
    this.fieldErrors = options.fieldErrors || {};
    this.requestId = options.requestId || "";
    this.retryable = Boolean(options.retryable);
  }
}

function defaultStatusMessage(status) {
  if (status === 400) return "The server rejected this request. Check the entered values and try again.";
  if (status === 401) return "Your sign-in session is missing or has expired. Sign in again and retry.";
  if (status === 403) return "Your account does not have permission to perform this action.";
  if (status === 404) return "The requested resource could not be found. Refresh the page and try again.";
  if (status === 409) return "This action conflicts with data that already exists. Refresh and try again.";
  if (status === 413) return "The uploaded file is larger than the server allows.";
  if (status === 429) return "Too many requests were made. Wait a moment before trying again.";
  if (status >= 500) return "The service could not complete this request. Try again shortly.";
  return `The request failed with HTTP status ${status}.`;
}

async function parseResponse(response) {
  const text = await response.text();
  if (!text) return { data: {}, invalid: false };
  try {
    return { data: JSON.parse(text), invalid: false };
  } catch {
    return { data: {}, invalid: true };
  }
}

function httpError(response, data, invalidBody) {
  const message = invalidBody
    ? `The server returned an unreadable response (HTTP ${response.status}). Try again or contact the administrator.`
    : data?.error || data?.message || defaultStatusMessage(response.status);
  return new ApiError(message, {
    status: response.status,
    code: data?.code || (invalidBody ? "invalid_server_response" : `http_${response.status}`),
    data,
    fieldErrors: data?.field_errors,
    requestId: data?.request_id || response.headers.get("X-Request-Id") || "",
    retryable: data?.retryable ?? response.status >= 500,
  });
}

async function requestJSON(url, options = {}, timeoutMs = READ_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(apiUrl(url), {
      credentials: "include",
      ...options,
      signal: controller.signal,
    });
    const { data, invalid } = await parseResponse(response);
    if (!response.ok) throw httpError(response, data, invalid);
    if (invalid) {
      throw new ApiError(
        "The server returned an unreadable response. Refresh the page and retry; contact the administrator if it continues.",
        { status: response.status, code: "invalid_server_response", retryable: true },
      );
    }
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error?.name === "AbortError") {
      throw new ApiError(
        "The server took too long to respond. Check your connection and try again; large submissions can take up to three minutes.",
        { code: "request_timeout", retryable: true },
      );
    }
    const offline = typeof navigator !== "undefined" && navigator.onLine === false;
    throw new ApiError(
      offline
        ? "You appear to be offline. Reconnect to the internet, then try again."
        : "The application could not reach the server. Check your connection and retry; if other pages also fail, the service may be unavailable.",
      { code: offline ? "offline" : "network_error", retryable: true },
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

function readCsrfToken() {
  try {
    return localStorage.getItem(CSRF_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

function saveCsrfToken(token) {
  try {
    if (token) localStorage.setItem(CSRF_STORAGE_KEY, token);
  } catch {
    /* The in-memory session still works; CSRF recovery will refresh the token. */
  }
}

function clearCsrfToken() {
  try {
    localStorage.removeItem(CSRF_STORAGE_KEY);
  } catch {
    /* Storage can be disabled by browser privacy settings. */
  }
}

function csrfHeaders() {
  const token = readCsrfToken();
  return token ? { "X-CSRF-Token": token } : {};
}

export async function getJSON(url) {
  if (IS_STATIC_DEMO) {
    const key = pathOnly(url);
    if (Object.prototype.hasOwnProperty.call(snapshots, key)) {
      return structuredClone(snapshots[key]);
    }
    throw new Error(`Static demo: no snapshot for ${key}`);
  }
  return requestJSON(url);
}

async function refreshCsrfToken() {
  const data = await requestJSON("/api/auth/me");
  if (!data?.authenticated || !data?.csrf_token) {
    clearCsrfToken();
    return false;
  }
  saveCsrfToken(data.csrf_token);
  return true;
}

async function postJSONAttempt(url, body) {
  return requestJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...csrfHeaders() },
    body: JSON.stringify(body),
  }, WRITE_TIMEOUT_MS);
}

export async function postJSON(url, body) {
  if (IS_STATIC_DEMO) {
    throw new ApiError("This is a static demo; submissions and sign-in are disabled.", { code: "static_demo" });
  }
  try {
    return await postJSONAttempt(url, body);
  } catch (error) {
    if (error.code !== "csrf_required" || !(await refreshCsrfToken())) throw error;
    return postJSONAttempt(url, body);
  }
}

async function postFormDataAttempt(url, formData) {
  return requestJSON(url, {
    method: "POST",
    headers: csrfHeaders(),
    body: formData,
  }, UPLOAD_TIMEOUT_MS);
}

export async function postFormData(url, formData) {
  if (IS_STATIC_DEMO) {
    throw new ApiError("This is a static demo; submissions are disabled.", { code: "static_demo" });
  }
  try {
    return await postFormDataAttempt(url, formData);
  } catch (error) {
    if (error.code !== "csrf_required" || !(await refreshCsrfToken())) throw error;
    return postFormDataAttempt(url, formData);
  }
}

export function errorMessage(error, fallback = "The action could not be completed.") {
  const base = String(error?.message || fallback).trim() || fallback;
  if (error?.requestId && (error?.status >= 500 || error?.retryable)) {
    return `${base} Request reference: ${error.requestId}.`;
  }
  return base;
}

export function isConnectionError(error) {
  return ["offline", "network_error", "request_timeout"].includes(error?.code);
}

export async function downloadFile(url, fallbackName = "download", options = {}) {
  if (IS_STATIC_DEMO) {
    throw new ApiError("Downloads requiring the live service are disabled in the static demo.", { code: "static_demo" });
  }
  const method = String(options.method || "GET").toUpperCase();
  if (!["GET", "POST"].includes(method)) {
    throw new ApiError("This download request uses an unsupported HTTP method.", { code: "invalid_download_method" });
  }
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 60_000);
  try {
    const attempt = async () => {
      const response = await fetch(apiUrl(url), {
        method,
        credentials: "include",
        headers: method === "POST" ? csrfHeaders() : undefined,
        signal: controller.signal,
      });
      if (!response.ok) {
        const { data, invalid } = await parseResponse(response);
        throw httpError(response, data, invalid);
      }
      return response;
    };
    let response;
    try {
      response = await attempt();
    } catch (error) {
      if (method !== "POST" || error.code !== "csrf_required" || !(await refreshCsrfToken())) {
        throw error;
      }
      response = await attempt();
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
    const quoted = disposition.match(/filename="([^"]+)"/i)?.[1];
    const unquoted = disposition.match(/filename=([^;\s]+)/i)?.[1];
    let filename = quoted || unquoted || fallbackName;
    if (encoded) {
      try {
        filename = decodeURIComponent(encoded);
      } catch {
        filename = fallbackName;
      }
    }
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(objectUrl);
    return filename;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error?.name === "AbortError") {
      throw new ApiError("The download timed out. Check your connection and try again.", { code: "request_timeout", retryable: true });
    }
    const offline = typeof navigator !== "undefined" && navigator.onLine === false;
    throw new ApiError(
      offline ? "You are offline. Reconnect before downloading this file." : "The download could not reach the server. Check your connection and try again.",
      { code: offline ? "offline" : "network_error", retryable: true },
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export function readUser() {
  try {
    return JSON.parse(localStorage.getItem("lb_user") || "null");
  } catch {
    return null;
  }
}

export function saveUser(user) {
  if (user?.csrfToken) saveCsrfToken(user.csrfToken);
  try {
    localStorage.setItem("lb_user", JSON.stringify(user));
  } catch {
    /* Session state remains authoritative and is reloaded from /api/auth/me. */
  }
  window.dispatchEvent(new CustomEvent("lb_auth", { detail: user }));
}

export function clearUser() {
  try {
    localStorage.removeItem("lb_user");
  } catch {
    /* Storage can be disabled by browser privacy settings. */
  }
  clearCsrfToken();
  window.dispatchEvent(new CustomEvent("lb_auth", { detail: null }));
}

// Ask the backend who is signed in (via the session cookie), or who is acting in
// explicit test-deployment mode.
export async function fetchMe() {
  if (IS_STATIC_DEMO) return null;
  const data = await getJSON("/api/auth/me");
  if (!data || !data.authenticated) {
    clearCsrfToken();
    return null;
  }
  saveCsrfToken(data.csrf_token);
  return {
    email: data.email,
    quota: data.quota,
    authDisabled: Boolean(data.auth_disabled),
    isAdmin: Boolean(data.is_admin),
    emailVerified: Boolean(data.email_verified),
    provider: data.auth_provider || "password",
    createdAt: data.created_at || null,
    csrfToken: data.csrf_token,
  };
}

export async function logout() {
  if (!IS_STATIC_DEMO) await postJSON("/api/auth/logout", {});
  clearUser();
}
