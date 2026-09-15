export function buildContentSecurityPolicy(apiOrigin = "") {
  const connections = ["'self'"];
  if (apiOrigin) {
    const parsed = new URL(apiOrigin);
    if (!["http:", "https:"].includes(parsed.protocol) || parsed.origin !== apiOrigin.replace(/\/$/, "")) {
      throw new Error("The CSP API endpoint must be an HTTP(S) origin.");
    }
    connections.push(parsed.origin);
  }
  return [
    "default-src 'self'", "base-uri 'self'", "object-src 'none'",
    "script-src 'self'", "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:", "font-src 'self' data:",
    `connect-src ${connections.join(" ")}`, "form-action 'self'",
    "manifest-src 'self'", "worker-src 'none'",
  ].join("; ");
}
