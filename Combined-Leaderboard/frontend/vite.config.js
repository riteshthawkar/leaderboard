import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { buildContentSecurityPolicy } from "./build-security.js";

function isHttpOrigin(value) {
  try {
    const parsed = new URL(value);
    return (
      ["http:", "https:"].includes(parsed.protocol) &&
      parsed.origin === value.replace(/\/$/, "")
    );
  } catch {
    return false;
  }
}

function normalizeBasePath(value) {
  const candidate = (value || "/").trim();
  if (
    !candidate.startsWith("/")
    || !candidate.endsWith("/")
    || candidate.includes("//")
    || /[?#\\]/.test(candidate)
  ) {
    throw new Error("VITE_BASE_PATH must start and end with one slash, for example /leaderboard/.");
  }
  return candidate;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const isStatic = (process.env.VITE_STATIC || env.VITE_STATIC) === "1";
  const isSameOrigin = (process.env.VITE_SAME_ORIGIN || env.VITE_SAME_ORIGIN) === "1";
  const isTest = mode === "test";
  const apiBaseUrl = (process.env.VITE_API_BASE_URL || env.VITE_API_BASE_URL || "").trim();
  const basePath = normalizeBasePath(process.env.VITE_BASE_PATH || env.VITE_BASE_PATH || "/");
  if (!isStatic && !isSameOrigin && !isTest && !isHttpOrigin(apiBaseUrl)) {
    throw new Error(
      "VITE_API_BASE_URL must be an absolute HTTP(S) backend origin unless VITE_SAME_ORIGIN=1.",
    );
  }
  return {
    base: basePath,
    plugins: [react(), {
      name: "production-csp",
      apply: "build",
      transformIndexHtml() {
        return [{
          tag: "meta",
          attrs: {
            "http-equiv": "Content-Security-Policy",
            content: buildContentSecurityPolicy(isSameOrigin || isStatic ? "" : apiBaseUrl),
          },
          injectTo: "head-prepend",
        }];
      },
    }],
    build: {
      outDir: isStatic ? "dist-static" : "dist",
      emptyOutDir: true,
      chunkSizeWarningLimit: 700,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes("node_modules")) return undefined;
            if (/[\\/]node_modules[\\/](@visx|d3-)/.test(id)) return "charts";
            if (/[\\/]node_modules[\\/]motion/.test(id)) return "motion";
            if (/[\\/]node_modules[\\/]react-router/.test(id)) return "router";
            if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return "react";
            return "vendor";
          },
        },
      },
    },
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },
    server: { port: 5173, strictPort: true },
    test: {
      environment: "jsdom",
      setupFiles: "./tests/setup.js",
      clearMocks: true,
      restoreMocks: true,
    },
  };
});
