import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function fmtPct(value) {
  return value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;
}

export function fmtVci(value) {
  return value == null ? "N/A" : (value * 100).toFixed(1);
}

export function fmtDelta(value) {
  if (value == null) return "N/A";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}`;
}

export function prettyLabel(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function modelType(meta) {
  if (!meta) return "N/A";
  const value = meta.type || meta.access;
  return value ? prettyLabel(value) : "N/A";
}

export function safeNext(value, fallback = "/submit") {
  const candidate = String(value || "");
  const hasControlCharacter = Array.from(candidate).some((character) => {
    const code = character.charCodeAt(0);
    return code <= 31 || code === 127;
  });
  return /^\/[^/]/.test(candidate) && !candidate.includes("\\") && !hasControlCharacter
    ? candidate
    : fallback;
}
