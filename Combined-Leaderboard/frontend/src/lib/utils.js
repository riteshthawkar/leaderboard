import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function fmtPct(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function fmtVci(value) {
  return value == null ? "—" : (value * 100).toFixed(1);
}

export function fmtDelta(value) {
  if (value == null) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}`;
}

export function fmtMeanStd(mean, std) {
  if (mean == null) return "—";
  let text = (mean * 100).toFixed(1);
  if (std != null) text += ` ± ${(std * 100).toFixed(1)}`;
  return `${text}%`;
}

export function prettyLabel(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function modelType(meta) {
  if (!meta) return "—";
  const value = meta.type || meta.access || meta.org || meta.family;
  return value ? prettyLabel(value) : "—";
}

export function safeNext(value, fallback = "/submit") {
  return /^\/[^/]/.test(value || "") ? value : fallback;
}
