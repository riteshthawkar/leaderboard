/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--bg)",
        foreground: "var(--text)",
        muted: "var(--text-muted)",
        border: "var(--border)",
        surface: "var(--surface)",
        ring: "var(--ring)",
        card: "var(--card)",
        "card-foreground": "var(--card-foreground)",
        "chart-label": "var(--chart-label)",
        "chart-foreground": "var(--chart-foreground)",
        "chart-foreground-muted": "var(--chart-foreground-muted)",
      },
      fontFamily: {
        sans: ["var(--font)"],
        mono: ["var(--mono)"],
        display: ["var(--display)"],
      },
      borderRadius: {
        DEFAULT: "var(--radius)",
      },
    },
  },
  plugins: [],
};