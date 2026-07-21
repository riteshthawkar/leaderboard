import { useRef } from "react";
import { cn } from "@/lib/utils";

export function TabBar({ tabs, active, onChange, className }) {
  const tabRefs = useRef([]);

  function moveFocus(event, index) {
    const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
    if (!keys.includes(event.key) || !tabs.length) return;
    event.preventDefault();
    let nextIndex = index;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
    tabRefs.current[nextIndex]?.focus();
    onChange(tabs[nextIndex].id);
  }

  return (
    <div className={cn("inline-flex flex-wrap border border-border bg-surface p-1.5", className)} role="tablist" aria-label="Tracks">
      {tabs.map((tab, index) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          tabIndex={active === tab.id ? 0 : -1}
          ref={(node) => { tabRefs.current[index] = node; }}
          className={cn(
            "min-h-10 border-0 border-r border-border bg-transparent px-5 py-2.5 text-sm font-semibold text-muted transition-colors last:border-r-0 hover:bg-surface-subtle hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand",
            active === tab.id && "bg-invert-bg text-invert-text hover:bg-invert-bg-hover hover:text-invert-text",
          )}
          onClick={() => onChange(tab.id)}
          onKeyDown={(event) => moveFocus(event, index)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
