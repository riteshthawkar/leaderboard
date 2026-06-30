import { cn } from "@/lib/utils";

export function TabBar({ tabs, active, onChange, className }) {
  return (
    <div className={cn("tabbar", className)} role="tablist" aria-label="Tracks">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          className={cn("tab-btn", active === tab.id && "is-active")}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}