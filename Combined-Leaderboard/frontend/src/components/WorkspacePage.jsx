import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ui } from "@/lib/styles";

const accountLinks = [
  { to: "/submit", label: "Submit" },
  { to: "/submissions", label: "Submissions" },
  { to: "/profile", label: "Profile" },
];

export function AccountNavigation({ includeAdmin = false }) {
  const links = includeAdmin
    ? [...accountLinks, { to: "/admin", label: "Admin" }]
    : accountLinks;

  return (
    <nav className="overflow-x-auto border-b border-border-strong px-6 py-4 lg:px-8" aria-label="Account pages">
      <div className="inline-flex min-w-max border border-border bg-surface p-1">
        {links.map((link) => (
          <NavLink
            className={({ isActive }) => cn(
              "inline-flex min-h-10 items-center border-r border-border px-4 py-2 text-sm font-medium text-muted last:border-r-0 hover:bg-surface-subtle hover:text-foreground",
              isActive && "bg-invert-bg text-invert-text hover:bg-invert-bg-hover hover:text-invert-text",
            )}
            key={link.to}
            to={link.to}
          >
            {link.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}

export function WorkspacePage({
  eyebrow,
  title,
  description,
  accountNavigation = false,
  includeAdmin = false,
  children,
}) {
  return (
    <section>
      <div className={ui.sectionFrame}>
        <header className={ui.sectionBand}>
          <div className="w-full">
            <div className={ui.sectionTag}>{eyebrow}</div>
            <h1 className={ui.heading1}>{title}</h1>
            {description && <p className={cn(ui.lede, "mt-4 w-full")}>{description}</p>}
          </div>
        </header>
        {accountNavigation && <AccountNavigation includeAdmin={includeAdmin} />}
        {children}
      </div>
    </section>
  );
}
