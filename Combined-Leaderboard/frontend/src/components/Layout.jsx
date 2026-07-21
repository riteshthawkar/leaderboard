import { useEffect, useRef, useState } from "react";
import {
  Link,
  NavLink,
  Outlet,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  ClipboardList,
  LogOut,
  Menu,
  Moon,
  ShieldCheck,
  Sun,
  TriangleAlert,
  UserRound,
  X,
} from "lucide-react";
import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  errorMessage,
  fetchMe,
  getJSON,
  IS_STATIC_DEMO,
  logout,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const nav = [
  ["/", "Overview", "home"],
  ["/leaderboard", "Leaderboard", "leaderboard"],
  ["/benchmarks/do-you-see-me", "Do You See Me", "dysm"],
  ["/benchmarks/minds-eye", "Mind's Eye", "minds_eye"],
  ["/benchmarks/spatial", "Spatial Reasoning", "spatial"],
];

const navAccentClass = {
  dysm: "!text-dysm",
  minds_eye: "!text-me",
  spatial: "!text-spatial",
};

const privacyPolicyUrl = (import.meta.env.VITE_PRIVACY_POLICY_URL || "/privacy").trim();

function pageId(pathname) {
  if (pathname === "/") return "home";
  if (pathname.includes("do-you-see-me")) return "dysm";
  if (pathname.includes("minds-eye")) return "minds_eye";
  if (pathname.includes("spatial")) return "spatial";
  if (pathname.includes("leaderboard")) return "leaderboard";
  if (pathname.includes("submissions")) return "submissions";
  if (pathname.includes("profile")) return "profile";
  if (pathname.includes("admin")) return "admin";
  if (pathname.includes("submit")) return "submit";
  if (pathname.includes("login")) return "login";
  return "";
}

function healthIssueMessage(payload) {
  const components = payload?.components || {};
  const issues = [];
  if (components.database === "unhealthy") {
    issues.push(
      "Database integrity, WAL mode, or persistent storage failed its readiness check.",
    );
  }
  if (components.submission_store === "unhealthy") {
    issues.push(
      "Stored submission records failed a model, score, or answer-count integrity check.",
    );
  }
  if (components.leaderboard_store === "unhealthy") {
    issues.push("Published leaderboard results are unavailable or out of sync with stored scores.");
  }
  if (["unhealthy", "overdue", "disabled"].includes(components.backup)) {
    issues.push("Automatic database backups are unavailable or overdue.");
  }
  if (components.auth === "unhealthy") {
    issues.push("Authentication configuration is incomplete.");
  }
  if (components.email === "unhealthy") {
    issues.push("Verification and password-reset email delivery is not ready.");
  }
  if (components.deployment === "unhealthy") {
    issues.push(
      "Frontend, API, cookie, or CORS production settings are invalid.",
    );
  }
  if (components.ground_truth === "unhealthy") {
    issues.push(
      "One or more enabled benchmark answer sets could not be loaded.",
    );
  }
  return issues.slice(0, 2).join(" ");
}

export function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const accountMenuRef = useRef(null);
  const accountTriggerRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [accountError, setAccountError] = useState("");
  const [loggingOut, setLoggingOut] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("vci-theme") || "dark";
    } catch {
      return "dark";
    }
  });
  const [serviceStatus, setServiceStatus] = useState("checking");
  const [serviceIssue, setServiceIssue] = useState("");
  const [dismissedServiceWarning, setDismissedServiceWarning] = useState("");
  const [pageWarning, setPageWarning] = useState(null);
  const [authUser, setAuthUser] = useState(undefined);

  useEffect(() => {
    document.body.setAttribute("data-page", pageId(location.pathname));
    setOpen(false);
    setAccountMenuOpen(false);
    setAccountError("");
    setPageWarning(null);
  }, [location.pathname]);

  useEffect(() => {
    const handlePageWarning = (event) => setPageWarning(event.detail || null);
    window.addEventListener("app-warning", handlePageWarning);
    return () => window.removeEventListener("app-warning", handlePageWarning);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("vci-theme", theme);
    } catch {
      /* Browser storage is optional. */
    }
    window.dispatchEvent(new Event("themechange"));
  }, [theme]);

  useEffect(() => {
    if (IS_STATIC_DEMO) {
      setServiceStatus("snapshot");
      return;
    }
    let live = true;
    const check = () =>
      getJSON("/api/health")
        .then((data) => {
          if (!live) return;
          const issue =
            data?.status === "healthy"
              ? ""
              : healthIssueMessage(data) ||
              (!data?.components
                ? "Service readiness could not be determined."
                : "");
          setServiceStatus(issue ? "degraded" : "online");
          setServiceIssue(issue);
        })
        .catch((error) => {
          if (!live) return;
          const issue = error?.status
            ? healthIssueMessage(error.data) ||
            (!error.data?.components
              ? "Service readiness could not be determined."
              : "")
            : "";
          setServiceStatus(
            error?.status ? (issue ? "degraded" : "online") : "offline",
          );
          setServiceIssue(issue);
        });
    check();
    const id = setInterval(check, 30000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (serviceStatus === "online") setDismissedServiceWarning("");
  }, [serviceStatus]);

  useEffect(() => {
    if (IS_STATIC_DEMO) {
      setAuthUser(null);
      return undefined;
    }
    let live = true;
    const refresh = () =>
      fetchMe()
        .then((user) => {
          if (live) setAuthUser(user);
        })
        .catch(() => {
          /* Keep the last known header state during a transient outage. */
        });
    refresh();
    window.addEventListener("lb_auth", refresh);
    return () => {
      live = false;
      window.removeEventListener("lb_auth", refresh);
    };
  }, []);

  useEffect(() => {
    if (!accountMenuOpen) return undefined;
    const firstItem =
      accountMenuRef.current?.querySelector("[role='menuitem']");
    firstItem?.focus();

    const onPointerDown = (event) => {
      if (!accountMenuRef.current?.contains(event.target))
        setAccountMenuOpen(false);
    };
    const onKeyDown = (event) => {
      if (event.key !== "Escape") return;
      setAccountMenuOpen(false);
      accountTriggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [accountMenuOpen]);

  const handleLogout = async () => {
    setLoggingOut(true);
    setAccountError("");
    try {
      await logout();
      setAuthUser(null);
      setAccountMenuOpen(false);
      navigate("/");
    } catch (error) {
      setAccountError(
        errorMessage(
          error,
          "You could not be signed out. Your session is still active; try again.",
        ),
      );
    } finally {
      setLoggingOut(false);
    }
  };

  const serviceWarning =
    serviceStatus === "offline"
      ? {
        message:
          "The leaderboard service is unavailable. Cached content may still work, but signing in, submissions, and downloads are unavailable until the connection recovers.",
        tone: "negative",
      }
      : serviceStatus === "degraded"
        ? {
          message:
            serviceIssue ||
            "The leaderboard service is degraded. Some account, scoring, or download actions may be temporarily unavailable.",
          tone: "warning",
        }
        : null;
  const visibleServiceWarning =
    serviceWarning && dismissedServiceWarning !== serviceStatus;
  const activeWarning = visibleServiceWarning ? serviceWarning : pageWarning;
  const activeWarningSource = visibleServiceWarning ? "service" : "page";

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="sticky top-0 z-50 h-[var(--header-h)] border-b border-border bg-background">
        <div className="container grid h-full grid-cols-[1fr_auto] items-center gap-4 border-x border-solid border-border-strong min-[861px]:grid-cols-[1fr_auto_1fr]">
          <Link
            to="/"
            className="inline-flex items-center font-display font-bold text-foreground no-underline"
            aria-label="Home"
          >
            <span className="flex flex-col leading-none">
              <strong className="text-xl font-bold">MS VISTA</strong>
            </span>
          </Link>
          <nav
            className={cn(
              "invisible fixed inset-x-0 top-[var(--header-h)] flex -translate-y-[calc(100%+var(--header-h)+0.75rem)] flex-col items-stretch gap-0.5 border-b border-border bg-background px-4 py-3 shadow-lg transition-[transform,visibility] min-[861px]:visible min-[861px]:static min-[861px]:translate-y-0 min-[861px]:flex-row min-[861px]:items-center min-[861px]:justify-center min-[861px]:border-0 min-[861px]:bg-transparent min-[861px]:p-0 min-[861px]:shadow-none",
              open && "visible translate-y-0",
            )}
            id="nav_menu"
            aria-label="Primary"
          >
            {nav.map(([to, label, navId]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  cn(
                    "group relative px-3 py-2 text-sm font-medium text-muted transition-colors hover:text-foreground",
                    isActive && "text-brand-strong",
                    isActive && navAccentClass[navId],
                  )
                }
              >
                {({ isActive }) => (
                  <span
                    className={cn(
                      "relative inline-block after:absolute after:inset-x-0 after:-bottom-1 after:h-px after:origin-left after:scale-x-0 after:bg-current after:transition-transform group-hover:after:scale-x-100 group-focus-visible:after:scale-x-100",
                      isActive && "after:scale-x-100",
                    )}
                  >
                    {label}
                  </span>
                )}
              </NavLink>
            ))}
            <NavLink
              to="/submit"
              className={({ isActive }) =>
                cn(
                  "relative px-3 py-2 text-sm font-medium text-muted min-[861px]:hidden",
                  isActive && "text-brand-strong",
                )
              }
            >
              Submit a model
            </NavLink>
            {authUser === null && (
              <NavLink
                to="/login"
                className={({ isActive }) =>
                  cn(
                    "relative px-3 py-2 text-sm font-medium text-muted min-[861px]:hidden",
                    isActive && "text-brand-strong",
                  )
                }
              >
                Sign In
              </NavLink>
            )}
          </nav>
          <div className="flex items-center justify-end gap-2.5">
            <Button
              asChild
              variant="brand"
              size="sm"
              className="max-[860px]:hidden"
            >
              <Link to="/submit">Submit a model</Link>
            </Button>
            {authUser === null && (
              <Button
                asChild
                variant="ghost"
                size="sm"
                className="max-[860px]:hidden border"
              >
                <Link to="/login">Sign In</Link>
              </Button>
            )}
            {authUser && (
              <div className="relative" ref={accountMenuRef}>
                <button
                  ref={accountTriggerRef}
                  className={cn(
                    "grid size-10 cursor-pointer place-items-center border border-border-strong bg-surface text-foreground transition-colors hover:bg-surface-subtle focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand",
                    (accountMenuOpen ||
                      location.pathname === "/profile" ||
                      location.pathname === "/submissions") &&
                    "border-foreground bg-surface-subtle",
                  )}
                  type="button"
                  aria-label="Account menu"
                  aria-haspopup="menu"
                  aria-expanded={accountMenuOpen}
                  title="Account menu"
                  onClick={() => {
                    setAccountError("");
                    setAccountMenuOpen((value) => !value);
                  }}
                >
                  <UserRound size={19} />
                </button>
                {accountMenuOpen && (
                  <div
                    className="fixed inset-x-4 top-[calc(var(--header-h)+0.5rem)] z-[80] border border-border-strong bg-surface p-1 shadow-lg min-[601px]:absolute min-[601px]:inset-x-auto min-[601px]:right-0 min-[601px]:top-[calc(100%+0.625rem)] min-[601px]:w-[248px]"
                    role="menu"
                    aria-label="Account"
                  >
                    <div
                      className="truncate border-b border-border px-3 py-2.5 text-xs text-muted"
                      title={authUser.email}
                    >
                      {authUser.email}
                    </div>
                    <Link
                      className="flex min-h-10 w-full items-center gap-2.5 px-2.5 py-2 text-sm text-foreground hover:bg-surface-subtle focus-visible:bg-surface-subtle focus-visible:outline-none"
                      role="menuitem"
                      to="/submissions"
                    >
                      <ClipboardList size={17} />
                      <span>Submissions</span>
                    </Link>
                    <Link
                      className="flex min-h-10 w-full items-center gap-2.5 px-2.5 py-2 text-sm text-foreground hover:bg-surface-subtle focus-visible:bg-surface-subtle focus-visible:outline-none"
                      role="menuitem"
                      to="/profile"
                    >
                      <UserRound size={17} />
                      <span>Profile</span>
                    </Link>
                    {authUser.isAdmin && (
                      <Link
                        className={cn(
                          "flex min-h-10 w-full items-center gap-2.5 px-2.5 py-2 text-sm text-foreground hover:bg-surface-subtle focus-visible:bg-surface-subtle focus-visible:outline-none",
                          location.pathname === "/admin" && "bg-surface-subtle",
                        )}
                        role="menuitem"
                        to="/admin"
                        aria-current={location.pathname === "/admin" ? "page" : undefined}
                      >
                        <ShieldCheck size={17} />
                        <span>Admin</span>
                      </Link>
                    )}
                    <button
                      className="flex min-h-10 w-full items-center gap-2.5 border-0 border-t border-border bg-transparent px-2.5 py-2 text-left text-sm text-foreground hover:bg-surface-subtle focus-visible:bg-surface-subtle focus-visible:outline-none disabled:cursor-wait disabled:text-faint"
                      role="menuitem"
                      type="button"
                      disabled={loggingOut}
                      onClick={handleLogout}
                    >
                      <LogOut size={17} />
                      <span>{loggingOut ? "Logging out..." : "Logout"}</span>
                    </button>
                    {accountError && (
                      <div
                        className="m-1 border-l-[3px] border-negative bg-negative-soft p-2 text-xs leading-relaxed text-negative"
                        role="alert"
                      >
                        {accountError}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            <button
              className="grid size-10 cursor-pointer place-items-center border border-border-strong bg-surface text-foreground transition-colors hover:bg-surface-subtle min-[861px]:hidden"
              type="button"
              aria-label="Menu"
              aria-controls="nav_menu"
              aria-expanded={open}
              onClick={() => setOpen((value) => !value)}
            >
              {open ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </div>
      </header>

      {activeWarning && (
        <div
          className={cn(
            "fixed left-1/2 top-[calc(var(--header-h)+0.75rem)] z-[45] flex max-h-[calc(100dvh-var(--header-h)-1.5rem)] w-[calc(100%-2rem)] max-w-[720px] -translate-x-1/2 items-start gap-3 overflow-y-auto border border-l-[3px] bg-background px-4 py-3 text-sm leading-relaxed shadow-lg max-sm:gap-2 max-sm:px-3 max-sm:py-2.5 max-sm:text-xs",
            activeWarning.tone === "negative"
              ? "border-negative text-negative"
              : "border-warning text-warning",
          )}
          role="alert"
          aria-atomic="true"
        >
          <TriangleAlert
            className="mt-0.5 size-4 shrink-0"
            aria-hidden="true"
          />
          <span className="min-w-0 flex-1">{activeWarning.message}</span>
          {activeWarning.action && (
            <button
              className="min-h-8 shrink-0 border border-current bg-transparent px-3 text-xs font-semibold transition-colors hover:bg-surface"
              type="button"
              onClick={activeWarning.action}
            >
              {activeWarning.actionLabel || "Retry"}
            </button>
          )}
          <button
            className="grid size-8 shrink-0 cursor-pointer place-items-center border border-current bg-transparent transition-colors hover:bg-surface focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
            type="button"
            aria-label="Dismiss service warning"
            title="Dismiss"
            onClick={() => {
              if (activeWarningSource === "service")
                setDismissedServiceWarning(serviceStatus);
              else setPageWarning(null);
            }}
          >
            <X size={15} aria-hidden="true" />
          </button>
        </div>
      )}

      <main className="relative flex-1 overflow-x-clip">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 left-1/2 z-20 hidden w-full max-w-content -translate-x-1/2 border-x border-solid border-border-strong md:block"
        />
        <Outlet />
      </main>

      <footer className="border-t border-border bg-background">
        <div className="container border-x border-solid border-border-strong bg-[color-mix(in_srgb,var(--surface-2)_18%,var(--bg))] pt-10">
          <div className="flex flex-wrap justify-between gap-6 max-sm:flex-col">
            <div className="max-w-[320px]">
              <Link
                to="/"
                className="mb-3 inline-flex items-center font-display font-bold text-foreground"
              >
                <span>
                  <strong>MS VISTA</strong>
                </span>
              </Link>
              <p className="text-sm leading-relaxed text-muted">
                MS VISTA is a unified evaluation faithful to each paper for
                multimodal LLM visual perception, visual cognition, and spatial
                reasoning across three Microsoft Research benchmarks.
              </p>
            </div>
            <div className="flex flex-col gap-1">
              <h5 className="mb-1 text-xs font-semibold uppercase text-faint">
                Benchmarks
              </h5>
              <Link
                className="text-sm text-muted hover:text-brand-strong"
                to="/benchmarks/do-you-see-me"
              >
                Do You See Me
              </Link>
              <Link
                className="text-sm text-muted hover:text-brand-strong"
                to="/benchmarks/minds-eye"
              >
                Mind's Eye
              </Link>
              <Link
                className="text-sm text-muted hover:text-brand-strong"
                to="/benchmarks/spatial"
              >
                Spatial Reasoning
              </Link>
            </div>
            <div className="flex flex-col gap-1">
              <h5 className="mb-1 text-xs font-semibold uppercase text-faint">
                Explore
              </h5>
              <Link
                className="text-sm text-muted hover:text-brand-strong"
                to="/leaderboard"
              >
                Leaderboard
              </Link>
              <Link
                className="text-sm text-muted hover:text-brand-strong"
                to="/submit"
              >
                Submit a model
              </Link>
              <a
                className="text-sm text-muted hover:text-brand-strong"
                href="/#findings"
              >
                Key findings
              </a>
            </div>
            <div className="flex flex-col gap-1">
              <h5 className="mb-1 text-xs font-semibold uppercase text-faint">
                Papers
              </h5>
              <a
                className="text-sm text-muted hover:text-brand-strong flex items-center gap-1"
                href="https://arxiv.org/abs/2506.02022"
                target="_blank"
                rel="noreferrer"
              >
                Do You See Me <ArrowUpRight className="size-3.5" strokeWidth={1.75} aria-hidden="true" />
              </a>
              <a
                className="text-sm text-muted hover:text-brand-strong flex items-center gap-1"
                href="https://arxiv.org/abs/2604.16054"
                target="_blank"
                rel="noreferrer"
              >
                Mind's Eye <ArrowUpRight className="size-3.5" strokeWidth={1.75} aria-hidden="true" />

              </a>
              <a
                className="text-sm text-muted hover:text-brand-strong flex items-center gap-1"
                href="https://arxiv.org/abs/2604.16060"
                target="_blank"
                rel="noreferrer"
              >
                CoT degrades spatial reasoning <ArrowUpRight className="size-3.5" strokeWidth={1.75} aria-hidden="true" />
              </a>
            </div>
          </div>
          <div className="w-full mt-7 flex flex-wrap items-center justify-between gap-3 border-t border-border py-4 text-xs text-faint max-sm:flex-col max-sm:items-start">
            <span>
              Benchmarks © Microsoft Research. Leaderboard for noncommercial
              research use.
            </span>
            <div className="flex flex-wrap items-center gap-2.5">
              {privacyPolicyUrl && (
                <a
                  className="px-2 text-xs font-medium text-muted hover:text-foreground"
                  href={privacyPolicyUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  Privacy
                </a>
              )}
              <span
                className="inline-flex h-10 items-center gap-2 border border-solid border-border-strong bg-surface px-3 text-xs font-semibold text-muted"
                title={IS_STATIC_DEMO ? "Static data status" : "Backend status"}
              >
                <span
                  className={cn(
                    "size-2 bg-red-500 shadow-[0_0_0_3px_rgba(239,68,68,0.16)]",
                    serviceStatus === "online" &&
                    "bg-green-500 shadow-[0_0_0_3px_rgba(34,197,94,0.18)]",
                    serviceStatus === "snapshot" && "bg-faint shadow-none",
                  )}
                />
                <span>
                  {serviceStatus === "online"
                    ? "Online"
                    : serviceStatus === "snapshot"
                      ? "Snapshot"
                    : serviceStatus === "degraded"
                      ? "Degraded"
                      : serviceStatus === "checking"
                        ? "Checking"
                        : "Offline"}
                </span>
              </span>
              <button
                className="grid size-10 cursor-pointer place-items-center border border-border-strong bg-surface text-foreground transition-colors hover:bg-surface-subtle"
                id="theme_toggle"
                type="button"
                aria-label={
                  theme === "dark"
                    ? "Switch to light theme"
                    : "Switch to dark theme"
                }
                title={
                  theme === "dark"
                    ? "Switch to light theme"
                    : "Switch to dark theme"
                }
                onClick={() =>
                  setTheme((value) => (value === "dark" ? "light" : "dark"))
                }
              >
                {theme === "dark" ? (
                  <Sun size={16} aria-hidden="true" />
                ) : (
                  <Moon size={16} aria-hidden="true" />
                )}
              </button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
