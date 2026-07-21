import { useEffect, useState } from "react";
import { ClipboardList, Send, UserRound } from "lucide-react";
import { Link } from "react-router-dom";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";
import { errorMessage, fetchMe } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ui } from "@/lib/styles";

function formatProvider(provider) {
  const labels = {
    password: "Email and password",
    google: "Google",
    microsoft: "Microsoft",
    development: "Development access",
  };
  return labels[provider] || "External identity provider";
}

function formatDate(value) {
  if (!value) return "Not available";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not available";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "long" }).format(date);
}

function quotaLabel(user) {
  if (user.authDisabled) return "Not enforced in this environment";
  const quota = user.quota;
  if (!quota || !Number.isFinite(quota.limit)) return "Not available";
  if (Number.isFinite(quota.per_benchmark_limit)) {
    const remaining = Number.isFinite(quota.remaining) ? quota.remaining : 0;
    return `${quota.per_benchmark_limit} per benchmark every 24 hours · ${remaining} quota ${remaining === 1 ? "slot" : "slots"} remaining across all tracks. Track availability is shown on the submission page.`;
  }
  return `${quota.remaining} of ${quota.limit} submissions remaining`;
}

export function Profile() {
  const [user, setUser] = useState(undefined);
  const [message, setMessage] = useState("");
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let live = true;
    setUser(undefined);
    setMessage("");
    fetchMe()
      .then((account) => { if (live) setUser(account); })
      .catch((error) => {
        if (!live) return;
        setUser(null);
        setMessage(errorMessage(error, "Your profile could not be loaded."));
      });
    return () => { live = false; };
  }, [reloadKey]);

  return (
    <WorkspacePage
      eyebrow="Account"
      title="Profile"
      description="Review your verified identity, sign in method, and current submission access."
      accountNavigation
    >
      <div className={ui.sectionBody}>
          {user === undefined && <p className="text-muted" role="status">Loading your profile...</p>}

          {message && (
            <div className={cn(ui.message, ui.messageError)} role="alert">
              {message}{" "}
              <button type="button" className={ui.linkButton} onClick={() => setReloadKey((value) => value + 1)}>Retry</button>
            </div>
          )}

          {user === null && !message && (
            <div className="grid border-y border-border-strong sm:grid-cols-[96px_minmax(0,1fr)]">
              <span className="grid min-h-24 place-items-center border-b border-border bg-surface-subtle sm:border-b-0 sm:border-r" aria-hidden="true"><UserRound size={28} /></span>
              <div className="p-6">
                <h2 className={ui.heading3}>Sign in required</h2>
                <p className="my-2 text-muted">Your profile is available after you sign in to a verified account.</p>
                <Button asChild variant="brand"><Link to="/login?next=/profile">Sign in</Link></Button>
              </div>
            </div>
          )}

          {user && (
            <div className="border-y border-border-strong">
              <div className="grid border-b border-border sm:grid-cols-[112px_minmax(0,1fr)_auto]">
                <span className="grid min-h-28 place-items-center border-b border-border bg-surface-subtle sm:border-b-0 sm:border-r" aria-hidden="true"><UserRound size={30} /></span>
                <div className="min-w-0 p-6">
                  <span className={ui.sectionTag}>Signed in account</span>
                  <h2 className="mb-1 [overflow-wrap:anywhere] font-display text-xl font-bold">{user.email}</h2>
                  <p className="text-muted">{user.isAdmin ? "Administrator" : "Leaderboard member"}</p>
                </div>
                <div className="flex items-start border-t border-border p-6 sm:border-l sm:border-t-0">
                  <span className={cn(ui.badge, user.emailVerified ? ui.badgePositive : ui.badgeNegative)}>{user.emailVerified ? "Verified" : "Verification required"}</span>
                </div>
              </div>

              <dl className="grid grid-cols-1 border-l border-t border-border sm:grid-cols-2 lg:grid-cols-3">
                <div className="min-w-0 border-b border-r border-border p-5"><dt className="mb-1 text-xs font-semibold uppercase text-faint">Email address</dt><dd className="m-0 break-words">{user.email}</dd></div>
                <div className="min-w-0 border-b border-r border-border p-5"><dt className="mb-1 text-xs font-semibold uppercase text-faint">Email status</dt><dd className="m-0">{user.emailVerified ? "Verified" : "Verification required"}</dd></div>
                <div className="min-w-0 border-b border-r border-border p-5"><dt className="mb-1 text-xs font-semibold uppercase text-faint">Sign in method</dt><dd className="m-0">{formatProvider(user.provider)}</dd></div>
                <div className="min-w-0 border-b border-r border-border p-5"><dt className="mb-1 text-xs font-semibold uppercase text-faint">Account created</dt><dd className="m-0">{formatDate(user.createdAt)}</dd></div>
                <div className="min-w-0 border-b border-r border-border p-5 sm:col-span-2"><dt className="mb-1 text-xs font-semibold uppercase text-faint">Submission quota</dt><dd className="m-0">{quotaLabel(user)}</dd></div>
              </dl>

              <div className="flex flex-wrap gap-2.5 p-6 max-sm:flex-col max-sm:[&>*]:w-full">
                <Button asChild variant="brand"><Link to="/submit"><Send size={16} />Submit a model</Link></Button>
                <Button asChild variant="ghost"><Link to="/submissions"><ClipboardList size={16} />View submissions</Link></Button>
              </div>
            </div>
          )}
      </div>
    </WorkspacePage>
  );
}
