import { useEffect, useState } from "react";
import { CircleCheck, ClipboardList, Download, Send, Trash2, UserRound } from "lucide-react";
import { Link, useNavigate } from "react-router";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  clearUser,
  deleteJSON,
  downloadFile,
  errorMessage,
  fetchMe,
} from "@/lib/api";
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
    return `${quota.per_benchmark_limit} per module every 24 hours · ${remaining} quota ${remaining === 1 ? "slot" : "slots"} remaining across the framework. Module availability is shown on the submission page.`;
  }
  return `${quota.remaining} of ${quota.limit} submissions remaining`;
}

function DeleteAccountDialog({
  confirmation,
  error,
  onClose,
  onConfirm,
  onConfirmationChange,
  onDone,
  open,
  status,
}) {
  const isDeleting = status === "deleting";
  const isDeleted = status === "success";

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !isDeleting) onClose();
      }}
    >
      <DialogContent onEscapeKeyDown={(event) => { if (isDeleting) event.preventDefault(); }}>
        <div className="flex items-start gap-4 pr-10">
          <div
            className={cn(
              "grid size-11 shrink-0 place-items-center border",
              isDeleted
                ? "border-positive bg-positive-soft text-positive"
                : "border-negative bg-negative-soft text-negative",
            )}
            aria-hidden="true"
          >
            {isDeleted ? <CircleCheck className="size-5" /> : <Trash2 className="size-5" />}
          </div>
          <div className="min-w-0 flex-1">
            <span className={ui.sectionTag}>Privacy control</span>
            <DialogTitle className={ui.heading2}>
              {isDeleted ? "Account deleted" : "Delete your account?"}
            </DialogTitle>
            <DialogDescription className="mt-3 text-sm leading-relaxed text-muted">
              {isDeleted
                ? "Your identity and sign-in credentials have been removed. Published research results remain available in anonymised form."
                : "This permanently removes your account identity, credentials, and sign-in access. Published leaderboard results remain available but are detached from your identity."}
            </DialogDescription>
            {!isDeleted && (
              <label className={cn(ui.field, "mt-5 mb-0")}>
                Type DELETE to confirm
                <input
                  className={ui.input}
                  value={confirmation}
                  autoComplete="off"
                  disabled={isDeleting}
                  onChange={(event) => onConfirmationChange(event.target.value)}
                />
              </label>
            )}
            {error && (
              <p className={cn(ui.message, ui.messageError, "mt-4")} role="alert">
                {error}
              </p>
            )}
          </div>
        </div>
        <div className="mt-6 flex flex-wrap justify-end gap-2 border-t border-border pt-4">
          {isDeleted ? (
            <Button type="button" variant="primary" onClick={onDone}>Return to overview</Button>
          ) : (
            <>
              <DialogClose asChild>
                <Button type="button" variant="ghost" disabled={isDeleting}>Cancel</Button>
              </DialogClose>
              <Button
                type="button"
                variant="ghost"
                className="border-negative text-negative hover:bg-negative-soft"
                disabled={isDeleting || confirmation !== "DELETE"}
                onClick={onConfirm}
              >
                <Trash2 size={15} aria-hidden="true" />
                {isDeleting ? "Deleting..." : error ? "Retry deletion" : "Delete account"}
              </Button>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function Profile() {
  const navigate = useNavigate();
  const [user, setUser] = useState(undefined);
  const [message, setMessage] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [privacyMessage, setPrivacyMessage] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteStatus, setDeleteStatus] = useState("idle");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [deleteError, setDeleteError] = useState("");

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

  const handleExport = async () => {
    setExporting(true);
    setPrivacyMessage(null);
    try {
      const filename = await downloadFile(
        "/api/auth/me/export",
        "ms-vista-account-data.json",
      );
      setPrivacyMessage({
        tone: "success",
        text: `Your account data was downloaded as ${filename}.`,
      });
    } catch (error) {
      setPrivacyMessage({
        tone: "error",
        text: errorMessage(error, "Your account data could not be downloaded."),
      });
    } finally {
      setExporting(false);
    }
  };

  const closeDeleteDialog = () => {
    setDeleteOpen(false);
    setDeleteStatus("idle");
    setDeleteConfirmation("");
    setDeleteError("");
  };

  const handleDelete = async () => {
    if (deleteConfirmation !== "DELETE") return;
    setDeleteStatus("deleting");
    setDeleteError("");
    try {
      await deleteJSON("/api/auth/me");
      clearUser();
      setUser(null);
      setDeleteStatus("success");
    } catch (error) {
      setDeleteStatus("idle");
      setDeleteError(errorMessage(error, "Your account could not be deleted."));
    }
  };

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
              <section className="border-t border-border p-6" aria-labelledby="privacy-controls-heading">
                <span className={ui.sectionTag}>Privacy controls</span>
                <h2 className={ui.heading3} id="privacy-controls-heading">Manage your account data</h2>
                <p className="my-2 max-w-3xl text-sm leading-relaxed text-muted">
                  Download the account and submission information associated with your identity, or permanently remove your identity and sign-in credentials. Published research results are retained in anonymised form.
                </p>
                <div className="mt-4 flex flex-wrap gap-2.5 max-sm:flex-col max-sm:[&>*]:w-full">
                  <Button
                    type="button"
                    variant="ghost"
                    disabled={exporting || user.authDisabled}
                    onClick={handleExport}
                  >
                    <Download size={16} aria-hidden="true" />
                    {exporting ? "Preparing download..." : "Download account data"}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    className="border-negative text-negative hover:bg-negative-soft"
                    disabled={user.authDisabled}
                    onClick={() => setDeleteOpen(true)}
                  >
                    <Trash2 size={16} aria-hidden="true" />
                    Delete account
                  </Button>
                </div>
                {user.authDisabled && (
                  <p className="mt-3 text-sm text-muted">Privacy actions are unavailable while authentication is disabled.</p>
                )}
                {privacyMessage && (
                  <p
                    className={cn(
                      ui.message,
                      privacyMessage.tone === "success" ? ui.messageSuccess : ui.messageError,
                      "mt-4",
                    )}
                    role={privacyMessage.tone === "error" ? "alert" : "status"}
                  >
                    {privacyMessage.text}
                  </p>
                )}
              </section>
            </div>
          )}
      </div>
      <DeleteAccountDialog
        confirmation={deleteConfirmation}
        error={deleteError}
        onClose={closeDeleteDialog}
        onConfirm={handleDelete}
        onConfirmationChange={setDeleteConfirmation}
        onDone={() => navigate("/", { replace: true })}
        open={deleteOpen}
        status={deleteStatus}
      />
    </WorkspacePage>
  );
}
