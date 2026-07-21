import { useEffect, useState } from "react";
import { CircleCheck, Download, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { downloadFile, errorMessage, fetchMe, getJSON, postJSON } from "@/lib/api";
import { cn, fmtPct, prettyLabel } from "@/lib/utils";
import { ui } from "@/lib/styles";

function formatDate(value) {
  if (!value) return "N/A";
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  } catch {
    return "N/A";
  }
}

function statusLabel(row) {
  const moderation = row.moderation_status && row.moderation_status !== "visible"
    ? ` · ${prettyLabel(row.moderation_status)}`
    : "";
  return `${prettyLabel(row.status || "unknown")}${moderation}`;
}

function recordCountLabel(count) {
  return `${count} ${count === 1 ? "record" : "records"}`;
}

function DeleteSubmissionDialog({ state, onClose, onConfirm }) {
  const isOpen = Boolean(state);
  const isDeleting = state?.status === "deleting";
  const isDeleted = state?.status === "success";
  const Icon = isDeleted ? CircleCheck : Trash2;

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && !isDeleting) onClose();
      }}
    >
      {state && (
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
              <Icon className="size-5" />
            </div>
            <div className="min-w-0 flex-1">
              <span className={ui.sectionTag}>{prettyLabel(state.row.task_id)}</span>
              <DialogTitle className={ui.heading2}>
                {isDeleted ? "Submission deleted" : "Delete this submission?"}
              </DialogTitle>
              <DialogDescription className="mt-3 text-sm leading-relaxed text-muted">
                {isDeleted
                  ? `${state.row.model_name || "This model"} was removed from your history. The leaderboard now uses the latest remaining visible run for this benchmark, when one exists.`
                  : `This removes ${state.row.model_name || "this model"} from your submission history. The audit record remains retained for integrity and quota enforcement, and any earlier visible run becomes active on the leaderboard.`}
              </DialogDescription>
              {state.error && (
                <p className={cn(ui.message, ui.messageError, "mt-4")} role="alert">
                  {state.error}
                </p>
              )}
            </div>
          </div>
          <div className="mt-6 flex flex-wrap justify-end gap-2 border-t border-border pt-4">
            {isDeleted ? (
              <DialogClose asChild><Button type="button" variant="primary">Done</Button></DialogClose>
            ) : (
              <>
                <DialogClose asChild>
                  <Button type="button" variant="ghost" disabled={isDeleting}>Cancel</Button>
                </DialogClose>
                <Button
                  type="button"
                  variant="ghost"
                  className="border-negative text-negative hover:bg-negative-soft"
                  disabled={isDeleting}
                  onClick={onConfirm}
                >
                  <Trash2 size={15} aria-hidden="true" />
                  {isDeleting ? "Deleting..." : state.error ? "Retry deletion" : "Delete submission"}
                </Button>
              </>
            )}
          </div>
        </DialogContent>
      )}
    </Dialog>
  );
}

export function Submissions() {
  const [authEmail, setAuthEmail] = useState(undefined);
  const [rows, setRows] = useState([]);
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState("loading");
  const [reloadKey, setReloadKey] = useState(0);
  const [downloading, setDownloading] = useState("");
  const [deleteState, setDeleteState] = useState(null);

  useEffect(() => {
    let live = true;
    setStatus("loading");
    setMessage("");
    fetchMe()
      .then((user) => {
        if (!live) return null;
        setAuthEmail(user ? user.email : null);
        if (!user) {
          setRows([]);
          setStatus("ready");
          return null;
        }
        return getJSON("/api/submissions/mine");
      })
      .then((data) => {
        if (!live || !data) return;
        setRows(data.submissions || []);
        setStatus("ready");
      })
      .catch((error) => {
        if (!live) return;
        setStatus("error");
        setMessage(errorMessage(error, "Your submission history could not be loaded."));
      });
    return () => { live = false; };
  }, [reloadKey]);

  const download = async (row) => {
    setDownloading(row.submission_id);
    setMessage("");
    try {
      await downloadFile(row.submission_export_url, `${row.task_id}_${row.submission_id}.jsonl`);
    } catch (error) {
      setMessage(errorMessage(error, "The submission export could not be downloaded."));
    } finally {
      setDownloading("");
    }
  };

  const confirmDelete = async () => {
    const row = deleteState?.row;
    if (!row?.submission_id || deleteState?.status === "deleting") return;
    setDeleteState({ row, status: "deleting", error: "" });
    try {
      await postJSON(`/api/submissions/${row.submission_id}/delete`, {});
      setRows((current) => current.filter((item) => item.submission_id !== row.submission_id));
      setDeleteState({ row, status: "success", error: "" });
    } catch (error) {
      setDeleteState({
        row,
        status: "error",
        error: errorMessage(error, "This submission could not be deleted."),
      });
    }
  };

  return (
    <WorkspacePage
      eyebrow="Submission records"
      title="Evaluation history"
      description="Review processing state, benchmark scores, timestamps, and stored response exports for every upload."
      accountNavigation
    >
      <DeleteSubmissionDialog
        state={deleteState}
        onClose={() => setDeleteState(null)}
        onConfirm={confirmDelete}
      />
      <div className={ui.sectionBody}>
          {authEmail === null && (
            <div className="flex flex-col items-start gap-2 border-y border-border-strong py-6">
              <h3 className={ui.heading3}>Sign in required</h3>
              <p className="text-sm text-muted">Submission history is tied to your verified account.</p>
              <Button asChild variant="brand"><Link to="/login?next=/submissions">Sign in</Link></Button>
            </div>
          )}
          {authEmail && (
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-y border-border-strong px-4 py-4 text-sm text-muted">
              <span>Signed in as <strong className="break-all text-foreground">{authEmail}</strong></span>
              <span className={ui.badge}>{recordCountLabel(rows.length)}</span>
            </div>
          )}
          {message && <div className={cn(ui.message, ui.messageError, "mb-6")} role="alert">{message} <button type="button" className={ui.linkButton} onClick={() => setReloadKey((value) => value + 1)}>Retry</button></div>}

          {authEmail !== null && status === "loading" && (
            <div className="border-y border-border-strong px-5 py-10 text-center text-muted" role="status">Loading your submission history...</div>
          )}

          {authEmail && status === "ready" && rows.length === 0 && (
            <div className="flex flex-col items-start gap-3 border-y border-border-strong px-5 py-8">
              <h3 className={ui.heading3}>No submissions yet</h3>
              <p className="text-sm text-muted">Completed benchmark uploads will appear here with their score and export.</p>
              <Button asChild variant="brand"><Link to="/submit">Submit a model</Link></Button>
            </div>
          )}

          {rows.length > 0 && (
            <>
          <div className={cn(ui.tableWrap, "hidden md:block")}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Benchmark</th>
                  <th>Status</th>
                  <th className={ui.tableNumber}>Accuracy</th>
                  <th className={ui.tableNumber}>Macro average</th>
                  <th className={ui.tableNumber}>Rows</th>
                  <th>Submitted</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.submission_id || `${row.task_id}-${row.created_at}`}>
                    <td>{row.model_name || "N/A"}</td>
                    <td>{prettyLabel(row.task_id)}</td>
                    <td><span className={cn(ui.badge, row.moderation_status === "deleted" ? ui.badgeNegative : row.moderation_status === "hidden" ? ui.badgeMuted : ui.badgePositive)}>{statusLabel(row)}</span></td>
                    <td className={ui.tableNumber}>{fmtPct(row.accuracy)}</td>
                    <td className={ui.tableNumber}>{fmtPct(row.macro_accuracy)}</td>
                    <td className={ui.tableNumber}>{row.row_count ?? "N/A"}</td>
                    <td>{formatDate(row.created_at)}</td>
                    <td>
                      <div className="flex flex-wrap gap-2">
                        {row.submission_export_url && (
                          <Button type="button" variant="ghost" size="sm" title="Download JSONL" disabled={downloading === row.submission_id} onClick={() => download(row)}>
                            <Download size={15} aria-hidden="true" /> {downloading === row.submission_id ? "Downloading..." : "JSONL"}
                          </Button>
                        )}
                        {row.submission_id && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="border-negative text-negative hover:bg-negative-soft"
                            onClick={() => setDeleteState({ row, status: "confirm", error: "" })}
                          >
                            <Trash2 size={15} aria-hidden="true" /> Delete
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid border-l border-t border-border md:hidden">
            {rows.map((row) => (
              <article className="min-w-0 border-b border-r border-border p-5" key={`mobile-${row.submission_id || `${row.task_id}-${row.created_at}`}`}>
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <span className={ui.sectionTag}>{prettyLabel(row.task_id)}</span>
                    <h3 className="break-words font-display text-lg font-semibold">{row.model_name || "N/A"}</h3>
                  </div>
                  <span className={cn(ui.badge, row.moderation_status === "deleted" ? ui.badgeNegative : row.moderation_status === "hidden" ? ui.badgeMuted : ui.badgePositive)}>{statusLabel(row)}</span>
                </div>
                <dl className="mt-5 grid grid-cols-2 border-l border-t border-border text-sm">
                  <div className="border-b border-r border-border p-3"><dt className="text-xs font-semibold uppercase text-faint">Accuracy</dt><dd className="mt-1 tabular-nums">{fmtPct(row.accuracy)}</dd></div>
                  <div className="border-b border-r border-border p-3"><dt className="text-xs font-semibold uppercase text-faint">Rows</dt><dd className="mt-1 tabular-nums">{row.row_count ?? "N/A"}</dd></div>
                  <div className="col-span-2 border-b border-r border-border p-3"><dt className="text-xs font-semibold uppercase text-faint">Submitted</dt><dd className="mt-1">{formatDate(row.created_at)}</dd></div>
                </dl>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {row.submission_export_url && (
                    <Button className="w-full" type="button" variant="ghost" size="sm" disabled={downloading === row.submission_id} onClick={() => download(row)}>
                      <Download size={15} aria-hidden="true" /> {downloading === row.submission_id ? "Downloading..." : "Download JSONL"}
                    </Button>
                  )}
                  {row.submission_id && (
                    <Button
                      className="w-full border-negative text-negative hover:bg-negative-soft"
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeleteState({ row, status: "confirm", error: "" })}
                    >
                      <Trash2 size={15} aria-hidden="true" /> Delete submission
                    </Button>
                  )}
                </div>
              </article>
            ))}
          </div>
            </>
          )}
      </div>
    </WorkspacePage>
  );
}
