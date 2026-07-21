import { useEffect, useState } from "react";
import { DatabaseBackup, Download, EyeOff, RotateCcw, Shield, Trash2, Undo2 } from "lucide-react";
import { Link } from "react-router-dom";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";
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

function moderationLabel(value) {
  return prettyLabel(value || "visible");
}

function AdminActions({ row, busy, isBusy, download, runAction }) {
  const actionInProgress = Boolean(busy);
  return (
    <div className="flex flex-wrap items-center gap-1">
      {row.submission_export_url && (
        <Button type="button" variant="ghost" size="sm" title="Download JSONL" disabled={actionInProgress} onClick={() => download(row.submission_export_url, `${row.task_id}_${row.submission_id}.jsonl`, `export:${row.submission_id}`)}>
          <Download size={15} />
        </Button>
      )}
      {row.submission_id && (
        <>
          <Button type="button" variant="ghost" size="sm" title="Rescore" disabled={actionInProgress || isBusy(row.submission_id, "rescore")} onClick={() => runAction(row.submission_id, "rescore", "Rescore")}><RotateCcw size={15} /></Button>
          {row.moderation_status === "visible" ? (
            <Button type="button" variant="ghost" size="sm" title="Hide" disabled={actionInProgress || isBusy(row.submission_id, "hide")} onClick={() => runAction(row.submission_id, "hide", "Hide")}><EyeOff size={15} /></Button>
          ) : (
            <Button type="button" variant="ghost" size="sm" title="Restore" disabled={actionInProgress || isBusy(row.submission_id, "restore")} onClick={() => runAction(row.submission_id, "restore", "Restore")}><Undo2 size={15} /></Button>
          )}
          {row.moderation_status !== "deleted" && (
            <Button type="button" variant="ghost" size="sm" title="Soft delete" disabled={actionInProgress || isBusy(row.submission_id, "delete")} onClick={() => runAction(row.submission_id, "delete", "Delete")}><Trash2 size={15} /></Button>
          )}
        </>
      )}
    </div>
  );
}

export function Admin() {
  const [user, setUser] = useState(undefined);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState("");
  const [backupStatus, setBackupStatus] = useState(null);
  const [message, setMessage] = useState(null);
  const [status, setStatus] = useState("loading");

  const load = async () => {
    const [submissionsResult, backupResult] = await Promise.allSettled([
      getJSON("/api/admin/submissions?limit=300"),
      getJSON("/api/admin/backups/status"),
    ]);
    if (submissionsResult.status === "rejected") throw submissionsResult.reason;
    setRows(submissionsResult.value.submissions || []);
    setBackupStatus(
      backupResult.status === "fulfilled"
        ? backupResult.value
        : {
            status: "unavailable",
            error: errorMessage(
              backupResult.reason,
              "Automatic backup status could not be loaded.",
            ),
          },
    );
    setStatus("ready");
    return submissionsResult.value;
  };

  useEffect(() => {
    let live = true;
    fetchMe()
      .then((current) => {
        if (!live) return null;
        setUser(current);
        if (current?.isAdmin) return load();
        setStatus("ready");
        return null;
      })
      .catch((error) => {
        if (!live) return;
        setStatus("error");
        setMessage({ ok: false, text: errorMessage(error, "Administrator status could not be checked.") });
      });
    return () => { live = false; };
  }, []);

  const runAction = async (submissionId, action, label) => {
    if (busy) return;
    setBusy(`${submissionId}:${action}`);
    setMessage(null);
    let actionCompleted = false;
    try {
      await postJSON(`/api/admin/submissions/${submissionId}/${action}`, {});
      actionCompleted = true;
      await load();
      setMessage({ ok: true, text: `${label} complete. The audit list has been refreshed.` });
    } catch (error) {
      setMessage({
        ok: false,
        text: actionCompleted
          ? `${label} completed, but the updated audit list could not be loaded. Refresh before taking another action. ${errorMessage(error)}`
          : errorMessage(error, `${label} could not be completed.`),
      });
    } finally {
      setBusy("");
    }
  };

  const rebuild = async () => {
    if (busy) return;
    setBusy("rebuild");
    setMessage(null);
    let rebuilt = false;
    try {
      const data = await postJSON("/api/admin/rescore", {});
      rebuilt = true;
      await load();
      setMessage({
        ok: true,
        text: `Rebuilt the leaderboard from ${data.rescored || 0} current model and benchmark results.`,
      });
    } catch (error) {
      setMessage({
        ok: false,
        text: rebuilt
          ? `The leaderboard rebuild completed, but the audit list refresh failed. Refresh this page before rebuilding again. ${errorMessage(error)}`
          : errorMessage(error, "The leaderboard rebuild could not be completed."),
      });
    } finally {
      setBusy("");
    }
  };

  const isBusy = (submissionId, action) => busy === `${submissionId}:${action}`;

  const createServerBackup = async () => {
    if (busy) return;
    setBusy("server-backup");
    setMessage(null);
    try {
      const data = await postJSON("/api/admin/backups/run", {});
      setBackupStatus(data);
      setMessage({
        ok: true,
        text: `Verified server backup ${data.filename || "archive"} was created and retained successfully.`,
      });
    } catch (error) {
      setMessage({
        ok: false,
        text: errorMessage(
          error,
          "A verified server backup could not be created. Existing backups were not changed.",
        ),
      });
    } finally {
      setBusy("");
    }
  };

  const download = async (url, fallbackName, busyKey, options) => {
    if (busy) return;
    setBusy(busyKey);
    setMessage(null);
    try {
      const filename = await downloadFile(url, fallbackName, options);
      setMessage({ ok: true, text: `Downloaded ${filename}.` });
    } catch (error) {
      setMessage({ ok: false, text: errorMessage(error, "The requested file could not be downloaded.") });
    } finally {
      setBusy("");
    }
  };

  return (
    <WorkspacePage
      eyebrow="Administration"
      title="Submission operations"
      description="Audit ranked submissions, control visibility, rescore stored responses, and recover the published leaderboard."
      accountNavigation
      includeAdmin
    >
      <div className={ui.sectionBody}>
          {user === null && (
            <div className="flex flex-col items-start gap-2 border-y border-border-strong py-6">
              <h3 className={ui.heading3}>Sign in required</h3>
              <p className="text-sm text-muted">Admin tools require an approved account.</p>
              <Button asChild variant="brand"><Link to="/login?next=/admin">Sign in</Link></Button>
            </div>
          )}
          {user && !user.isAdmin && (
            <div className="flex flex-col items-start gap-2 border-y border-border-strong py-6">
              <h3 className={ui.heading3}>Admin access required</h3>
              <p className="text-sm text-muted">Your signed-in account does not have administrator permissions.</p>
            </div>
          )}
          {message && <div className={cn(ui.message, message.ok ? ui.messageSuccess : ui.messageError, "mb-6")} role={message.ok ? "status" : "alert"}>{message.text} {status === "error" && <button type="button" className={ui.linkButton} onClick={() => window.location.reload()}>Retry</button>}</div>}
          {user?.isAdmin && (
            <>
              <div className="mb-6 border-y border-border-strong">
                <div className="flex items-start justify-between gap-5 border-b border-border p-5 max-sm:flex-col">
                  <div>
                    <span className={ui.sectionTag}>Admin account</span>
                    <h3 className={ui.heading3}>{user.email}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className={ui.badge}>{rows.length} submissions</span>
                    <span className={ui.badge}>Audit + rescore</span>
                  </div>
                </div>
                <div className="grid border-b border-border sm:grid-cols-3">
                  <div className="border-b border-border p-5 sm:border-b-0 sm:border-r">
                    <span className={ui.sectionTag}>Automatic backups</span>
                    <strong className="block font-display text-lg">
                      {backupStatus?.backup?.enabled
                        ? backupStatus.backup.interval_hours === 48
                          ? "Every 2 days"
                          : `Every ${backupStatus.backup.interval_hours} hours`
                        : "Not active"}
                    </strong>
                  </div>
                  <div className="border-b border-border p-5 sm:border-b-0 sm:border-r">
                    <span className={ui.sectionTag}>Latest verified backup</span>
                    <strong className="block break-words font-display text-sm">
                      {formatDate(backupStatus?.backup?.latest_backup_at)}
                    </strong>
                  </div>
                  <div className="p-5">
                    <span className={ui.sectionTag}>Retention</span>
                    <strong className="block font-display text-lg">
                      {backupStatus?.backup?.retention_count ?? "N/A"} archives
                    </strong>
                  </div>
                  {backupStatus?.error && (
                    <p className="border-t border-border p-5 text-sm text-negative sm:col-span-3">
                      {backupStatus.error}
                    </p>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 p-5 max-sm:flex-col max-sm:[&>*]:w-full">
                  <Button type="button" variant="brand" onClick={rebuild} disabled={Boolean(busy)}>
                    <Shield size={16} /> Rebuild leaderboard
                  </Button>
                  <Button type="button" variant="ghost" disabled={Boolean(busy)} onClick={createServerBackup}>
                    <DatabaseBackup size={16} /> {busy === "server-backup" ? "Creating backup..." : "Create server backup"}
                  </Button>
                  <Button type="button" variant="ghost" disabled={Boolean(busy)} onClick={() => download("/api/admin/backups/download", "ms-vista-backup.zip", "backup", { method: "POST" })}>
                    <Download size={16} /> {busy === "backup" ? "Preparing backup..." : "Download SQLite backup"}
                  </Button>
                </div>
              </div>
              <div className={cn(ui.tableWrap, "hidden md:block")}>
                <table className={ui.table}>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Benchmark</th>
                      <th>User</th>
                      <th>Status</th>
                      <th className={ui.tableNumber}>Accuracy</th>
                      <th className={ui.tableNumber}>Rows</th>
                      <th>Submitted</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status === "loading" && (
                      <tr><td className={ui.emptyRow} colSpan="8">Loading the submission audit list...</td></tr>
                    )}
                    {status === "ready" && rows.length === 0 && (
                      <tr><td className={ui.emptyRow} colSpan="8">No submissions found.</td></tr>
                    )}
                    {rows.map((row) => (
                      <tr key={row.submission_id || `${row.task_id}-${row.created_at}`}>
                        <td>{row.model_name || "N/A"}</td>
                        <td>{prettyLabel(row.task_id)}</td>
                        <td>{row.user_email || "N/A"}</td>
                        <td><span className={cn(ui.badge, row.moderation_status === "deleted" ? ui.badgeNegative : row.moderation_status === "hidden" ? ui.badgeMuted : ui.badgePositive)}>{moderationLabel(row.moderation_status)}</span></td>
                        <td className={ui.tableNumber}>{fmtPct(row.accuracy)}</td>
                        <td className={ui.tableNumber}>{row.row_count ?? "N/A"}</td>
                        <td>{formatDate(row.created_at)}</td>
                        <td>
                          <div className="flex items-center gap-1">
                            <AdminActions row={row} busy={busy} isBusy={isBusy} download={download} runAction={runAction} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="grid border-l border-t border-border md:hidden">
                {status === "loading" && <p className="border-b border-r border-border p-6 text-muted" role="status">Loading the submission audit list...</p>}
                {status === "ready" && rows.length === 0 && <p className="border-b border-r border-border p-6 text-muted">No submissions found.</p>}
                {rows.map((row) => (
                  <article className="min-w-0 border-b border-r border-border p-5" key={`mobile-${row.submission_id || `${row.task_id}-${row.created_at}`}`}>
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <span className={ui.sectionTag}>{prettyLabel(row.task_id)}</span>
                        <h3 className="break-words font-display text-lg font-semibold">{row.model_name || "N/A"}</h3>
                        <p className="mt-1 break-all text-sm text-muted">{row.user_email || "N/A"}</p>
                      </div>
                      <span className={cn(ui.badge, row.moderation_status === "deleted" ? ui.badgeNegative : row.moderation_status === "hidden" ? ui.badgeMuted : ui.badgePositive)}>{moderationLabel(row.moderation_status)}</span>
                    </div>
                    <dl className="my-4 grid grid-cols-2 border-l border-t border-border text-sm">
                      <div className="border-b border-r border-border p-3"><dt className="text-xs font-semibold uppercase text-faint">Accuracy</dt><dd className="mt-1 tabular-nums">{fmtPct(row.accuracy)}</dd></div>
                      <div className="border-b border-r border-border p-3"><dt className="text-xs font-semibold uppercase text-faint">Rows</dt><dd className="mt-1 tabular-nums">{row.row_count ?? "N/A"}</dd></div>
                      <div className="col-span-2 border-b border-r border-border p-3"><dt className="text-xs font-semibold uppercase text-faint">Submitted</dt><dd className="mt-1">{formatDate(row.created_at)}</dd></div>
                    </dl>
                    <AdminActions row={row} busy={busy} isBusy={isBusy} download={download} runAction={runAction} />
                  </article>
                ))}
              </div>
            </>
          )}
      </div>
    </WorkspacePage>
  );
}
