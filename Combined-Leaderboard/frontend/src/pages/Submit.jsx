import { cloneElement, isValidElement, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Box, CircleAlert, CircleCheck, ExternalLink, Plus } from "lucide-react";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FileDropzone } from "@/components/FileDropzone";
import { apiUrl, downloadFile, getJSON, postFormData, postJSON, fetchMe, clearUser, logout, errorMessage } from "@/lib/api";
import { cn, fmtPct } from "@/lib/utils";
import { ui } from "@/lib/styles";
import { submitTasks } from "@/data/benchmarks";

const modelAccessOptions = [
  ["open", "Open source"],
  ["open_weights", "Open weights"],
  ["closed", "Closed source"],
  ["research", "Research preview"],
];

function fieldValue(formData, key) {
  return String(formData.get(key) || "").trim();
}

function wordCount(value) {
  return String(value || "").trim().split(/\s+/).filter(Boolean).length;
}

function collectModelMeta(formData, task) {
  return {
    method_description: fieldValue(formData, "method_description"),
    cot_used: fieldValue(formData, "cot_used"),
    prompt_template: fieldValue(formData, "prompt_template"),
    changes_from_previous: fieldValue(formData, "changes_from_previous"),
    submission_track: task.id,
  };
}

const RUN_REQUIRED_FIELDS = [
  ["cot_used", "CoT used?"],
  ["method_description", "Method description"],
  ["prompt_template", "Prompt template"],
  ["changes_from_previous", "Changes from previous submission"],
];

const MODEL_REQUIRED_FIELDS = [
  ["model_name", "Model name"],
  ["organization", "Organisation"],
  ["access", "Source status"],
];

const submissionIssueLabels = {
  invalid_jsonl_syntax: "Invalid JSON syntax",
  jsonl_row_not_object: "Incorrect JSONL row format",
  missing_question_id: "Missing question ID",
  conflicting_question_ids: "Conflicting question IDs",
  missing_answer_field: "Missing answer field",
  multiple_answer_fields: "Multiple answer fields",
  invalid_answer_type: "Incorrect answer type",
  invalid_answer_value: "Invalid numeric answer",
  empty_sample_outputs: "Blank model outputs",
  invalid_submission_condition: "Unknown evaluation condition",
  condition_not_supported_for_task: "Condition not supported",
  duplicate_sample_output: "Duplicate sample output",
  empty_submission_file: "Empty response file",
  submission_line_too_long: "JSONL row is too large",
  answer_too_long: "Final answer is too long",
  too_many_submission_rows: "Too many response rows",
  missing_standard_condition: "Standard outputs missing",
  missing_required_conditions: "Required evaluation conditions missing",
  unknown_sample_ids: "Unknown question IDs",
  missing_sample_outputs: "Sample outputs missing",
  sample_id_coverage_mismatch: "Question ID coverage mismatch",
  invalid_run_manifest_encoding: "Run manifest encoding error",
  invalid_run_manifest_json: "Invalid run manifest JSON",
  unsupported_run_manifest_version: "Outdated spatial harness",
  unsupported_spatial_submission_version: "Outdated spatial output format",
  debug_spatial_run_not_allowed: "Incomplete debug run",
  spatial_model_name_mismatch: "Model names do not match",
  spatial_dataset_set_mismatch: "Official datasets missing",
  spatial_condition_set_mismatch: "Official conditions missing",
  spatial_run_contains_errors: "Harness run contains errors",
  spatial_condition_count_mismatch: "Condition counts do not match",
  spatial_submission_hash_mismatch: "Spatial files do not match",
  spatial_submission_row_count_mismatch: "Response row count does not match",
  spatial_benchmark_version_mismatch: "Benchmark version does not match",
  spatial_provenance_mismatch: "Evaluation provenance does not match",
  spatial_judge_mismatch: "Judge revision does not match",
  spatial_harness_version_mismatch: "Harness version does not match",
  spatial_decoding_mismatch: "Decoding settings do not match",
  spatial_judge_count_mismatch: "Judge output counts do not match",
  invalid_spatial_upload_parts: "Use one spatial ZIP package",
  invalid_spatial_archive_file: "Incorrect spatial package type",
  invalid_spatial_archive_size: "Spatial package size is invalid",
  empty_spatial_submission_archive: "Spatial package is empty",
  spatial_submission_archive_too_large: "Spatial package is too large",
  invalid_spatial_archive_contents: "Spatial package contents are invalid",
  duplicate_spatial_archive_members: "Spatial package contains duplicate files",
  empty_spatial_archive_member: "Spatial package contains an empty file",
  spatial_archive_member_too_large: "Spatial package content is too large",
  encrypted_spatial_archive: "Encrypted spatial packages are unsupported",
  unsupported_spatial_archive_compression: "Unsupported ZIP compression",
  unsafe_spatial_archive_member: "Unsafe spatial package content",
  unsafe_spatial_archive_ratio: "Unsafe ZIP compression ratio",
  unreadable_spatial_archive_member: "Unreadable spatial package content",
  spatial_archive_size_mismatch: "Spatial package size mismatch",
  invalid_spatial_evidence_json: "Invalid public evidence JSON",
  invalid_spatial_evidence_fields: "Outdated public evidence format",
  unknown_spatial_evidence_sample: "Unknown spatial evidence sample",
  duplicate_spatial_evidence_sample: "Duplicate spatial evidence sample",
  missing_spatial_evidence_samples: "Spatial evidence samples missing",
  spatial_evidence_metadata_mismatch: "Spatial evidence metadata mismatch",
  spatial_evidence_condition_count_mismatch: "Spatial evidence counts do not match",
  spatial_evidence_group_count_mismatch: "Spatial scoring groups do not match",
  spatial_report_artifact_mismatch: "Aggregate report does not match the package",
  spatial_report_evidence_mismatch: "Aggregate scores do not match the evidence",
  unsupported_spatial_report_version: "Outdated spatial report format",
};

function SubmissionIssueDetails({ issue }) {
  if (!issue?.code) return null;
  const ids = issue.question_ids || issue.missing_question_ids || [];
  const unknownIds = issue.unknown_question_ids || [];
  return (
    <div className="mt-2 border-t border-current/30 pt-2" aria-label="Submission validation details">
      <strong>{submissionIssueLabels[issue.code] || "Submission file issue"}</strong>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs">
        {issue.line_number != null && <span>Line {issue.line_number}</span>}
        {issue.condition && <span>Condition: {issue.condition}</span>}
        {issue.count != null && <span>Affected rows: {issue.count}</span>}
        {issue.missing_count != null && <span>Missing: {issue.missing_count}</span>}
        {issue.unknown_count != null && <span>Unknown: {issue.unknown_count}</span>}
      </div>
      {ids.length > 0 && <p className="mt-1 break-words">Example IDs: <code className="bg-surface-subtle px-1.5 py-0.5">{ids.slice(0, 8).join(", ")}</code></p>}
      {unknownIds.length > 0 && <p className="mt-1 break-words">Unknown IDs: <code className="bg-surface-subtle px-1.5 py-0.5">{unknownIds.slice(0, 8).join(", ")}</code></p>}
    </div>
  );
}

export function SubmissionResultDialog({ result, onClose }) {
  const isOpen = Boolean(result);
  const succeeded = result?.ok === true;
  const pending = result?.pending === true;
  const StatusIcon = succeeded ? CircleCheck : CircleAlert;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) onClose(); }}>
      {result && (
        <DialogContent>
          <div className="flex items-start gap-4 pr-10">
            <div
              className={cn(
                "grid size-11 shrink-0 place-items-center border",
                succeeded
                  ? "border-positive bg-positive-soft text-positive"
                  : pending
                    ? "border-warning bg-warning-soft text-warning"
                  : "border-negative bg-negative-soft text-negative",
              )}
              aria-hidden="true"
            >
              <StatusIcon className="size-5" />
            </div>
            <div className="min-w-0 flex-1">
              <span className={ui.sectionTag}>{result.taskLabel}</span>
              <DialogTitle className={ui.heading2}>{result.title}</DialogTitle>
              <DialogDescription className="mt-3 text-sm leading-relaxed text-muted">
                {result.text}
              </DialogDescription>
              {result.evidenceUrl && (
                <a
                  className="mt-4 inline-flex min-h-10 items-center gap-2 border border-border-strong px-3 py-2 text-sm font-semibold text-foreground hover:bg-surface-subtle"
                  href={apiUrl(result.evidenceUrl)}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  View public evidence <ExternalLink className="size-4" aria-hidden="true" />
                </a>
              )}
              <SubmissionIssueDetails issue={result.validation} />
            </div>
          </div>
          <div className="mt-6 flex justify-end border-t border-border pt-4">
            <DialogClose asChild>
              <Button type="button" variant={succeeded || pending ? "primary" : "ghost"}>
                {succeeded || pending ? "Done" : "Review submission"}
              </Button>
            </DialogClose>
          </div>
        </DialogContent>
      )}
    </Dialog>
  );
}

function Field({ label, required = false, error, className = "", children, as: Tag = "label" }) {
  const control = isValidElement(children) && typeof children.type === "string"
    ? cloneElement(children, {
        className: cn(children.props.className, children.type === "textarea" ? ui.textarea : ui.input, error && "border-negative focus:ring-negative-soft"),
      })
    : children;
  return (
    <Tag className={cn(ui.field, className)}>
      <span>
        {label} {required && <em className="not-italic text-negative">*</em>}
      </span>
      {control}
      {error && (
        <span className={ui.fieldError} role="alert">
          {error}
        </span>
      )}
    </Tag>
  );
}

export function Submit() {
  const [authEmail, setAuthEmail] = useState(undefined);
  const [quota, setQuota] = useState(null);
  const [authDisabled, setAuthDisabled] = useState(false);
  const [authError, setAuthError] = useState("");
  const refreshAuth = async () => {
    try {
      const u = await fetchMe();
      setAuthEmail(u ? u.email : null);
      setQuota(u?.quota || null);
      setAuthDisabled(Boolean(u?.authDisabled));
      setAuthError("");
      return u;
    } catch (error) {
      setAuthError(errorMessage(error, "Your account status could not be checked."));
      return null;
    }
  };
  useEffect(() => { refreshAuth(); }, []);
  const [taskInfo, setTaskInfo] = useState({});
  const [taskInfoError, setTaskInfoError] = useState("");
  const [accountNotice, setAccountNotice] = useState("");
  const [messages, setMessages] = useState({});
  const [submissionResult, setSubmissionResult] = useState(null);
  const [errors, setErrors] = useState({});
  const [busyTask, setBusyTask] = useState("");
  const [busyDownload, setBusyDownload] = useState("");
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [modelsStatus, setModelsStatus] = useState("idle");
  const [modelRegistryError, setModelRegistryError] = useState("");
  const [modelRegistrationError, setModelRegistrationError] = useState("");
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [modelErrors, setModelErrors] = useState({});
  const [registeringModel, setRegisteringModel] = useState(false);
  const selectedModel = models.find((model) => model.model_id === selectedModelId) || null;

  const refreshModels = async () => {
    setModelsStatus("loading");
    setModelRegistryError("");
    try {
      const data = await getJSON("/api/models/mine");
      const nextModels = data.models || [];
      setModels(nextModels);
      setSelectedModelId((current) => (
        nextModels.some((model) => model.model_id === current)
          ? current
          : nextModels[0]?.model_id || ""
      ));
      setModelsStatus("ready");
      return nextModels;
    } catch (error) {
      setModelsStatus("error");
      setModelRegistryError(errorMessage(error, "Your registered models could not be loaded."));
      return [];
    }
  };

  useEffect(() => {
    if (authEmail) {
      refreshModels();
    } else {
      setModels([]);
      setSelectedModelId("");
      setModelsStatus("idle");
    }
  }, [authEmail]);

  const openModelDialog = () => {
    setModelErrors({});
    setModelRegistrationError("");
    setModelDialogOpen(true);
  };

  const handleModelDialogOpenChange = (open) => {
    if (!open && registeringModel) return;
    setModelDialogOpen(open);
    if (!open) {
      setModelErrors({});
      setModelRegistrationError("");
    }
  };

  const registerModel = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const payload = Object.fromEntries(
      ["model_name", "organization", "access", "parameter_count", "paper_url"]
        .map((name) => [name, fieldValue(data, name)]),
    );
    const fieldErrors = {};
    for (const [name, label] of MODEL_REQUIRED_FIELDS) {
      if (!payload[name]) fieldErrors[name] = `${label} is required.`;
    }
    if (Object.keys(fieldErrors).length) {
      setModelErrors(fieldErrors);
      return;
    }
    setRegisteringModel(true);
    setModelErrors({});
    setModelRegistrationError("");
    try {
      const response = await postJSON("/api/models", payload);
      const created = response.model;
      const nextModels = await refreshModels();
      setSelectedModelId(
        nextModels.find((model) => model.model_id === created.model_id)?.model_id
          || created.model_id,
      );
      setModelDialogOpen(false);
      form.reset();
    } catch (error) {
      if (error.fieldErrors && Object.keys(error.fieldErrors).length) {
        setModelErrors(error.fieldErrors);
      }
      setModelRegistrationError(errorMessage(error, "The model could not be registered."));
    } finally {
      setRegisteringModel(false);
    }
  };
  useEffect(() => {
    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    if (hash.get("verified")) {
      setAccountNotice("Your email address is verified and you are signed in. You can now submit model responses.");
      window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    }
  }, []);
  const download = async (taskId, kind, url, filename) => {
    const key = `${taskId}:${kind}`;
    setBusyDownload(key);
    setMessages((current) => ({ ...current, [taskId]: { text: `Preparing ${kind} download...` } }));
    try {
      const downloaded = await downloadFile(url, filename);
      setMessages((current) => ({ ...current, [taskId]: { ok: true, text: `Downloaded ${downloaded}.` } }));
    } catch (error) {
      setMessages((current) => ({ ...current, [taskId]: { ok: false, text: errorMessage(error, `The ${kind} file could not be downloaded.`) } }));
    } finally {
      setBusyDownload("");
    }
  };
  const clearFieldError = (taskId, name) => {
    if (!name) return;
    setErrors((current) => {
      const taskErrors = current[taskId];
      if (!taskErrors || !taskErrors[name]) return current;
      const next = { ...taskErrors };
      delete next[name];
      return { ...current, [taskId]: next };
    });
  };
  useEffect(() => {
    Promise.allSettled(
      submitTasks.map((task) => getJSON(`/api/tasks/${task.id}/info`)),
    ).then((results) => {
      const failures = results.filter((result) => result.status === "rejected");
      const infos = results.map((result) => result.status === "fulfilled" ? result.value : {});
      setTaskInfo(
        Object.fromEntries(submitTasks.map((task, i) => [task.id, infos[i]])),
      );
      setTaskInfoError(
        failures.length
          ? `${failures.length} benchmark description${failures.length > 1 ? "s" : ""} could not be loaded. Downloads and submissions remain available. ${errorMessage(failures[0].reason)}`
          : "",
      );
    });
  }, []);
  const submit = async (event, task) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    data.set("model_id", selectedModelId);
    const modelMeta = collectModelMeta(data, task);
    const fieldErrors = {};
    if (!selectedModelId) fieldErrors.model_id = "Select a registered model before submitting.";
    for (const [name, label] of RUN_REQUIRED_FIELDS) {
      if (!fieldValue(data, name)) fieldErrors[name] = `${label} is required.`;
    }
    if (!fieldErrors.method_description && wordCount(modelMeta.method_description) < 100) {
      fieldErrors.method_description = "Method description must be at least 100 words.";
    }
    if (!fieldErrors.changes_from_previous && wordCount(modelMeta.changes_from_previous) < 50) {
      fieldErrors.changes_from_previous = "Changes from previous submission must be at least 50 words.";
    }
    const responseFile = data.get("file");
    if (!responseFile || !responseFile.name) {
      fieldErrors.file = task.harness
        ? "The spatial submission package (ZIP) is required."
        : "A response file (JSONL) is required.";
    }
    if (Object.keys(fieldErrors).length) {
      setErrors((current) => ({ ...current, [task.id]: fieldErrors }));
      const count = Object.keys(fieldErrors).length;
      setSubmissionResult({
        ok: false,
        taskLabel: task.label,
        title: "Review required fields",
        text: `Please complete ${count} required field${count > 1 ? "s" : ""} highlighted in the ${task.label} submission form.`,
      });
      return;
    }
    setErrors((current) => ({ ...current, [task.id]: {} }));
    data.set("model_meta", JSON.stringify(modelMeta));
    setBusyTask(task.id);
    setSubmissionResult(null);
    try {
      const json = await postFormData(`/api/tasks/${task.id}/submit`, data);
      if (json.published === false || json.code === "leaderboard_publication_pending") {
        setSubmissionResult({
          ok: false,
          pending: true,
          taskLabel: task.label,
          title: "Submission saved, publication pending",
          text: json.message || "Your submission was scored and stored. It is waiting for an administrator to republish the leaderboard, so do not upload it again.",
        });
        await Promise.all([refreshAuth(), refreshModels()]);
        return;
      }
      const visualAggregation = task.id === "do_you_see_me"
        ? "dimension balanced task macro"
        : "unweighted task macro";
      setSubmissionResult({
        ok: true,
        taskLabel: task.label,
        title: task.harness ? "Evidence published" : "Submission scored",
        evidenceUrl: task.harness ? json.public_evidence_url : null,
        text: task.harness && json.macro_accuracy != null
          ? `Validated and published a ${fmtPct(json.macro_accuracy)} macro average across ${Object.keys(json.groups || {}).length} datasets, with ${fmtPct(json.accuracy)} micro accuracy over ${json.total_samples || 0} evaluation groups. The package and per sample evidence are now public.`
          : json.macro_accuracy != null
            ? `Scored ${fmtPct(json.macro_accuracy)} ${visualAggregation}, with ${fmtPct(json.accuracy)} micro accuracy over ${json.total_samples || 0} released suite questions.`
            : `Scored ${fmtPct(json.accuracy)} over ${json.total_samples || 0} samples.`,
      });
      await Promise.all([refreshAuth(), refreshModels()]);
    } catch (error) {
      if (error.status === 401) { setAuthEmail(null); clearUser(); }
      if (error.fieldErrors && Object.keys(error.fieldErrors).length) {
        setErrors((current) => ({ ...current, [task.id]: error.fieldErrors }));
        const firstInvalid = [...RUN_REQUIRED_FIELDS.map(([name]) => name), "file"].find(
          (name) => error.fieldErrors[name],
        );
        const node = firstInvalid && form.querySelector(`[name="${firstInvalid}"]`);
        if (node && typeof node.focus === "function") node.focus();
      }
      setSubmissionResult({
        ok: false,
        taskLabel: task.label,
        title: error.code === "leaderboard_publication_pending"
          ? "Submission saved, publication pending"
          : "Submission could not be scored",
        text: errorMessage(error, "The submission could not be completed."),
        validation: error.data?.validation || null,
      });
      await Promise.all([refreshAuth(), refreshModels()]);
    } finally {
      setBusyTask("");
    }
  };
  const submissionSteps = [
    [
      "Download questions",
      "Grab each benchmark's question set and submission template from its task card.",
    ],
    [
      "Run your model",
      "Produce one final answer per question_id. The spatial harness runs all six required conditions and creates one upload package.",
    ],
    [
      "Upload and rank",
      "Submit one file per benchmark. Spatial provenance, per sample results, and the aggregate report remain bundled and are published for audit.",
    ],
  ];

  return (
    <WorkspacePage
      eyebrow="Submission workspace"
      title="Evaluate your model"
      description="Download released questions, preserve complete sample coverage, and submit one benchmark response file at a time."
      accountNavigation
    >
      <SubmissionResultDialog result={submissionResult} onClose={() => setSubmissionResult(null)} />
      <div className={ui.sectionBody}>
          {accountNotice && <div className={cn(ui.message, ui.messageSuccess)} role="status">{accountNotice}</div>}
          {authError && <div className={cn(ui.message, ui.messageError)} role="alert">{authError} Refresh this page before submitting.</div>}
          {taskInfoError && <div className={cn(ui.message, ui.messageError)} role="status">{taskInfoError}</div>}
          {authEmail === null && (
            <div className="mb-6 flex flex-col items-start gap-2 border-y border-border-strong py-6">
              <h3 className={ui.heading3}>Sign in to submit</h3>
              <p className="text-sm text-muted">Submissions require a verified MS VISTA account. Create one or sign in to upload model predictions.</p>
              <Button asChild variant="brand"><Link to="/login?next=/submit">Sign in or create account</Link></Button>
            </div>
          )}
          {authEmail && authDisabled && (
            <p className="mb-5 text-sm text-muted">
              Test deployment mode: submissions are recorded as{" "}
              <strong>{authEmail}</strong>. Account signup and email
              verification are disabled.
            </p>
          )}
          {authEmail && !authDisabled && (
            <p className="mb-5 text-sm text-muted">Signed in as <strong>{authEmail}</strong>{quota?.per_benchmark_limit ? ` · ${quota.per_benchmark_limit} submission per benchmark every 24 hours` : ""} · <button type="button" className={ui.linkButton} onClick={async () => { try { await logout(); setAuthEmail(null); setAuthError(""); } catch (error) { setAuthError(errorMessage(error, "Sign out could not be completed.")); } }}>Sign out</button></p>
          )}
          <p className="mb-6 max-w-[70ch] leading-relaxed text-muted">
            Submit one benchmark at a time so each model keeps separate
            perception, visual cognition, and spatial reasoning evidence.
          </p>
          <div className="mb-8 border-y border-border-strong">
            <div className="flex items-start justify-between gap-6 border-b border-border py-5 max-md:flex-col">
              <div>
                <span className={ui.sectionTag}>Submission workflow</span>
                <h3 className={ui.heading3}>Prepare, upload, and score</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className={ui.badge}>{submitTasks.length} tasks</span>
                <span className={ui.badge}>JSONL / ZIP</span>
                <span className={ui.badge}>{authDisabled ? "Open test uploads" : "Authenticated uploads"}</span>
              </div>
            </div>
            <ol className="grid list-none p-0 md:grid-cols-3">
              {submissionSteps.map(([title, body], i) => (
                <li className="border-b border-border py-6 md:border-b-0 md:border-r md:px-7 md:first:pl-0 md:last:border-r-0" key={title}>
                  <span className="mb-4 block text-xs font-semibold text-faint">
                    [{String(i + 1).padStart(2, "0")}]
                  </span>
                  <h3 className={ui.heading3}>{title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
                </li>
              ))}
            </ol>
          </div>
          {authEmail && (
            <section className="mb-8 border-y border-border-strong" aria-labelledby="model-workspace-title">
              <div className="flex items-start justify-between gap-5 border-b border-border px-5 py-5 max-sm:flex-col">
                <div>
                  <span className={ui.sectionTag}>Model workspace</span>
                  <h2 className={ui.heading2} id="model-workspace-title">Choose one model identity</h2>
                  <p className="mt-2 max-w-[68ch] text-sm leading-relaxed text-muted">
                    Every benchmark uploaded with this model is connected to the same leaderboard row. Canonical model details are registered once.
                  </p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  aria-haspopup="dialog"
                  aria-expanded={modelDialogOpen}
                  onClick={openModelDialog}
                >
                  <Plus size={16} aria-hidden="true" />
                  Register a model
                </Button>
              </div>
              <div className="grid md:grid-cols-[minmax(0,1fr)_minmax(280px,0.7fr)]">
                <div className="border-b border-border p-5 md:border-b-0 md:border-r">
                  <Field label="Selected model" required as="div">
                    <Select value={selectedModelId} onValueChange={setSelectedModelId} disabled={modelsStatus === "loading" || models.length === 0}>
                      <SelectTrigger aria-label="Selected model">
                        <SelectValue placeholder={modelsStatus === "loading" ? "Loading models..." : "Select a registered model"} />
                      </SelectTrigger>
                      <SelectContent>
                        {models.map((model) => (
                          <SelectItem value={model.model_id} key={model.model_id}>
                            {model.model_name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  {modelsStatus === "ready" && models.length === 0 && (
                    <p className="mt-3 text-sm text-muted">Register your first model before uploading benchmark outputs.</p>
                  )}
                  {modelRegistryError && <div className={cn(ui.message, ui.messageError, "mt-4")} role="alert">{modelRegistryError}</div>}
                </div>
                <div className="p-5">
                  {selectedModel ? (
                    <div className="flex items-start gap-3">
                      <div className="grid size-10 shrink-0 place-items-center border border-border-strong text-muted" aria-hidden="true"><Box size={18} /></div>
                      <div className="min-w-0">
                        <h3 className={ui.heading3}>{selectedModel.model_name}</h3>
                        <p className="mt-1 text-sm text-muted">{selectedModel.organization} · {selectedModel.access.replaceAll("_", " ")}</p>
                        <p className="mt-2 text-xs text-faint">{Object.keys(selectedModel.benchmarks || {}).length} of {submitTasks.length} benchmarks submitted</p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-muted">No model selected.</p>
                  )}
                </div>
              </div>
              <Dialog open={modelDialogOpen} onOpenChange={handleModelDialogOpenChange}>
                <DialogContent
                  className="max-w-2xl p-0"
                  onInteractOutside={(event) => {
                    const target = event.detail?.originalEvent?.target;
                    if (target instanceof Element && target.closest("[data-model-source-menu]")) {
                      event.preventDefault();
                    }
                  }}
                >
                  <div className="border-b border-border px-6 py-5 pr-16 max-sm:px-5 max-sm:pr-16">
                    <span className={ui.sectionTag}>New model</span>
                    <DialogTitle className={ui.heading2}>Register a model</DialogTitle>
                    <DialogDescription className="mt-2 max-w-[60ch] text-sm leading-relaxed text-muted">
                      Create one model identity, then use it for every benchmark submission that belongs on the same leaderboard row.
                    </DialogDescription>
                  </div>
                <form
                  noValidate
                  onSubmit={registerModel}
                  onChange={(event) => setModelErrors((current) => {
                    if (!current[event.target.name]) return current;
                    const next = { ...current };
                    delete next[event.target.name];
                    return next;
                  })}
                >
                  <div className="p-6 max-sm:p-5">
                    {modelRegistrationError && (
                      <div className={cn(ui.message, ui.messageError, "mb-5")} role="alert">
                        {modelRegistrationError}
                      </div>
                    )}
                    <div className="grid gap-4 sm:grid-cols-2">
                      <Field label="Model name" required error={modelErrors.model_name}>
                        <input type="text" name="model_name" maxLength="255" placeholder="e.g. Qwen2.5 VL 72B" required />
                      </Field>
                      <Field label="Organisation" required error={modelErrors.organization}>
                        <input type="text" name="organization" maxLength="200" placeholder="e.g. Qwen" required />
                      </Field>
                      <Field label="Source status" required as="div" error={modelErrors.access}>
                        <Select name="access" required onValueChange={() => setModelErrors((current) => ({ ...current, access: undefined }))}>
                          <SelectTrigger aria-label="Model source status"><SelectValue placeholder="Select status" /></SelectTrigger>
                          <SelectContent data-model-source-menu>
                            {modelAccessOptions.map(([value, label]) => <SelectItem value={value} key={value}>{label}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      </Field>
                      <Field label="Parameter count (optional)" error={modelErrors.parameter_count}>
                        <input type="text" name="parameter_count" maxLength="80" placeholder="e.g. 72B" />
                      </Field>
                      <Field label="Paper / arXiv link (optional)" className="sm:col-span-2" error={modelErrors.paper_url}>
                        <input type="url" name="paper_url" maxLength="500" placeholder="https://arxiv.org/abs/..." />
                      </Field>
                    </div>
                  </div>
                  <div className="flex justify-end gap-2 border-t border-border px-6 py-4 max-sm:px-5">
                    <DialogClose asChild>
                      <Button type="button" variant="ghost" disabled={registeringModel}>
                        Cancel
                      </Button>
                    </DialogClose>
                    <Button type="submit" variant="primary" disabled={registeringModel}>
                      {registeringModel ? "Registering model..." : "Register model"}
                    </Button>
                  </div>
                </form>
                </DialogContent>
              </Dialog>
            </section>
          )}
          <div className="grid gap-7 lg:gap-9">
            {submitTasks.map((task) => {
              const grading = taskInfo[task.id]?.grading;
              const submissionReady = taskInfo[task.id]?.submission_ready;
              const message = messages[task.id];
              const taskErrors = errors[task.id] || {};
              const benchmarkResult = selectedModel?.benchmarks?.[task.id];
              const benchmarkQuota = quota?.per_benchmark?.[task.id];
              return (
                <Card standalone className="border border-border-strong p-7 max-sm:p-5" key={task.id}>
                  <div className="flex items-start justify-between gap-4 max-sm:flex-col">
                    <h2 className={ui.heading2}>{task.label}</h2>
                    <div className="flex flex-wrap gap-2">
                      <span className={cn(ui.badge, benchmarkResult ? ui.badgePositive : ui.badgeMuted)}>
                        {benchmarkResult
                          ? `${task.harness ? "Published" : "Scored"} ${fmtPct(benchmarkResult.accuracy)}`
                          : "Not submitted"}
                      </span>
                      {benchmarkQuota && (
                        <span className={ui.badge}>{benchmarkQuota.remaining} submission available</span>
                      )}
                    </div>
                  </div>
                  <p className="mt-2 text-sm text-muted">{task.section}</p>
                  {grading && !task.harness && (
                    <p className="mt-2 text-sm text-muted">
                      Graded with deterministic final answer matching against
                      private ground truth.
                    </p>
                  )}
                  {grading && task.harness && (
                    <p className="mt-2 text-sm text-muted">
                      Final outputs are mapped by the pinned Qwen judge. The server verifies public sample coverage, provenance, hashes, and score arithmetic, then publishes the retained evidence without independently grading it again.
                    </p>
                  )}
                  <div className="my-3 flex flex-wrap gap-2 max-sm:[&>*]:w-full">
                    <Button type="button" disabled={busyDownload === `${task.id}:questions` || (task.harness && submissionReady === false)} onClick={() => download(task.id, "questions", `/api/tasks/${task.id}/questions`, `${task.id}_questions.jsonl`)}>
                      {busyDownload === `${task.id}:questions` ? "Downloading questions..." : "Questions (JSONL)"}
                    </Button>
                    <Button type="button" variant="ghost" disabled={busyDownload === `${task.id}:template` || (task.harness && submissionReady === false)} onClick={() => download(task.id, "template", `/api/tasks/${task.id}/template.jsonl`, `${task.id}_template.jsonl`)}>
                      {busyDownload === `${task.id}:template` ? "Downloading template..." : "Template (JSONL)"}
                    </Button>
                    {task.harness && (
                      <>
                        <Button type="button" variant="ghost" disabled={busyDownload === `${task.id}:harness`} onClick={() => download(task.id, "harness", "/api/spatial/harness", "spatial_reasoning_evaluation.zip")}>
                          {busyDownload === `${task.id}:harness` ? "Downloading harness..." : "Harness (ZIP)"}
                        </Button>
                        <Button type="button" variant="ghost" disabled={busyDownload === `${task.id}:manifest` || submissionReady === false} onClick={() => download(task.id, "manifest", "/api/spatial/manifest", "spatial_manifest.json")}>
                          {busyDownload === `${task.id}:manifest` ? "Downloading manifest..." : "Manifest (JSON)"}
                        </Button>
                      </>
                    )}
                  </div>
                  {task.harness && (
                    <p className="text-sm text-muted">
                      Run <code>spatial_reasoning/run_eval.sh</code>, then upload the generated <code>spatial_reasoning_submission.zip</code> package unchanged. Its final answer evidence, aggregate report, manifest, and original ZIP are retained and made public with the leaderboard result.
                    </p>
                  )}
                  {task.harness && submissionReady === false && (
                    <p className="mt-2 text-sm text-muted" role="status">
                      Spatial submissions remain closed until the official 13-dataset bundle is published on this server.
                    </p>
                  )}
                  {message && (
                    <div
                      className={cn(ui.message, message.ok === true && ui.messageSuccess, message.ok === false && ui.messageError)}
                      role={message.ok === false ? "alert" : "status"}
                    >
                      {message.text}
                      <SubmissionIssueDetails issue={message.validation} />
                    </div>
                  )}
                  {authEmail === null && (
                    <p className="mt-3 border-t border-border pt-3 text-sm text-muted">
                      <a href="/login?next=/submit">Sign in</a> to submit
                      predictions for {task.label}.
                    </p>
                  )}
                  {authEmail && (
                  <form
                    className="mt-5"
                    noValidate
                    onChange={(event) =>
                      clearFieldError(task.id, event.target.name)
                    }
                    onSubmit={(event) => submit(event, task)}
                  >
                    <input type="hidden" name="model_id" value={selectedModelId} />
                    {taskErrors.model_id && <span className={cn(ui.fieldError, "mb-4 block")} role="alert">{taskErrors.model_id}</span>}
                    <div className="my-4 border border-border bg-surface-subtle p-4">
                      <div className="mb-4">
                        <span className={ui.sectionTag}>Run metadata</span>
                        <p className="mt-1 text-sm leading-relaxed text-muted">
                          Describe this benchmark run. Canonical model details come from the selected model identity.
                        </p>
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        {task.harness ? (
                          <Field
                            label="Prompt modes"
                            required
                            as="div"
                            error={taskErrors.cot_used}
                          >
                            <div className="flex min-h-11 items-center border border-border bg-surface px-3 text-sm text-muted">
                              <input type="hidden" name="cot_used" value="mixed" />
                              Non CoT and CoT, fixed by the harness
                            </div>
                          </Field>
                        ) : (
                          <Field
                            label="CoT used?"
                            required
                            as="div"
                            error={taskErrors.cot_used}
                          >
                            <Select
                              name="cot_used"
                              required
                              onValueChange={() =>
                                clearFieldError(task.id, "cot_used")
                              }
                            >
                              <SelectTrigger aria-label="CoT used?">
                                <SelectValue placeholder="Select usage" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="no">No</SelectItem>
                                <SelectItem value="yes">Yes</SelectItem>
                                <SelectItem value="mixed">
                                  Mixed / task-specific
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </Field>
                        )}
                        <Field
                          label="Method description"
                          required
                          className="sm:col-span-2"
                          error={taskErrors.method_description}
                        >
                          <textarea
                            name="method_description"
                            rows="6"
                            placeholder="Describe prompting, decoding, image preprocessing, model version, ensembling, retrieval, or any other intervention. Minimum 100 words."
                            required
                          />
                        </Field>
                        {task.harness ? (
                          <Field
                            label="Prompt template"
                            required
                            className="sm:col-span-2"
                            as="div"
                            error={taskErrors.prompt_template}
                          >
                            <div className="flex min-h-11 items-center border border-border bg-surface px-3 text-sm text-muted">
                              <input
                                type="hidden"
                                name="prompt_template"
                                value="Official spatial harness non-CoT and CoT prompts, verified by the packaged run manifest."
                              />
                              Official non CoT and CoT prompts, verified by the packaged run manifest
                            </div>
                          </Field>
                        ) : (
                          <Field
                            label="Prompt template"
                            required
                            className="sm:col-span-2"
                            error={taskErrors.prompt_template}
                          >
                            <textarea
                              name="prompt_template"
                              rows="5"
                              placeholder="Paste the prompt template used to generate predictions for this benchmark."
                              required
                            />
                          </Field>
                        )}
                        <Field
                          label="Changes from previous submission"
                          required
                          className="sm:col-span-2"
                          error={taskErrors.changes_from_previous}
                        >
                          <textarea
                            name="changes_from_previous"
                            rows="5"
                            placeholder="Explain what changed since the previous submission, or state this is the first submission and describe the setup. Minimum 50 words."
                            required
                          />
                        </Field>
                      </div>
                    </div>
                    <Field
                      label={task.harness ? "Spatial submission package (ZIP)" : "Response file (JSONL)"}
                      required
                      error={taskErrors.file}
                    >
                      <FileDropzone
                        name="file"
                        accept={task.harness ? ".zip" : ".jsonl"}
                        required
                        maxBytes={task.harness ? taskInfo[task.id]?.max_upload_bytes : undefined}
                        hint={task.harness
                          ? "ZIP · spatial_reasoning_submission.zip from the completed harness run"
                          : "JSONL · one final answer per question_id"}
                      />
                    </Field>
                    <Button
                      type="submit"
                      variant="primary"
                      disabled={
                        Boolean(busyTask)
                        || !selectedModelId
                        || benchmarkQuota?.remaining === 0
                        || (task.harness && submissionReady === false)
                      }
                    >
                      {busyTask === task.id
                        ? task.harness
                          ? "Validating spatial evidence..."
                          : `Scoring ${task.label}...`
                        : !selectedModelId
                          ? "Select a model to submit"
                          : benchmarkQuota?.remaining === 0
                            ? "Benchmark quota used"
                        : task.harness && submissionReady === false
                          ? "Spatial submissions not open"
                          : task.harness
                            ? "Publish spatial evidence"
                            : `Submit ${task.label}`}
                    </Button>
                  </form>
                  )}
                </Card>
              );
            })}
          </div>
      </div>
    </WorkspacePage>
  );
}
