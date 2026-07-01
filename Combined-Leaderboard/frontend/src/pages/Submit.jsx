import { Fragment, useEffect, useState } from "react";
import { PageHero } from "@/components/Hero";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FileDropzone } from "@/components/FileDropzone";
import { getJSON, readUser } from "@/lib/api";
import { fmtMeanStd, fmtPct } from "@/lib/utils";
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
  const organization = fieldValue(formData, "organization");
  const access = fieldValue(formData, "model_access");
  return {
    org: organization,
    organization,
    type: access,
    access,
    parameter_count: fieldValue(formData, "parameter_count"),
    base_model: fieldValue(formData, "base_model"),
    training_data: fieldValue(formData, "training_data"),
    method_description: fieldValue(formData, "method_description"),
    cot_used: fieldValue(formData, "cot_used"),
    prompt_template: fieldValue(formData, "prompt_template"),
    changes_from_previous: fieldValue(formData, "changes_from_previous"),
    paper_url: fieldValue(formData, "paper_url"),
    submission_track: task.id,
  };
}

// Required text/select fields validated client-side (name -> human label).
const REQUIRED_FIELDS = [
  ["model_name", "Model name"],
  ["organization", "Organisation"],
  ["model_access", "Source status"],
  ["parameter_count", "Parameter count"],
  ["base_model", "Base model"],
  ["training_data", "Training data and fine-tuning"],
  ["cot_used", "CoT used?"],
  ["method_description", "Method description"],
  ["prompt_template", "Prompt template"],
  ["changes_from_previous", "Changes from previous submission"],
];

function Field({ label, required = false, error, className = "", children, as: Tag = "label" }) {
  return (
    <Tag className={`field ${className} ${error ? "has-error" : ""}`.trim()}>
      <span>
        {label} {required && <em>*</em>}
      </span>
      {children}
      {error && (
        <span className="field-error" role="alert">
          {error}
        </span>
      )}
    </Tag>
  );
}

export function Submit() {
  const user = readUser();
  const [taskInfo, setTaskInfo] = useState({});
  const [messages, setMessages] = useState({});
  const [errors, setErrors] = useState({});
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
    Promise.all(
      submitTasks.map((task) =>
        getJSON(`/api/tasks/${task.id}/info`).catch(() => ({})),
      ),
    ).then((infos) =>
      setTaskInfo(
        Object.fromEntries(submitTasks.map((task, i) => [task.id, infos[i]])),
      ),
    );
  }, []);
  const submit = async (event, task) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    data.set("model_name", fieldValue(data, "model_name"));
    const modelMeta = collectModelMeta(data, task);
    const fieldErrors = {};
    for (const [name, label] of REQUIRED_FIELDS) {
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
      fieldErrors.file = "A response file (JSON or CSV) is required.";
    }
    if (Object.keys(fieldErrors).length) {
      setErrors((current) => ({ ...current, [task.id]: fieldErrors }));
      const count = Object.keys(fieldErrors).length;
      setMessages((current) => ({
        ...current,
        [task.id]: {
          ok: false,
          text: `Please complete ${count} required field${count > 1 ? "s" : ""} highlighted below.`,
        },
      }));
      const firstInvalid = [...REQUIRED_FIELDS.map(([name]) => name), "file"].find(
        (name) => fieldErrors[name],
      );
      const node = firstInvalid && form.querySelector(`[name="${firstInvalid}"]`);
      if (node && typeof node.focus === "function") node.focus();
      return;
    }
    setErrors((current) => ({ ...current, [task.id]: {} }));
    data.set("model_meta", JSON.stringify(modelMeta));
    setMessages((current) => ({ ...current, [task.id]: { text: "Scoring…" } }));
    try {
      const headers = user?.api_token
        ? { Authorization: `Bearer ${user.api_token}` }
        : undefined;
      const response = await fetch(`/api/tasks/${task.id}/submit`, {
        method: "POST",
        body: data,
        headers,
      });
      const json = await response.json().catch(() => ({}));
      if (!response.ok)
        throw new Error(json.error || `HTTP ${response.status}`);
      const avg =
        json.macro_accuracy != null
          ? ` · avg ${fmtMeanStd(json.macro_accuracy, json.accuracy_std)}`
          : "";
      setMessages((current) => ({
        ...current,
        [task.id]: {
          ok: true,
          text: `Scored: ${fmtPct(json.accuracy)} over ${json.total_samples || 0} samples${avg}.`,
        },
      }));
    } catch (error) {
      setMessages((current) => ({
        ...current,
        [task.id]: { ok: false, text: `Error: ${error.message}` },
      }));
    }
  };
  const submissionSteps = [
    [
      "Download questions",
      "Grab each benchmark's question set and submission template from its task card.",
    ],
    [
      "Run your model",
      "Produce one answer per question id. Spatial submissions can use the harness for standard, CoT, and no-image conditions.",
    ],
    [
      "Upload and rank",
      "Submit per benchmark. Scores are computed with the paper-faithful grader and posted to the leaderboard.",
    ],
  ];

  return (
    <>
      <PageHero
        eyebrow="Submit"
        title="Evaluate your model"
        subtitle="Submit predictions per benchmark for offline, paper-faithful scoring. Download a task's questions, run your model, then upload a JSON or CSV response file. Results post to the leaderboard immediately."
      />
      <section className="section leaderboard-section submit-section">
        <div className="container">
          <p className="track-note">
            Submit one benchmark at a time so each model keeps separate
            perception, imagery, and spatial reasoning evidence.
          </p>
          <div className="control-deck submit-guide">
            <div className="control-deck-head">
              <div>
                <span className="deck-eyebrow">Submission workflow</span>
                <h3>Prepare, upload, and score</h3>
              </div>
              <div className="deck-meta">
                <span>{submitTasks.length} tasks</span>
                <span>JSON or CSV</span>
                <span>Authenticated uploads</span>
              </div>
            </div>
            <div className="grid cols-3 ruled">
              {submissionSteps.map(([title, body], i) => (
                <Card key={title}>
                  <span className="card-n">
                    [{String(i + 1).padStart(2, "0")}]
                  </span>
                  <h3>{title}</h3>
                  <p className="muted small">{body}</p>
                </Card>
              ))}
            </div>
          </div>
          <div className="submit-task-grid">
            {submitTasks.map((task, index) => {
              const grading = taskInfo[task.id]?.grading;
              const message = messages[task.id];
              const taskErrors = errors[task.id] || {};
              return (
                <Fragment key={task.id}>
                  {index > 0 && (
                    <div className="submit-card-divider" aria-hidden="true" />
                  )}
                  <Card standalone className="submit-card">
                  <h2>{task.label}</h2>
                  <p className="muted small">{task.section}</p>
                  {grading && (
                    <p className="muted small grading-note">
                      Graded as in the paper via{" "}
                      {grading.method === "judge"
                        ? "LLM-as-judge"
                        : "LLM answer-extractor"}{" "}
                      <code>{grading.judge_model || ""}</code>
                    </p>
                  )}
                  <div className="dl-row">
                    <Button asChild>
                      <a href={`/api/tasks/${task.id}/questions`}>
                        Questions (JSON)
                      </a>
                    </Button>
                    <Button asChild variant="ghost">
                      <a href={`/api/tasks/${task.id}/template.json`}>
                        Template (JSON)
                      </a>
                    </Button>
                    <Button asChild variant="ghost">
                      <a href={`/api/tasks/${task.id}/template.csv`}>
                        Template (CSV)
                      </a>
                    </Button>
                    {task.harness && (
                      <Button asChild variant="ghost">
                        <a href="/api/spatial/manifest">Manifest (JSON)</a>
                      </Button>
                    )}
                  </div>
                  {task.harness && (
                    <p className="muted small">
                      Run the harness in <code>spatial_harness/</code> to
                      produce the response file.
                    </p>
                  )}
                  <form
                    className="task-submit-form"
                    noValidate
                    onChange={(event) =>
                      clearFieldError(task.id, event.target.name)
                    }
                    onSubmit={(event) => submit(event, task)}
                  >
                    <Field
                      label="Model name"
                      required
                      error={taskErrors.model_name}
                    >
                      <input
                        type="text"
                        name="model_name"
                        maxLength="255"
                        placeholder="e.g. GPT-4o"
                        required
                      />
                    </Field>
                    <div className="submission-metadata">
                      <div className="metadata-head">
                        <span className="deck-eyebrow">Required metadata</span>
                        <p className="muted small">
                          Matches the proposal's submission audit fields.
                          Method descriptions need 100+ words; change logs need
                          50+ words.
                        </p>
                      </div>
                      <div className="metadata-grid">
                        <Field
                          label="Organisation"
                          required
                          error={taskErrors.organization}
                        >
                          <input
                            type="text"
                            name="organization"
                            maxLength="120"
                            placeholder="e.g. Microsoft Research"
                            required
                          />
                        </Field>
                        <Field
                          label="Source status"
                          required
                          as="div"
                          error={taskErrors.model_access}
                        >
                          <Select
                            name="model_access"
                            required
                            onValueChange={() =>
                              clearFieldError(task.id, "model_access")
                            }
                          >
                            <SelectTrigger aria-label="Source status">
                              <SelectValue placeholder="Select status" />
                            </SelectTrigger>
                            <SelectContent>
                              {modelAccessOptions.map(([value, label]) => (
                                <SelectItem value={value} key={value}>
                                  {label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </Field>
                        <Field
                          label="Parameter count"
                          required
                          error={taskErrors.parameter_count}
                        >
                          <input
                            type="text"
                            name="parameter_count"
                            maxLength="80"
                            placeholder="e.g. 72B, undisclosed"
                            required
                          />
                        </Field>
                        <Field
                          label="Base model"
                          required
                          error={taskErrors.base_model}
                        >
                          <input
                            type="text"
                            name="base_model"
                            maxLength="160"
                            placeholder="e.g. GPT-4o base"
                            required
                          />
                        </Field>
                        <Field
                          label="Training data and fine-tuning"
                          required
                          className="field-wide"
                          error={taskErrors.training_data}
                        >
                          <textarea
                            name="training_data"
                            rows="4"
                            placeholder="Summarize pretraining, multimodal data, synthetic data, and fine-tuning relevant to this submission."
                            required
                          />
                        </Field>
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
                        <Field label="Paper / arXiv link">
                          <input
                            type="url"
                            name="paper_url"
                            placeholder="https://arxiv.org/abs/..."
                          />
                        </Field>
                        <Field
                          label="Method description"
                          required
                          className="field-wide"
                          error={taskErrors.method_description}
                        >
                          <textarea
                            name="method_description"
                            rows="6"
                            placeholder="Describe prompting, decoding, image preprocessing, model version, ensembling, retrieval, or any other intervention. Minimum 100 words."
                            required
                          />
                        </Field>
                        <Field
                          label="Prompt template"
                          required
                          className="field-wide"
                          error={taskErrors.prompt_template}
                        >
                          <textarea
                            name="prompt_template"
                            rows="5"
                            placeholder="Paste the prompt template used to generate predictions for this benchmark."
                            required
                          />
                        </Field>
                        <Field
                          label="Changes from previous submission"
                          required
                          className="field-wide"
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
                      label="Response file (JSON/CSV)"
                      required
                      error={taskErrors.file}
                    >
                      <FileDropzone
                        name="file"
                        accept=".json,.csv"
                        required
                        hint="JSON or CSV · one answer per question id"
                      />
                    </Field>
                    <Button type="submit" variant="primary">
                      Submit {task.label}
                    </Button>
                    {message && (
                      <div
                        className={`form-msg ${message.ok ? "ok" : "err"}`}
                        role="status"
                      >
                        {message.text}
                      </div>
                    )}
                  </form>
                </Card>
                </Fragment>
              );
            })}
          </div>
        </div>
      </section>
    </>
  );
}
