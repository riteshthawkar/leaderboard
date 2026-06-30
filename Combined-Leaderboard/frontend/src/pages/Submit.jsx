import { useEffect, useState } from "react";
import { PageHero } from "@/components/Hero";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
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

export function Submit() {
  const user = readUser();
  const [taskInfo, setTaskInfo] = useState({});
  const [messages, setMessages] = useState({});
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
    if (wordCount(modelMeta.method_description) < 100) {
      setMessages((current) => ({ ...current, [task.id]: { ok: false, text: "Method description must be at least 100 words." } }));
      return;
    }
    if (wordCount(modelMeta.changes_from_previous) < 50) {
      setMessages((current) => ({ ...current, [task.id]: { ok: false, text: "Changes from previous submission must be at least 50 words." } }));
      return;
    }
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
            {submitTasks.map((task) => {
              const grading = taskInfo[task.id]?.grading;
              const message = messages[task.id];
              return (
                <Card standalone className="submit-card" key={task.id}>
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
                  </div>
                  {task.harness && (
                    <>
                      <div className="dl-row">
                        <Button asChild variant="ghost">
                          <a href="/api/spatial/manifest">Manifest (JSON)</a>
                        </Button>
                      </div>
                      <p className="muted small">
                        Run the harness in <code>spatial_harness/</code> to
                        produce the response file.
                      </p>
                    </>
                  )}
                  <form
                    className="task-submit-form"
                    onSubmit={(event) => submit(event, task)}
                  >
                    <label className="field">
                      <span>
                        Model name <em>*</em>
                      </span>
                      <input
                        type="text"
                        name="model_name"
                        maxLength="255"
                        placeholder="e.g. GPT-4o"
                        required
                      />
                    </label>
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
                        <label className="field">
                          <span>
                            Organisation <em>*</em>
                          </span>
                          <input
                            type="text"
                            name="organization"
                            maxLength="120"
                            placeholder="e.g. Microsoft Research"
                            required
                          />
                        </label>
                        <label className="field">
                          <span>
                            Source status <em>*</em>
                          </span>
                          <select name="model_access" defaultValue="" required>
                            <option value="" disabled>
                              Select status
                            </option>
                            {modelAccessOptions.map(([value, label]) => (
                              <option value={value} key={value}>
                                {label}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="field">
                          <span>
                            Parameter count <em>*</em>
                          </span>
                          <input
                            type="text"
                            name="parameter_count"
                            maxLength="80"
                            placeholder="e.g. 72B, undisclosed"
                            required
                          />
                        </label>
                        <label className="field">
                          <span>
                            Base model <em>*</em>
                          </span>
                          <input
                            type="text"
                            name="base_model"
                            maxLength="160"
                            placeholder="e.g. GPT-4o base"
                            required
                          />
                        </label>
                        <label className="field field-wide">
                          <span>
                            Training data and fine-tuning <em>*</em>
                          </span>
                          <textarea
                            name="training_data"
                            rows="4"
                            placeholder="Summarize pretraining, multimodal data, synthetic data, and fine-tuning relevant to this submission."
                            required
                          />
                        </label>
                        <label className="field">
                          <span>
                            CoT used? <em>*</em>
                          </span>
                          <select name="cot_used" defaultValue="" required>
                            <option value="" disabled>
                              Select usage
                            </option>
                            <option value="no">No</option>
                            <option value="yes">Yes</option>
                            <option value="mixed">Mixed / task-specific</option>
                          </select>
                        </label>
                        <label className="field">
                          <span>Paper / arXiv link</span>
                          <input
                            type="url"
                            name="paper_url"
                            placeholder="https://arxiv.org/abs/..."
                          />
                        </label>
                        <label className="field field-wide">
                          <span>
                            Method description <em>*</em>
                          </span>
                          <textarea
                            name="method_description"
                            rows="6"
                            placeholder="Describe prompting, decoding, image preprocessing, model version, ensembling, retrieval, or any other intervention. Minimum 100 words."
                            required
                          />
                        </label>
                        <label className="field field-wide">
                          <span>
                            Prompt template <em>*</em>
                          </span>
                          <textarea
                            name="prompt_template"
                            rows="5"
                            placeholder="Paste the prompt template used to generate predictions for this benchmark."
                            required
                          />
                        </label>
                        <label className="field field-wide">
                          <span>
                            Changes from previous submission <em>*</em>
                          </span>
                          <textarea
                            name="changes_from_previous"
                            rows="5"
                            placeholder="Explain what changed since the previous submission, or state this is the first submission and describe the setup. Minimum 50 words."
                            required
                          />
                        </label>
                      </div>
                    </div>
                    <label className="field">
                      <span>
                        Response file (JSON/CSV) <em>*</em>
                      </span>
                      <input
                        type="file"
                        name="file"
                        accept=".json,.csv"
                        required
                      />
                    </label>
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
              );
            })}
          </div>
        </div>
      </section>
    </>
  );
}
