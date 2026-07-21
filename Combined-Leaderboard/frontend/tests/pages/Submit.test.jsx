import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  clearUser: vi.fn(),
  downloadFile: vi.fn(),
  errorMessage: (error, fallback) => error?.message || fallback,
  fetchMe: vi.fn().mockResolvedValue({
    email: "tester@example.com",
    quota: {
      remaining: 3,
      limit: 3,
      per_benchmark_limit: 1,
      per_benchmark: {
        do_you_see_me: { remaining: 1, limit: 1, used: 0 },
        minds_eye: { remaining: 1, limit: 1, used: 0 },
        spatial: { remaining: 1, limit: 1, used: 0 },
      },
    },
    authDisabled: false,
  }),
  getJSON: vi.fn().mockImplementation((url) => {
    if (url === "/api/models/mine") {
      return Promise.resolve({
        models: [{
          model_id: "mdl_test",
          model_name: "Registered Test Model",
          organization: "Example Lab",
          access: "open_weights",
          benchmarks: {
            do_you_see_me: { accuracy: 0.5, status: "scored" },
          },
        }],
      });
    }
    return Promise.resolve(url.includes("/spatial/")
      ? {
          grading: { method: "judged_jsonl_exact" },
          submission_ready: false,
          required_uploads: ["spatial_reasoning_submission.zip"],
          upload_processing: "in_memory",
          max_upload_bytes: 10 * 1024 * 1024,
        }
      : { grading: { method: "jsonl_exact" } },
    );
  }),
  logout: vi.fn(),
  postFormData: vi.fn(),
  postJSON: vi.fn(),
}));

import { Submit, SubmissionResultDialog } from "@/pages/Submit";
import { postJSON } from "@/lib/api";

afterEach(() => {
  cleanup();
  vi.mocked(postJSON).mockReset();
});


describe("submission result dialog", () => {
  it("shows a scored result and closes from the primary action", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <SubmissionResultDialog
        result={{
          ok: true,
          taskLabel: "Do You See Me",
          title: "Submission scored",
          text: "Scored 54.2% over 2,612 samples.",
        }}
        onClose={onClose}
      />,
    );

    expect(screen.getByRole("dialog")).toHaveTextContent("Submission scored");
    expect(screen.getByRole("dialog")).toHaveTextContent("54.2%");
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("shows actionable validation details returned by the backend", () => {
    render(
      <SubmissionResultDialog
        result={{
          ok: false,
          taskLabel: "Mind's Eye",
          title: "Submission could not be scored",
          text: "Some required model outputs are missing.",
          validation: {
            code: "missing_sample_outputs",
            missing_count: 2,
            missing_question_ids: ["sample-14", "sample-27"],
          },
        }}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByRole("dialog")).toHaveTextContent("Sample outputs missing");
    expect(screen.getByRole("dialog")).toHaveTextContent("Missing: 2");
    expect(screen.getByRole("dialog")).toHaveTextContent("sample-14, sample-27");
  });

  it("shows a non-retry warning when a scored submission awaits publication", () => {
    render(
      <SubmissionResultDialog
        result={{
          ok: false,
          pending: true,
          taskLabel: "Do You See Me",
          title: "Submission saved, publication pending",
          text: "The score is stored. Do not upload it again.",
        }}
        onClose={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveTextContent("Submission saved, publication pending");
    expect(dialog).toHaveTextContent("Do not upload it again");
    expect(screen.getByRole("button", { name: "Done" })).toBeInTheDocument();
  });
});


describe("spatial submission contract", () => {
  it("opens an error dialog when required submission fields are missing", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Submit />
      </MemoryRouter>,
    );

    await user.click(await screen.findByRole("button", { name: "Submit Do You See Me" }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent("Review required fields");
    expect(dialog).toHaveTextContent("Please complete 5 required fields");
    expect(screen.getByText("Method description is required.")).toBeInTheDocument();
  });

  it("requires one ZIP package and blocks uploads until the official bundle is ready", async () => {
    render(
      <MemoryRouter>
        <Submit />
      </MemoryRouter>,
    );

    expect(await screen.findByText(
      "Spatial submissions remain closed until the official 13-dataset bundle is published on this server.",
    )).toBeInTheDocument();

    const spatialPackageInput = document.querySelector('input[name="file"][accept=".zip"]');
    expect(spatialPackageInput).toBeTruthy();
    expect(spatialPackageInput).toBeRequired();
    expect(document.querySelector('input[name="run_manifest"]')).not.toBeInTheDocument();
    const spatialForm = spatialPackageInput.closest("form");
    expect(spatialForm.querySelector('input[name="cot_used"]')).toHaveValue("mixed");
    expect(spatialForm.querySelector('input[name="prompt_template"]')).toHaveValue(
      "Official spatial harness non-CoT and CoT prompts, verified by the packaged run manifest.",
    );
    expect(screen.getByText("spatial_reasoning_submission.zip")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Spatial submissions not open" })).toBeDisabled();
    });
  });

  it("links every benchmark form to the selected registered model", async () => {
    render(
      <MemoryRouter>
        <Submit />
      </MemoryRouter>,
    );

    expect((await screen.findAllByText("Registered Test Model")).length).toBeGreaterThan(0);
    expect(screen.getByText("1 of 3 benchmarks submitted")).toBeInTheDocument();
    expect(screen.getByText("Scored 50.0%")).toBeInTheDocument();
    const linkedInputs = [...document.querySelectorAll('input[name="model_id"]')];
    expect(linkedInputs).toHaveLength(3);
    expect(linkedInputs.every((input) => input.value === "mdl_test")).toBe(true);
    expect(document.querySelector('input[name="model_name"]')).not.toBeInTheDocument();
  });

  it("registers a model in a dialog without deprecated metadata fields", async () => {
    const user = userEvent.setup();
    vi.mocked(postJSON).mockResolvedValueOnce({
      model: {
        model_id: "mdl_dialog",
        model_name: "Dialog Model",
      },
    });

    render(
      <MemoryRouter>
        <Submit />
      </MemoryRouter>,
    );

    const trigger = await screen.findByRole("button", { name: "Register a model" });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await user.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveTextContent("Create one model identity");
    expect(dialog.querySelector('input[name="base_model"]')).not.toBeInTheDocument();
    expect(dialog.querySelector('[name="training_data"]')).not.toBeInTheDocument();

    await user.type(
      within(dialog).getByRole("textbox", { name: /Model name/i }),
      "Dialog Model",
    );
    await user.type(
      within(dialog).getByRole("textbox", { name: /Organisation/i }),
      "Example Lab",
    );
    await user.click(within(dialog).getByRole("combobox", { name: "Model source status" }));
    await user.click(await screen.findByRole("option", { name: "Open weights" }));
    await user.type(
      within(dialog).getByRole("textbox", { name: /Parameter count/i }),
      "7B",
    );
    await user.type(
      within(dialog).getByRole("textbox", { name: /Paper.*arXiv link/i }),
      "https://example.com/paper",
    );
    await user.click(within(dialog).getByRole("button", { name: "Register model" }));

    await waitFor(() => {
      expect(postJSON).toHaveBeenCalledWith("/api/models", {
        model_name: "Dialog Model",
        organization: "Example Lab",
        access: "open_weights",
        parameter_count: "7B",
        paper_url: "https://example.com/paper",
      });
    });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(trigger).toHaveAttribute("aria-expanded", "false");
  });
});
