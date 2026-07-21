import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  downloadFile: vi.fn(),
  errorMessage: (error, fallback) => error?.message || fallback,
  fetchMe: vi.fn(),
  getJSON: vi.fn(),
  postJSON: vi.fn(),
}));

import { fetchMe, getJSON, postJSON } from "@/lib/api";
import { Submissions } from "@/pages/Submissions";

const submission = {
  submission_id: "submission-owned-1",
  task_id: "do_you_see_me",
  model_name: "Member Model",
  status: "scored",
  moderation_status: "visible",
  accuracy: 0.5,
  macro_accuracy: 0.5,
  accuracy_std: 0.1,
  row_count: 10,
  created_at: "2026-07-13T10:00:00Z",
  submission_export_url: "/api/submissions/submission-owned-1/export.jsonl",
};

function renderPage() {
  return render(
    <MemoryRouter>
      <Submissions />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  fetchMe.mockResolvedValue({ email: "member@example.com" });
  getJSON.mockResolvedValue({ submissions: [submission], count: 1 });
  postJSON.mockReset();
});

afterEach(() => cleanup());

describe("member submission deletion", () => {
  it("confirms deletion, calls the owner endpoint, and removes the row", async () => {
    const user = userEvent.setup();
    postJSON.mockResolvedValue({ success: true, removed_from_leaderboard: true });
    renderPage();

    expect(await screen.findByText("1 record")).toBeInTheDocument();

    await user.click((await screen.findAllByRole("button", { name: "Delete", exact: true }))[0]);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveTextContent("Delete this submission?");
    expect(dialog).toHaveTextContent("audit record remains retained");

    await user.click(screen.getByRole("button", { name: "Delete submission", exact: true }));

    expect(await screen.findByText("Submission deleted")).toBeInTheDocument();
    expect(postJSON).toHaveBeenCalledWith(
      "/api/submissions/submission-owned-1/delete",
      {},
    );
    await user.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() => expect(screen.queryByText("Member Model")).not.toBeInTheDocument());
  });

  it("keeps the row and presents an actionable error when deletion fails", async () => {
    const user = userEvent.setup();
    postJSON.mockRejectedValue(new Error("The submission could not be deleted. Retry shortly."));
    renderPage();

    await user.click((await screen.findAllByRole("button", { name: "Delete", exact: true }))[0]);
    await user.click(screen.getByRole("button", { name: "Delete submission", exact: true }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The submission could not be deleted. Retry shortly.",
    );
    expect(screen.getByRole("button", { name: "Retry deletion" })).toBeInTheDocument();
    expect(screen.getAllByText("Member Model").length).toBeGreaterThan(0);
  });
});
