import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  clearUser: vi.fn(),
  deleteJSON: vi.fn(),
  downloadFile: vi.fn(),
  errorMessage: (error, fallback) => error?.message || fallback,
  fetchMe: vi.fn(),
}));

import { clearUser, deleteJSON, downloadFile, fetchMe } from "@/lib/api";
import { Profile } from "@/pages/Profile";

beforeEach(() => {
  vi.clearAllMocks();
  fetchMe.mockResolvedValue({
    email: "member@example.com",
    emailVerified: true,
    provider: "password",
    createdAt: "2026-07-13T10:00:00Z",
    isAdmin: false,
    authDisabled: false,
    quota: {
      limit: 3,
      remaining: 1,
      per_benchmark_limit: 1,
    },
  });
  downloadFile.mockResolvedValue("account-data.json");
  deleteJSON.mockResolvedValue({ status: "ok", anonymized: true });
});

afterEach(() => cleanup());

describe("profile quota summary", () => {
  it("uses singular quota wording and separates quota from track availability", async () => {
    render(
      <MemoryRouter>
        <Profile />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(
        "1 per module every 24 hours · 1 quota slot remaining across the framework. Module availability is shown on the submission page.",
      ),
    ).toBeInTheDocument();
  });

  it("downloads the signed-in user's account data", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Profile />
      </MemoryRouter>,
    );

    await user.click(await screen.findByRole("button", { name: "Download account data" }));

    expect(downloadFile).toHaveBeenCalledWith(
      "/api/auth/me/export",
      "ms-vista-account-data.json",
    );
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Your account data was downloaded as account-data.json.",
    );
  });

  it("requires explicit confirmation before deleting an account", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Profile />
      </MemoryRouter>,
    );

    await user.click(await screen.findByRole("button", { name: "Delete account" }));
    const dialog = screen.getByRole("dialog");
    const confirmButton = within(dialog).getByRole("button", { name: "Delete account" });
    expect(dialog).toHaveTextContent("Personal information inside uploaded files is not automatically scrubbed");
    expect(dialog).toHaveTextContent("older backups remain until they expire");
    expect(confirmButton).toBeDisabled();

    await user.type(within(dialog).getByLabelText("Type DELETE to confirm"), "DELETE");
    expect(confirmButton).toBeEnabled();
    await user.click(confirmButton);

    expect(deleteJSON).toHaveBeenCalledWith("/api/auth/me");
    expect(clearUser).toHaveBeenCalledOnce();
    expect(await within(dialog).findByText("Account deleted")).toBeInTheDocument();
  });
});
