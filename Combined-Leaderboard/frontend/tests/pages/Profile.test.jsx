import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  errorMessage: (error, fallback) => error?.message || fallback,
  fetchMe: vi.fn(),
}));

import { fetchMe } from "@/lib/api";
import { Profile } from "@/pages/Profile";

beforeEach(() => {
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
        "1 per benchmark every 24 hours · 1 quota slot remaining across all tracks. Track availability is shown on the submission page.",
      ),
    ).toBeInTheDocument();
  });
});
