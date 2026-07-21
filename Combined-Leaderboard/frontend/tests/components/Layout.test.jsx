import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";


const apiMocks = vi.hoisted(() => ({
  getJSON: vi.fn(),
  fetchMe: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  IS_STATIC_DEMO: false,
  getJSON: apiMocks.getJSON,
  fetchMe: apiMocks.fetchMe,
  logout: vi.fn(),
  errorMessage: (error, fallback) => error?.message || fallback,
}));


function renderLayout() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<main>Overview content</main>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}


describe("service warning relevance", () => {
  beforeEach(() => {
    apiMocks.fetchMe.mockResolvedValue(null);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("does not show a global account warning for the pending spatial bundle", async () => {
    apiMocks.getJSON.mockRejectedValue({
      status: 503,
      data: {
        status: "degraded",
        components: {
          auth: "healthy",
          email: "healthy",
          database: "healthy",
          spatial_bundle: "unhealthy",
        },
      },
    });
    renderLayout();

    await waitFor(() => expect(apiMocks.getJSON).toHaveBeenCalledWith("/api/health"));
    expect(await screen.findByText("Online")).toBeVisible();
    expect(screen.getByText("Overview content")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Privacy" })).toHaveAttribute("href", "/privacy");
  });

  it("shows a specific warning when authentication readiness fails", async () => {
    apiMocks.getJSON.mockRejectedValue({
      status: 503,
      data: {
        status: "degraded",
        components: {
          auth: "unhealthy",
          email: "healthy",
          database: "healthy",
          spatial_bundle: "unhealthy",
        },
      },
    });
    renderLayout();

    expect(await screen.findByText(/Authentication configuration is incomplete/)).toBeVisible();
  });

  it("shows a specific warning when stored submission invariants fail", async () => {
    apiMocks.getJSON.mockRejectedValue({
      status: 503,
      data: {
        status: "degraded",
        components: {
          auth: "healthy",
          email: "healthy",
          database: "healthy",
          submission_store: "unhealthy",
          leaderboard_store: "healthy",
        },
      },
    });
    renderLayout();

    expect(await screen.findByText(/Stored submission records failed/)).toBeVisible();
  });
});
