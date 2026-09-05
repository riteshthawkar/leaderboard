import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
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
    expect(screen.getByRole("link", { name: "Privacy & Cookies" })).toHaveAttribute(
      "href",
      "https://go.microsoft.com/fwlink/?LinkId=521839",
    );
    expect(screen.getByRole("link", { name: "Leaderboard Data Notice" })).toHaveAttribute(
      "href",
      "/privacy",
    );
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

  it("renders every mandatory Microsoft disclosure", async () => {
    apiMocks.getJSON.mockResolvedValue({ status: "healthy", components: {} });
    renderLayout();

    await waitFor(() => expect(apiMocks.getJSON).toHaveBeenCalledWith("/api/health"));
    expect(screen.getByRole("link", { name: "Consumer Health Privacy" })).toHaveAttribute(
      "href",
      "https://go.microsoft.com/fwlink/?linkid=2259814",
    );
    expect(screen.getByRole("link", { name: "Trademarks" })).toHaveAttribute(
      "href",
      "https://www.microsoft.com/trademarks",
    );
    expect(screen.getByRole("link", { name: "Terms of Use" })).toHaveAttribute(
      "href",
      "https://go.microsoft.com/fwlink/?LinkID=206977",
    );
    expect(
      screen.getByLabelText(`Copyright ${new Date().getUTCFullYear()} Microsoft`),
    ).toBeInTheDocument();
  });
});
