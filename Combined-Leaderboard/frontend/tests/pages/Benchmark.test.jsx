import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { cleanup, render, screen } from "@testing-library/react";
import { Benchmark } from "@/pages/Benchmark";


const apiMocks = vi.hoisted(() => ({
  getJSON: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getJSON: apiMocks.getJSON,
  apiUrl: (path) => path,
  errorMessage: (error, fallback = "The action could not be completed.") => error?.message || fallback,
}));


describe("benchmark routing", () => {
  beforeEach(() => {
    apiMocks.getJSON.mockReset();
    apiMocks.getJSON.mockResolvedValue({ leaderboard: [] });
  });

  afterEach(() => cleanup());

  it("renders the not-found state for an unknown benchmark slug", () => {
    render(
      <MemoryRouter initialEntries={["/benchmarks/not-real"]}>
        <Routes>
          <Route path="/benchmarks/:slug" element={<Benchmark />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByRole("heading", { name: "Page not found" })).toBeVisible();
    expect(apiMocks.getJSON).not.toHaveBeenCalled();
  });

  it("renders the complete DYSM rotation matrix and bounded paper gap", async () => {
    render(
      <MemoryRouter initialEntries={["/benchmarks/do-you-see-me"]}>
        <Routes>
          <Route path="/benchmarks/:slug" element={<Benchmark />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "Rotation detection depends on both object size and angle" })).toBeVisible();
    expect(screen.getByText(">45.8 point gap")).toBeVisible();
    expect(screen.getByRole("columnheader", { name: "4°" })).toBeVisible();
    expect(screen.getByRole("rowheader", { name: "1P · 14 px" })).toBeVisible();
  });

  it("renders source qualified Mind's Eye prompt effects and task references", async () => {
    render(
      <MemoryRouter initialEntries={["/benchmarks/minds-eye"]}>
        <Routes>
          <Route path="/benchmarks/:slug" element={<Benchmark />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "Prompt effects change direction across ART" })).toBeVisible();
    expect(screen.getByText(/does not label every bar numerically/i)).toBeVisible();
    expect(screen.getByRole("columnheader", { name: "Eliminate" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "The task profile exposes where the human model gap sits" })).toBeVisible();
  });
});
