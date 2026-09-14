import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router";
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
    expect(screen.getByRole("heading", { name: "Do You See Me: A Multidimensional Benchmark for Evaluating Visual Perception in Multimodal LLMs" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Abstract" })).toBeVisible();
    expect(screen.getByRole("navigation", { name: "Do You See Me paper sections" })).toBeVisible();
    expect(screen.getByText("Visual perception module", { selector: "dd" })).toBeVisible();
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
    expect(screen.getByRole("heading", { name: "Mind's Eye: A Benchmark of Visual Abstraction, Transformation and Composition for Multimodal LLMs" })).toBeVisible();
    expect(screen.getByText("Visual cognition module", { selector: "dd" })).toBeVisible();
    expect(screen.getByText(/does not label every bar numerically/i)).toBeVisible();
    expect(screen.getByRole("columnheader", { name: "Eliminate" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "The task profile exposes where the human model gap sits" })).toBeVisible();
  });

  it("presents the spatial study as a detailed paper project", async () => {
    apiMocks.getJSON
      .mockResolvedValueOnce({ datasets: [] })
      .mockResolvedValueOnce({ leaderboard: [] });

    render(
      <MemoryRouter initialEntries={["/benchmarks/spatial"]}>
        <Routes>
          <Route path="/benchmarks/:slug" element={<Benchmark />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs" })).toBeVisible();
    expect(screen.getByText("Reasoning analysis module", { selector: "dd" })).toBeVisible();
    expect(screen.getByRole("navigation", { name: "Spatial Reasoning paper sections" })).toBeVisible();
    expect(screen.getByRole("link", { name: "Read the paper" })).toHaveAttribute("href", "https://arxiv.org/abs/2604.16060");
  });
});
