import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Home } from "@/pages/Home";

const apiMocks = vi.hoisted(() => ({
  getJSON: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getJSON: apiMocks.getJSON,
}));

describe("overview page", () => {
  beforeEach(() => {
    apiMocks.getJSON.mockReset();
    apiMocks.getJSON.mockImplementation((path) => Promise.resolve(
      path === "/api/statistics/overview"
        ? { ranked_models: 2, visual_cognition_models: 2, spatial_models: 1 }
        : { total_samples: 10 },
    ));
  });

  afterEach(() => cleanup());

  it("renders interactive benchmark motifs and submission focused FAQs", async () => {
    const { container } = render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    );

    expect(container.querySelectorAll("[data-overview-motif]")).toHaveLength(3);
    const cube = container.querySelector('[data-overview-motif="cube"]');
    expect(cube.getAttribute("viewBox")).toBe("0 0 144 130");
    const cubeFlap = Array.from(cube.querySelectorAll("g")).find((node) =>
      node.getAttribute("class")?.includes("[transform-box:view-box]"),
    );
    expect(cubeFlap).toBeTruthy();
    const flapPoints = cubeFlap
      .querySelector("path")
      .getAttribute("d")
      .match(/-?\d+(?:\.\d+)?/g)
      .map(Number);
    const points = Array.from({ length: 4 }, (_, index) =>
      flapPoints.slice(index * 2, index * 2 + 2),
    );
    const [hingeStart, hingeEnd] = points;
    const reflectAcrossHinge = ([x, y]) => {
      const dx = hingeEnd[0] - hingeStart[0];
      const dy = hingeEnd[1] - hingeStart[1];
      const projection =
        ((x - hingeStart[0]) * dx + (y - hingeStart[1]) * dy) /
        (dx * dx + dy * dy);
      return [
        2 * (hingeStart[0] + projection * dx) - x,
        2 * (hingeStart[1] + projection * dy) - y,
      ];
    };
    expect(reflectAcrossHinge(points[2])[0]).toBeCloseTo(92, 5);
    expect(reflectAcrossHinge(points[2])[1]).toBeCloseTo(47, 5);
    expect(reflectAcrossHinge(points[3])[0]).toBeCloseTo(52, 5);
    expect(reflectAcrossHinge(points[3])[1]).toBeCloseTo(31, 5);
    expect(cubeFlap.getAttribute("class")).toContain("duration-[1100ms]");
    expect(cubeFlap.getAttribute("class")).not.toContain(
      "[transform:rotate3d",
    );
    fireEvent.pointerEnter(cube.closest("a"));
    expect(cubeFlap.getAttribute("class")).toContain("[transform:rotate3d");
    fireEvent.pointerLeave(cube.closest("a"));
    expect(cubeFlap.getAttribute("class")).not.toContain(
      "[transform:rotate3d",
    );
    fireEvent.focus(cube.closest("a"));
    expect(cubeFlap.getAttribute("class")).toContain("[transform:rotate3d");
    fireEvent.blur(cube.closest("a"));
    expect(cubeFlap.getAttribute("class")).not.toContain(
      "[transform:rotate3d",
    );

    const perspective = container.querySelector(
      '[data-overview-motif="perspective"]',
    );
    expect(perspective.querySelectorAll("rect")).toHaveLength(3);
    expect(perspective.querySelectorAll("line")).toHaveLength(4);

    expect(container.querySelectorAll("#faq details")).toHaveLength(8);
    expect(
      screen.getByText("How do I submit the same model to multiple benchmarks?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("What file should I upload for each benchmark?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("How often can I submit, and can I delete a result?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/deletion does not restore a consumed quota slot/i),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("CoT degrades spatial reasoning"),
    ).toHaveLength(2);
    const visualItemsLabel = screen.getByText(
      "Scored items across the two visual leaderboard tracks",
    );
    await waitFor(() =>
      expect(visualItemsLabel.previousElementSibling).toHaveTextContent("20"),
    );
    const rankedModelsLabel = screen.getByText(
      "Unique models currently ranked across all tracks",
    );
    expect(rankedModelsLabel.previousElementSibling).toHaveTextContent("2");
  });
});
