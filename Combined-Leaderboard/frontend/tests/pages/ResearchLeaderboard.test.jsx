import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  errorMessage: (error, fallback) => error?.message || fallback,
  getJSON: vi.fn(),
}));

import { getJSON } from "@/lib/api";
import { ResearchLeaderboard } from "@/pages/ResearchLeaderboard";

const visualRows = [
  {
    model_name: "Model Alpha",
    model_meta: { organization: "Org One", params_b: 7 },
    complete: true,
    has_perception: true,
    has_cognition: true,
    vci: 80,
    perception_accuracy: 0.9,
    cognition_accuracy: 0.7,
    perception_task_spread: 0.12,
    cognition_task_spread: 0.08,
    task_spread: 0.1,
    perception_groups: {
      shape_discrimination: { accuracy: 0.92 },
      spatial_relation: { accuracy: 0.81 },
    },
    perception_dimensions: {
      "2D": { accuracy: 0.92 },
      "3D": { accuracy: 0.88 },
    },
    cognition_groups: {
      mental_rotation: { accuracy: 0.72 },
      spatial_visualization: { accuracy: 0.68 },
    },
    cognition_art: {
      abstraction: { accuracy: 0.73 },
      relation: { accuracy: 0.69 },
      transformation: { accuracy: 0.68 },
    },
  },
  {
    model_name: "Model Beta",
    model_meta: { organization: "Org Two", params_b: 14 },
    complete: true,
    has_perception: true,
    has_cognition: true,
    vci: 66,
    perception_accuracy: 0.74,
    cognition_accuracy: 0.58,
    perception_task_spread: 0.2,
    cognition_task_spread: 0.1,
    task_spread: 0.15,
    perception_groups: {
      shape_discrimination: { accuracy: 0.76 },
      spatial_relation: { accuracy: 0.69 },
    },
    perception_dimensions: {
      "2D": { accuracy: 0.78 },
      "3D": { accuracy: 0.7 },
    },
    cognition_groups: {
      mental_rotation: { accuracy: 0.57 },
      spatial_visualization: { accuracy: 0.59 },
    },
    cognition_art: {
      abstraction: { accuracy: 0.6 },
      relation: { accuracy: 0.57 },
      transformation: { accuracy: 0.57 },
    },
  },
];

const defaultSpatialRows = [
  {
    model_name: "Model Alpha",
    model_meta: { organization: "Org One", params_b: 7 },
    accuracy: 0.61,
    macro_accuracy: 0.59,
    accuracy_std: 0.04,
    groups: {
      BLINK: { accuracy: 0.67 },
      MindCube: { accuracy: 0.51 },
    },
    diagnostics: {
      cot_delta: -0.03,
      shortcut_score: 0.18,
      hallucination_resistance: 0.81,
      standard_accuracy: 0.61,
      cot_accuracy: 0.58,
      conditions_present: ["main_noncot", "main_cot", "no_image_noncot", "no_image_plus_noncot"],
    },
  },
  {
    model_name: "Model Gamma",
    model_meta: { organization: "Org Three", params_b: 32 },
    accuracy: 0.52,
    macro_accuracy: 0.5,
    accuracy_std: 0.05,
    groups: {
      BLINK: { accuracy: 0.45 },
      MindCube: { accuracy: 0.55 },
    },
  },
];

const modelReport = {
  model_name: "Model Alpha",
  model_meta: {
    organization: "Org One",
    access: "open_weights",
    base_model: "Alpha Base",
    training_data: "Public training data summary.",
  },
  visual_cognition: {
    vci: 0.8,
    perception_accuracy: 0.9,
    cognition_accuracy: 0.7,
  },
  tasks: {
    do_you_see_me: {
      accuracy: 0.9,
      groups: {},
      model_meta: {
        cot_used: "no",
        method_description: "Perception method only.",
        prompt_template: "Return the perception answer.",
      },
    },
    minds_eye: {
      accuracy: 0.7,
      groups: {},
      model_meta: {
        cot_used: "yes",
        method_description: "Cognition method only.",
        prompt_template: "Reason before returning the cognition answer.",
      },
    },
  },
};

let spatialRows = defaultSpatialRows;

beforeEach(() => {
  spatialRows = defaultSpatialRows;
  getJSON.mockReset();
  getJSON.mockImplementation((url) => {
    if (url === "/api/leaderboard/visual-cognition") {
      return Promise.resolve({ leaderboard: visualRows });
    }
    if (url === "/api/leaderboard/spatial") {
      return Promise.resolve({ leaderboard: spatialRows });
    }
    if (url === "/api/model/Model%20Alpha/report") {
      return Promise.resolve(modelReport);
    }
    return Promise.resolve({});
  });
});

afterEach(() => {
  cleanup();
});

describe("leaderboard filter contracts", () => {
  it("describes leaderboard metrics using their actual aggregation", async () => {
    render(<ResearchLeaderboard />);

    expect(await screen.findByRole("heading", { name: "Combined visual rankings" })).toBeInTheDocument();
    const combinedHumanLabel = screen.getByText("Human VPCI reference");
    expect(within(combinedHumanLabel.parentElement).getByText("87.9%")).toBeInTheDocument();
    expect(within(screen.getByRole("columnheader", { name: /Perception avg/ })).getByRole("button"))
      .toHaveAttribute(
        "title",
        "Do You See Me dimension balanced macro average: task accuracies are averaged within 2D and 3D, then the two dimension averages receive equal weight.",
      );
    expect(within(screen.getByRole("columnheader", { name: /Cognition avg/ })).getByRole("button"))
      .toHaveAttribute(
        "title",
        "Mind's Eye macro average: the eight task accuracies receive equal weight.",
      );
    expect(screen.queryByRole("columnheader", { name: /Params/ })).not.toBeInTheDocument();
    expect(within(screen.getByRole("columnheader", { name: /Spread/ })).getByRole("button"))
      .toHaveAttribute(
        "title",
        "Equal-weight mean of the Do You See Me and Mind's Eye task-score standard deviations, reported in percentage points. Lower values indicate more consistent performance across tasks; this is not repeated-run uncertainty.",
      );
    expect(screen.getByRole("cell", { name: "10.0" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "15.0" })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Model or organization")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/parameters/i)).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "##" })).toHaveAttribute(
      "title",
      "Position after applying the current filters and sort order.",
    );
  });

  it("keeps capability ranking valid when capabilities are selected or cleared", async () => {
    const user = userEvent.setup();
    render(<ResearchLeaderboard />);

    expect(await screen.findByRole("heading", { name: "Combined visual rankings" })).toBeInTheDocument();
    const rankSelect = screen.getByLabelText("Rank by");
    const capabilitySelect = screen.getByLabelText("Rank by capability");

    expect(within(rankSelect).queryByRole("option", { name: /capability/i })).not.toBeInTheDocument();
    expect(within(rankSelect).getByRole("option", { name: "Task spread, lower is better" })).toBeInTheDocument();
    await user.selectOptions(rankSelect, "spread");
    expect(screen.getByRole("columnheader", { name: /Spread/ })).toHaveAttribute("aria-sort", "ascending");
    await user.selectOptions(capabilitySelect, "perception:shape_discrimination");

    expect(rankSelect).toHaveValue("capability");
    expect(within(rankSelect).getByRole("option", { name: "P · Shape Discrimination" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /P · Shape Discrimination/ })).toBeInTheDocument();

    await user.selectOptions(capabilitySelect, "all");
    expect(rankSelect).toHaveValue("vci");
    expect(screen.queryByRole("columnheader", { name: /P · Shape Discrimination/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Do You See Me" }));
    const perceptionHumanLabel = screen.getByText("Human accuracy");
    expect(within(perceptionHumanLabel.parentElement).getByText("95.8%")).toBeInTheDocument();
    expect(within(rankSelect).getAllByRole("option").map((option) => option.value)).toEqual(["perception"]);
    expect(screen.queryByText("Also have cognition")).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Overall avg/ })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /2D avg/ })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /3D avg/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Mind's Eye" }));
    const cognitionHumanLabel = screen.getByText("Human accuracy");
    expect(within(cognitionHumanLabel.parentElement).getByText("80%")).toBeInTheDocument();
    expect(within(screen.getByRole("columnheader", { name: /^A/ })).getByRole("button")).toHaveAttribute(
      "title",
      "Unweighted mean of the Mind's Eye Abstraction task accuracies.",
    );
    expect(screen.getByText("Paper human reference")).toBeInTheDocument();
  });

  it("uses one scope column for a selected spatial benchmark", async () => {
    const user = userEvent.setup();
    render(<ResearchLeaderboard />);

    await user.click(await screen.findByRole("tab", { name: "Spatial Reasoning and Robustness" }));
    const benchmarkSelect = await screen.findByLabelText("Benchmark");
    const rankSelect = screen.getByLabelText("Rank by");

    expect(within(rankSelect).queryByRole("option", { name: /selected dataset/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Advanced filters")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Model type")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("CoT")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Direction")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Diagnostics")).toBeInTheDocument();

    await user.selectOptions(benchmarkSelect, "BLINK");
    expect(rankSelect).toHaveValue("accuracy");
    expect(within(rankSelect).getByRole("option", { name: "Selected scope average" })).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader", { name: /Scope avg\./ })).toHaveLength(1);
    expect(screen.queryByText(/59\.0% ±/)).not.toBeInTheDocument();
  });

  it("changes comparison columns with the selected benchmark scope", async () => {
    const user = userEvent.setup();
    render(<ResearchLeaderboard />);

    await user.click(await screen.findByRole("tab", { name: "Compare Models" }));
    expect(await screen.findByRole("heading", { name: "Selected model comparison" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Spatial/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Do You See Me" }));
    await waitFor(() => {
      expect(screen.getByRole("columnheader", { name: /Perception/ })).toBeInTheDocument();
      expect(screen.queryByRole("columnheader", { name: /VPCI/ })).not.toBeInTheDocument();
      expect(screen.queryByRole("columnheader", { name: /Cognition/ })).not.toBeInTheDocument();
      expect(screen.queryByRole("columnheader", { name: /Spatial/ })).not.toBeInTheDocument();
      expect(screen.queryByRole("columnheader", { name: /CoT delta/ })).not.toBeInTheDocument();
    });
    expect(screen.getByRole("heading", { name: "Do You See Me profile matrix" })).toBeInTheDocument();
  });

  it("hides unavailable spatial comparison scope and columns", async () => {
    spatialRows = [];
    const user = userEvent.setup();
    render(<ResearchLeaderboard />);

    await user.click(await screen.findByRole("tab", { name: "Compare Models" }));
    expect(await screen.findByRole("heading", { name: "Selected model comparison" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Spatial" })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: /Spatial/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: /NI\+\+ resist/ })).not.toBeInTheDocument();
  });

  it("distinguishes an unpublished spatial leaderboard from an empty filter result", async () => {
    spatialRows = [];
    const user = userEvent.setup();
    render(<ResearchLeaderboard />);

    await user.click(await screen.findByRole("tab", { name: "Spatial Reasoning and Robustness" }));
    expect(await screen.findByText("No spatial submissions are published yet.")).toBeVisible();
    expect(screen.queryByText("No models match these filters.")).not.toBeInTheDocument();
  });

  it("keeps benchmark run metadata attached to the matching model report section", async () => {
    const user = userEvent.setup();
    render(<ResearchLeaderboard />);

    await user.click(await screen.findByRole("button", { name: "View model report for Model Alpha" }));

    expect(await screen.findByRole("heading", { name: "Model details" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Do You See Me run details" })).toBeInTheDocument();
    expect(screen.getByText("Perception method only.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Mind's Eye run details" })).toBeInTheDocument();
    expect(screen.getByText("Cognition method only.")).toBeInTheDocument();
  });
});
