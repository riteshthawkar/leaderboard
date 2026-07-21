import { useEffect, useId, useMemo, useRef, useState } from "react";
import { ChevronDown, Download, ExternalLink, X } from "lucide-react";
import { BarChart, CapabilityRadar, ScatterChart, chartPalette } from "@/components/Charts";
import { TabBar } from "@/components/ui/tabs";
import { apiUrl, errorMessage, getJSON } from "@/lib/api";
import { cn, fmtDelta, fmtPct, fmtVci, modelType, prettyLabel } from "@/lib/utils";
import { ui } from "@/lib/styles";

const trackTabs = [
  { id: "vc", label: "Visual Perception and Cognition Index" },
  { id: "spatial", label: "Spatial Reasoning and Robustness" },
  { id: "compare", label: "Compare Models" },
];

const DEFAULT_COMPARE_MODELS = 4;
const DEFAULT_CHART_MODELS = 4;

const mindsEyeArtByCapability = {
  analogical_reasoning: "abstraction",
  visual_relation_abstraction: "abstraction",
  hierarchical_reasoning: "abstraction",
  conceptual_slippage: "relation",
  dynamic_reasoning: "relation",
  symmetry_analysis: "relation",
  mental_composition: "transformation",
  mental_rotation: "transformation",
  spatial_visualization: "transformation",
};

const difficultyLevels = ["easy", "medium", "hard"];
const artDimensions = ["abstraction", "relation", "transformation"];
const perceptionCapabilityOrder = [
  "shape_discrimination",
  "feature_binding",
  "form_discrimination",
  "form_constancy",
  "spatial_relation",
  "figure_ground",
  "visual_closure",
];
const cognitionCapabilityOrder = [
  "analogical_reasoning",
  "hierarchical_reasoning",
  "dynamic_reasoning",
  "conceptual_slippage",
  "symmetry_analysis",
  "mental_rotation",
  "spatial_visualization",
  "mental_composition",
];
const artShortLabel = {
  abstraction: "A",
  relation: "R",
  transformation: "T",
};

const visualHumanReference = {
  all: { value: "87.9%", label: "Human VPCI reference" },
  do_you_see_me: { value: "95.8%", label: "Human accuracy" },
  minds_eye: { value: "80%", label: "Human accuracy" },
};

const mindsEyePaperReferenceRows = [
  {
    model_name: "Paper human reference",
    reference: true,
    cognition_groups: {
      analogical_reasoning: { accuracy: 0.68 },
      hierarchical_reasoning: { accuracy: 0.88 },
      dynamic_reasoning: { accuracy: 0.812 },
      conceptual_slippage: { accuracy: 0.87 },
      symmetry_analysis: { accuracy: 0.78 },
      mental_rotation: { accuracy: 0.81 },
      spatial_visualization: { accuracy: 0.801 },
      mental_composition: { accuracy: 0.82 },
    },
  },
  {
    model_name: "Paper random choice",
    reference: true,
    cognition_groups: {
      analogical_reasoning: { accuracy: 0.16 },
      hierarchical_reasoning: { accuracy: 0.25 },
      dynamic_reasoning: { accuracy: 0.25 },
      conceptual_slippage: { accuracy: 0.16 },
      symmetry_analysis: { accuracy: 0.25 },
      mental_rotation: { accuracy: 0.25 },
      spatial_visualization: { accuracy: 0.25 },
      mental_composition: { accuracy: 0.25 },
    },
  },
];

const leaderboardSurfaceClasses = [
  "[&_.tab-panel]:block",
  "[&_.primary-controls]:mt-4 [&_.primary-controls]:grid [&_.primary-controls]:grid-cols-[repeat(auto-fit,minmax(190px,1fr))] [&_.primary-controls]:gap-3",
  "[&_.primary-controls.narrow]:grid-cols-[minmax(220px,360px)] max-sm:[&_.primary-controls.narrow]:grid-cols-1",
  "[&_.table-wrap]:max-h-[1200px] [&_.table-wrap]:w-full [&_.table-wrap]:overflow-auto [&_.table-wrap]:border [&_.table-wrap]:border-border-strong",
  "[&_.leaderboard-table]:mb-6",
  "[&_.lb-table]:w-full [&_.lb-table]:min-w-[640px] [&_.lb-table]:border-collapse [&_.lb-table]:text-sm",
  "[&_.lb-table_th]:sticky [&_.lb-table_th]:top-0 [&_.lb-table_th]:z-10 [&_.lb-table_th]:border-b [&_.lb-table_th]:border-border-strong [&_.lb-table_th]:bg-surface-subtle [&_.lb-table_th]:px-4 [&_.lb-table_th]:py-4 [&_.lb-table_th]:text-left [&_.lb-table_th]:text-xs [&_.lb-table_th]:font-semibold [&_.lb-table_th]:uppercase [&_.lb-table_th]:text-faint",
  "[&_.lb-table_td]:border-b [&_.lb-table_td]:border-border [&_.lb-table_td]:px-4 [&_.lb-table_td]:py-3 [&_.lb-table_td]:text-left",
  "[&_.lb-table_tbody_tr:last-child_td]:border-b-0",
  "[&_.rank-col]:sticky [&_.rank-col]:left-0 [&_.rank-col]:z-[6] [&_.rank-col]:w-16 [&_.rank-col]:border-r [&_.rank-col]:border-border-strong [&_tbody_.rank-col]:bg-background",
  "[&_.model-col]:sticky [&_.model-col]:left-16 [&_.model-col]:z-[5] [&_.model-col]:w-[160px] [&_.model-col]:min-w-[160px] [&_.model-col]:max-w-[160px] [&_tbody_.model-col]:bg-background",
  "[&_.model-col-first]:!left-0",
  "[&_.lb-table_thead_.rank-col]:z-30 [&_.lb-table_thead_.model-col]:z-20",
  "[&_.clickable:hover_.rank-col]:bg-brand-soft [&_.clickable:hover_.model-col]:bg-brand-soft",
  "[&_.num]:text-right [&_.num]:tabular-nums",
  "[&_.vci-val]:font-bold [&_.vci-val]:text-brand-strong",
  "[&_.pos]:text-positive [&_.neg]:text-negative",
  "[&_.clickable]:cursor-pointer [&_.clickable:hover]:bg-brand-soft",
  "[&_.empty-row]:py-12 [&_.empty-row]:text-center [&_.empty-row]:text-muted",
  "[&_.chip]:inline-flex [&_.chip]:min-h-7 [&_.chip]:items-center [&_.chip]:border [&_.chip]:border-border [&_.chip]:bg-surface-subtle [&_.chip]:px-2.5 [&_.chip]:py-1 [&_.chip]:text-xs [&_.chip]:text-muted",
  "[&_.dashboard-grid]:grid [&_.dashboard-grid]:grid-cols-1 [&_.dashboard-grid]:border-l [&_.dashboard-grid]:border-t [&_.dashboard-grid]:border-border lg:[&_.dashboard-grid.two-col]:grid-cols-2",
  "[&_.viz-card]:min-w-0 [&_.viz-card]:border-b [&_.viz-card]:border-r [&_.viz-card]:border-border [&_.viz-card]:p-6",
  "lg:[&_.viz-card.wide]:col-span-2",
  "[&_.viz-card_h3]:mb-3 [&_.viz-card_h3]:font-display [&_.viz-card_h3]:text-lg [&_.viz-card_h3]:font-bold",
].join(" ");

const visualMetricOptions = [
  { value: "vci", label: "VPCI score" },
  { value: "perception", label: "Perception macro average" },
  { value: "imagery", label: "Cognition macro average" },
  { value: "gap", label: "Perception cognition gap, lower is better" },
  { value: "spread", label: "Task spread, lower is better" },
];

function visualMetricOptionsForBenchmark(benchmark) {
  if (benchmark === "do_you_see_me") {
    return visualMetricOptions.filter(({ value }) =>
      value === "perception",
    );
  }
  if (benchmark === "minds_eye") {
    return visualMetricOptions.filter(({ value }) =>
      value === "imagery",
    );
  }
  return visualMetricOptions;
}

function visualDefaultMetric(benchmark) {
  if (benchmark === "do_you_see_me") return "perception";
  if (benchmark === "minds_eye") return "imagery";
  return "vci";
}

const visualBenchmarkOptions = [
  { value: "all", label: "All visual benchmarks" },
  { value: "do_you_see_me", label: "Do You See Me" },
  { value: "minds_eye", label: "Mind's Eye" },
];

const allSpatialTypeOptions = [
  { value: "all", label: "All spatial" },
  { value: "2D", label: "2D" },
  { value: "3D", label: "3D" },
  { value: "dynamic", label: "Dynamic" },
];

const compareBenchmarkOptions = [
  { value: "all", label: "All benchmarks" },
  { value: "do_you_see_me", label: "Do You See Me" },
  { value: "minds_eye", label: "Mind's Eye" },
  { value: "spatial", label: "Spatial" },
];

const spatialDiagnosticMetricOptions = [
  { value: "cot_delta", label: "CoT delta" },
  { value: "shortcut", label: "No image shortcut score, lower is better" },
  { value: "hallucination", label: "No image plus resistance" },
];

function spatialMetricOptionsForScope(scopeActive, hasDiagnostics) {
  return [
    { value: "macro", label: "Overall macro average" },
    {
      value: "accuracy",
      label: scopeActive ? "Selected scope average" : "Main micro accuracy",
    },
    ...(hasDiagnostics ? spatialDiagnosticMetricOptions : []),
  ];
}

const compareMetricConfig = {
  vci: { label: "VPCI", chartLabel: "VPCI" },
  perception: { label: "Perception", chartLabel: "Perception" },
  imagery: { label: "Cognition", chartLabel: "Cognition" },
  spatial: { label: "Spatial", chartLabel: "Spatial" },
  cot_delta: { label: "CoT delta", chartLabel: "CoT delta" },
  shortcut: { label: "Shortcut ↓", chartLabel: "Shortcut" },
  hallucination: { label: "NI++ resist. ↑", chartLabel: "NI++ resistance" },
};

const compareTableMetricsByScope = {
  all: ["vci", "perception", "imagery", "spatial", "cot_delta", "shortcut", "hallucination"],
  do_you_see_me: ["perception"],
  minds_eye: ["imagery"],
  spatial: ["spatial", "cot_delta", "shortcut", "hallucination"],
};

const compareChartMetricsByScope = {
  all: ["vci", "perception", "imagery", "spatial", "hallucination"],
  do_you_see_me: ["perception"],
  minds_eye: ["imagery"],
  spatial: ["spatial", "cot_delta", "shortcut", "hallucination"],
};

// A metric maps to an existing dedicated column when possible, so ranking by it
// simply sorts that column instead of adding a duplicate "rank metric" column.
// Capability ranking has no dedicated base column, so it falls back to a
// temporary rank metric column while selected.
function visualRankSortKey(metric) {
  return (
    {
      vci: "vci",
      perception: "perception",
      imagery: "imagery",
      gap: "gap",
      spread: "spread",
    }[metric] || "rankMetric"
  );
}

function spatialRankSortKey(metric) {
  return (
    {
      accuracy: "scope",
      macro: "macro",
      cot_delta: "cot_delta",
      shortcut: "shortcut",
      hallucination: "hallucination",
    }[metric] || "rankMetric"
  );
}

const spatialDatasetTypes = {
  BLINK: "2D",
  "CV-Bench (2D)": "2D",
  MMVP: "2D",
  RealWorldQA: "2D",
  SpatialBench: "2D",
  VSR: "2D",
  "V*Bench": "2D",
  "3DSRBench": "3D",
  "CV-Bench (3D)": "3D",
  MindCube: "3D",
  "MMSI-Bench": "3D",
  OmniSpatial: "3D",
  "SAT (Real)": "dynamic",
};

function RankBadge({ rank }) {
  return (
    <span className={cn("inline-grid size-8 place-items-center border border-border bg-surface-subtle font-display text-sm font-bold tabular-nums", rank === 1 && "border-amber-400 bg-amber-400/15 text-amber-500", rank === 2 && "border-zinc-400 bg-zinc-400/15", rank === 3 && "border-orange-600 bg-orange-600/15 text-orange-600")}>
      {rank}
    </span>
  );
}

function DiagnosticsChip({ diagnostics }) {
  if (!diagnostics) return <span className={ui.badge}>Standard only</span>;
  const count = diagnostics.conditions_present?.length || 1;
  return <span className={cn(ui.badge, "border-transparent bg-spatial-soft text-spatial")}>{count}/4 conditions</span>;
}

function mean(rows, resolveValue) {
  const values = rows
    .map(resolveValue)
    .filter((value) => value != null && Number.isFinite(value));
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

function cognitionAccuracy(row) {
  return row?.cognition_accuracy ?? row?.imagery_accuracy ?? null;
}

function cognitionGroups(row) {
  return row?.cognition_groups || row?.imagery_groups || {};
}

function hasPerception(row) {
  return Boolean(
    row?.has_perception ||
      row?.perception_accuracy != null ||
      Object.keys(row?.perception_groups || {}).length,
  );
}

function hasCognition(row) {
  return Boolean(
    row?.has_cognition ||
      row?.has_imagery ||
      cognitionAccuracy(row) != null ||
      Object.keys(cognitionGroups(row)).length,
  );
}

function modelOrg(meta) {
  return meta?.organization || meta?.org || "N/A";
}

function parameterValueMissing(value) {
  const normalized = String(value ?? "").trim();
  return (
    !normalized ||
    normalized === "-" ||
    /^(n\/?a|none|unknown|not (provided|available|specified))(\b|$)/i.test(
      normalized,
    )
  );
}

function modelParams(meta) {
  const directValue = meta?.parameter_count ?? meta?.params;
  if (!parameterValueMissing(directValue)) return String(directValue).trim();

  const billions = meta?.params_b;
  if (parameterValueMissing(billions)) return "-";
  const normalizedBillions = String(billions).trim();
  return /b$/i.test(normalizedBillions)
    ? normalizedBillions
    : `${normalizedBillions}B`;
}

function modelBase(meta) {
  return meta?.base_model || meta?.family || "N/A";
}

function modelCot(meta) {
  const value = meta?.cot_used;
  if (value == null || value === "") return "N/A";
  if (value === true) return "Yes";
  if (value === false) return "No";
  return prettyLabel(value);
}

const COMPACT_TABLE_META_KEYS = ["org"];

function compactTableMetaColumns() {
  return new Set(COMPACT_TABLE_META_KEYS);
}

function modelSearchText(row) {
  return [row.model_name, modelOrg(row.model_meta)]
    .join(" ")
    .toLowerCase();
}

function visualCapabilities(rows) {
  const capabilities = [];
  const seen = new Set();
  rows.forEach((row) => {
    Object.keys(row.perception_groups || {}).forEach((key) => {
      const id = `perception:${key}`;
      if (!seen.has(id)) {
        seen.add(id);
        capabilities.push({
          id,
          key,
          layer: "perception",
          label: `P · ${prettyLabel(key)}`,
          order: perceptionCapabilityOrder.indexOf(key),
        });
      }
    });
    Object.keys(cognitionGroups(row)).forEach((key) => {
      const id = `imagery:${key}`;
      if (!seen.has(id)) {
        seen.add(id);
        capabilities.push({
          id,
          key,
          layer: "imagery",
          label: `${artShortLabel[mindsEyeArtByCapability[key]] || "C"} · ${prettyLabel(key)}`,
          order: cognitionCapabilityOrder.indexOf(key),
        });
      }
    });
  });
  return capabilities.sort((left, right) => {
    if (left.layer !== right.layer) return left.layer === "perception" ? -1 : 1;
    const leftOrder = left.order < 0 ? Number.MAX_SAFE_INTEGER : left.order;
    const rightOrder = right.order < 0 ? Number.MAX_SAFE_INTEGER : right.order;
    return leftOrder - rightOrder || left.label.localeCompare(right.label);
  });
}

function getVisualCapability(row, capabilityId) {
  if (!capabilityId || capabilityId === "all") return null;
  const [layer, key] = capabilityId.split(":");
  const groups =
    layer === "perception" ? row.perception_groups : cognitionGroups(row);
  return groups?.[key]?.accuracy ?? null;
}

function visualGap(row) {
  const cognition = cognitionAccuracy(row);
  if (row.perception_accuracy == null || cognition == null)
    return null;
  return row.perception_accuracy - cognition;
}

function combinedTaskSpread(row) {
  const perception = row?.perception_task_spread;
  const cognition = row?.cognition_task_spread;
  if (Number.isFinite(perception) && Number.isFinite(cognition)) {
    return (perception + cognition) / 2;
  }
  return Number.isFinite(row?.task_spread) ? row.task_spread : null;
}

function visualMetricValue(row, metric, capabilityId) {
  if (metric === "perception") return row.perception_accuracy;
  if (metric === "imagery") return cognitionAccuracy(row);
  if (metric === "gap") {
    const gap = visualGap(row);
    return gap == null ? null : Math.abs(gap);
  }
  if (metric === "spread") return combinedTaskSpread(row);
  if (metric === "capability") return getVisualCapability(row, capabilityId);
  return row.vci;
}

function visualMetricDisplay(row, metric, capabilityId) {
  if (metric === "gap") {
    const gap = visualGap(row);
    return gap == null
      ? "N/A"
      : `${gap >= 0 ? "+" : ""}${(gap * 100).toFixed(1)} pts`;
  }
  if (metric === "spread") {
    const spread = combinedTaskSpread(row);
    return spread == null ? "-" : `${(spread * 100).toFixed(1)} pts`;
  }
  const value = visualMetricValue(row, metric, capabilityId);
  return metric === "vci" ? fmtVci(value) : fmtPct(value);
}

function visualMeanDisplay(value, metric) {
  if (value == null) return "N/A";
  if (metric === "vci") return fmtVci(value);
  if (metric === "gap") return `${(value * 100).toFixed(1)} pts`;
  if (metric === "spread") return `${(value * 100).toFixed(1)} pts`;
  return fmtPct(value);
}

function spatialDatasets(rows) {
  const names = new Set();
  rows.forEach((row) =>
    Object.keys(row.groups || {}).forEach((key) => names.add(key)),
  );
  return Array.from(names).sort();
}

function spatialDatasetValue(row, dataset) {
  if (!dataset || dataset === "all") return null;
  return row.groups?.[dataset]?.accuracy ?? null;
}

function getSpatialScopedDatasets(datasets, filters) {
  if (filters.dataset !== "all")
    return datasets.includes(filters.dataset) ? [filters.dataset] : [];
  if (filters.datasetType === "all") return datasets;
  return datasets.filter(
    (dataset) => spatialDatasetTypes[dataset] === filters.datasetType,
  );
}

function spatialScopeAccuracy(row, scopedDatasets) {
  const values = scopedDatasets
    .map((dataset) => row.groups?.[dataset]?.accuracy)
    .filter((value) => value != null && Number.isFinite(value));
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

function spatialMetricValue(
  row,
  metric,
  _dataset,
  scopedDatasets = [],
  scopeActive = false,
) {
  if (metric === "macro") return row.macro_accuracy;
  if (metric === "cot_delta") return row.diagnostics?.cot_delta;
  if (metric === "shortcut") return row.diagnostics?.shortcut_score;
  if (metric === "hallucination")
    return row.diagnostics?.hallucination_resistance;
  if (scopeActive) return spatialScopeAccuracy(row, scopedDatasets);
  return row.accuracy;
}

function spatialMetricDisplay(
  row,
  metric,
  dataset,
  scopedDatasets = [],
  scopeActive = false,
) {
  const value = spatialMetricValue(
    row,
    metric,
    dataset,
    scopedDatasets,
    scopeActive,
  );
  if (metric === "cot_delta")
    return value == null ? "N/A" : `${fmtDelta(value)} pts`;
  return fmtPct(value);
}

function spatialMeanDisplay(value, metric) {
  if (value == null) return "N/A";
  if (metric === "cot_delta") return `${fmtDelta(value)} pts`;
  return fmtPct(value);
}

function metricSortDirection(metric) {
  return metric === "gap" || metric === "spread" || metric === "shortcut"
    ? "asc"
    : "desc";
}

function compareMetricValues(leftValue, rightValue, direction) {
  const leftMissing = leftValue == null || !Number.isFinite(leftValue);
  const rightMissing = rightValue == null || !Number.isFinite(rightValue);
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  return direction === "asc" ? leftValue - rightValue : rightValue - leftValue;
}

function compareSortValues(leftValue, rightValue, direction) {
  const leftMissing =
    leftValue == null ||
    leftValue === "" ||
    (typeof leftValue === "number" && !Number.isFinite(leftValue));
  const rightMissing =
    rightValue == null ||
    rightValue === "" ||
    (typeof rightValue === "number" && !Number.isFinite(rightValue));
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  if (typeof leftValue === "string" || typeof rightValue === "string") {
    const result = String(leftValue).localeCompare(
      String(rightValue),
      undefined,
      { numeric: true, sensitivity: "base" },
    );
    return direction === "asc" ? result : -result;
  }
  return direction === "asc" ? leftValue - rightValue : rightValue - leftValue;
}

function nextSortState(current, key, defaultDirection = "desc") {
  if (current.key === key)
    return { key, direction: current.direction === "asc" ? "desc" : "asc" };
  return { key, direction: defaultDirection };
}

const COLUMN_DESCRIPTIONS = {
  model: "Name of the evaluated vision-language model.",
  org: "Organization or team that produced the model.",
  params: "Reported model parameter count. A dash means the submitter did not provide it.",
  rankMetric: "Value for the currently selected ranking metric or capability.",
  vci: "Visual Perception and Cognition Index (VPCI), the configured weighted mean of the paper aligned Do You See Me and Mind's Eye macro averages, reported from 0 to 100.",
  perception:
    "Do You See Me dimension balanced macro average: task accuracies are averaged within 2D and 3D, then the two dimension averages receive equal weight.",
  perception_2d: "Unweighted mean of the seven Do You See Me 2D task accuracies.",
  perception_3d: "Unweighted mean of the five Do You See Me 3D task accuracies.",
  imagery: "Mind's Eye macro average: the eight task accuracies receive equal weight.",
  art_abstraction: "Unweighted mean of the Mind's Eye Abstraction task accuracies.",
  art_relation: "Unweighted mean of the Mind's Eye Relation task accuracies.",
  art_transformation: "Unweighted mean of the Mind's Eye Transformation task accuracies.",
  spatial: "Unweighted mean of main non-CoT accuracy across the included spatial-reasoning datasets.",
  gap: "Absolute percentage-point difference between overall perception and cognition accuracy. Lower is more balanced.",
  spread: "Equal-weight mean of the Do You See Me and Mind's Eye task-score standard deviations, reported in percentage points. Lower values indicate more consistent performance across tasks; this is not repeated-run uncertainty.",
  scope: "Overall main non-CoT sample accuracy when no scope is selected; otherwise, the unweighted mean of the selected dataset accuracies.",
  accuracy: "Overall sample-level accuracy for the main non-CoT spatial condition.",
  macro: "Unweighted mean of main non-CoT accuracy across datasets.",
  cot_delta: "Main CoT macro accuracy minus main non-CoT macro accuracy, in percentage points. A negative value means CoT reduced accuracy.",
  shortcut:
    "Macro accuracy in the no-image, non-CoT condition. Lower is better because success can indicate language or dataset shortcuts.",
  hallucination:
    "Macro accuracy in the no-image-plus, non-CoT condition, where recognizing that the answer cannot be determined is rewarded. Higher is better.",
  coverage: "Whether the submission includes the required spatial diagnostic conditions.",
};

const RANK_DESCRIPTION = "Position after applying the current filters and sort order.";

function SortHeader({
  label,
  sortKey,
  sort,
  onSort,
  className,
  defaultDirection = "desc",
}) {
  const active = sort.key === sortKey;
  const description = COLUMN_DESCRIPTIONS[sortKey];
  return (
    <th
      className={cn("relative !p-0 whitespace-normal", active && "text-foreground", className)}
      data-tip={description || undefined}
      aria-sort={
        active
          ? sort.direction === "asc"
            ? "ascending"
            : "descending"
          : "none"
      }
    >
      <button className="flex min-h-14 w-full items-center justify-between gap-3 bg-transparent px-4 py-4 text-left text-xs font-semibold uppercase text-inherit hover:bg-brand-soft hover:text-foreground" type="button" onClick={() => onSort(sortKey, defaultDirection)} title={description || undefined}>
        <span className="min-w-0 flex-1 whitespace-normal text-left leading-tight sm:whitespace-nowrap">{label}</span>
        <span aria-hidden="true" className={cn("shrink-0 text-[0.65rem]", active ? "text-brand-strong" : "text-faint")}>
          {active ? (sort.direction === "asc" ? "↑" : "↓") : "↕"}
        </span>
      </button>
    </th>
  );
}

const SORT_LABELS = {
  model: "Model",
  org: "Organization",
  params: "Parameters",
  vci: "VPCI",
  perception: "Perception",
  perception_2d: "2D average",
  perception_3d: "3D average",
  imagery: "Cognition",
  art_abstraction: "Abstraction",
  art_relation: "Relation",
  art_transformation: "Transformation",
  spatial: "Spatial accuracy",
  gap: "Perception cognition gap",
  spread: "Task spread",
  coverage: "Coverage",
  scope: "Scope average",
  accuracy: "Accuracy",
  macro: "Macro average",
  cot_delta: "CoT delta",
  shortcut: "No image shortcut",
  hallucination: "No image plus resistance",
};

function sortSummary(sort, selectedMetric) {
  const label = sort.key === "rankMetric"
    ? selectedMetric
    : SORT_LABELS[sort.key] || prettyLabel(sort.key);
  return `${label}, ${sort.direction === "asc" ? "low to high" : "high to low"}`;
}

function LeaderboardTableHeading({ count, id, metric, scope, title = "Model rankings" }) {
  return (
    <div className="flex items-end justify-between gap-6 border-x border-b border-border-strong px-5 py-5 max-md:flex-col max-md:items-start lg:px-6">
      <div>
        <span className="mb-1.5 block text-xs font-semibold uppercase text-faint">Rankings</span>
        <h3 className="font-display text-2xl font-bold leading-tight" id={id}>{title}</h3>
      </div>
      <dl className="grid min-w-0 grid-cols-1 gap-x-7 gap-y-3 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-xs font-semibold uppercase text-faint">Models</dt>
          <dd className="mt-1 font-medium text-foreground">{count}</dd>
        </div>
        <div>
          <dt className="text-xs font-semibold uppercase text-faint">Scope</dt>
          <dd className="mt-1 max-w-48 font-medium text-foreground">{scope}</dd>
        </div>
        <div>
          <dt className="text-xs font-semibold uppercase text-faint">Sorted by</dt>
          <dd className="mt-1 max-w-52 font-medium text-foreground">{metric}</dd>
        </div>
      </dl>
    </div>
  );
}

function visualColumnValue(row, sortKey, filters) {
  if (sortKey === "model") return row.model_name;
  if (sortKey === "org") return modelOrg(row.model_meta);
  if (sortKey === "params") return modelParams(row.model_meta);
  if (sortKey === "rankMetric")
    return visualMetricValue(row, filters.metric, filters.capability);
  if (sortKey === "vci") return row.vci;
  if (sortKey === "perception") return row.perception_accuracy;
  if (sortKey === "perception_2d")
    return analysisAccuracy(row.perception_dimensions, "2D");
  if (sortKey === "perception_3d")
    return analysisAccuracy(row.perception_dimensions, "3D");
  if (sortKey === "imagery") return cognitionAccuracy(row);
  if (sortKey.startsWith("art_"))
    return analysisAccuracy(cognitionArtGroups(row), sortKey.replace("art_", ""));
  if (sortKey === "gap")
    return visualGap(row) == null ? null : Math.abs(visualGap(row));
  if (sortKey === "spread") return combinedTaskSpread(row);
  if (sortKey.startsWith("capability:"))
    return getVisualCapability(row, sortKey.replace("capability:", ""));
  return visualMetricValue(row, filters.metric, filters.capability);
}

function spatialColumnValue(
  row,
  sortKey,
  filters,
  scopedDatasets,
  scopeActive,
) {
  if (sortKey === "model") return row.model_name;
  if (sortKey === "org") return modelOrg(row.model_meta);
  if (sortKey === "params") return modelParams(row.model_meta);
  if (sortKey === "rankMetric")
    return spatialMetricValue(
      row,
      filters.metric,
      filters.dataset,
      scopedDatasets,
      scopeActive,
    );
  if (sortKey === "scope")
    return scopeActive
      ? spatialScopeAccuracy(row, scopedDatasets)
      : row.accuracy;
  if (sortKey === "accuracy") return row.accuracy;
  if (sortKey === "macro") return row.macro_accuracy;
  if (sortKey === "cot_delta") return row.diagnostics?.cot_delta;
  if (sortKey === "shortcut") return row.diagnostics?.shortcut_score;
  if (sortKey === "hallucination")
    return row.diagnostics?.hallucination_resistance;
  if (sortKey === "coverage")
    return (
      row.diagnostics?.conditions_present?.length || (row.diagnostics ? 1 : 0)
    );
  return row.accuracy;
}

function compareColumnValue(row, sortKey) {
  if (sortKey === "model") return row.model_name;
  if (sortKey === "org") return modelOrg(row.model_meta);
  if (sortKey === "params") return modelParams(row.model_meta);
  if (sortKey === "vci") return row.visual?.vci;
  if (sortKey === "perception") return row.visual?.perception_accuracy;
  if (sortKey === "imagery") return cognitionAccuracy(row.visual);
  if (sortKey === "spatial") return row.spatial?.macro_accuracy ?? row.spatial?.accuracy;
  if (sortKey === "cot_delta") return row.spatial?.diagnostics?.cot_delta;
  if (sortKey === "shortcut") return row.spatial?.diagnostics?.shortcut_score;
  if (sortKey === "hallucination")
    return row.spatial?.diagnostics?.hallucination_resistance;
  return row.visual?.vci ?? row.spatial?.macro_accuracy ?? row.spatial?.accuracy;
}

function compareMetricKeysForScope(scope, rows, chart = false) {
  const keys = (chart ? compareChartMetricsByScope : compareTableMetricsByScope)[scope] || [];
  return keys.filter((key) =>
    rows.some((row) => {
      const value = compareColumnValue(row, key);
      return value != null && Number.isFinite(value);
    }),
  );
}

function compareMetricDisplay(row, key) {
  const value = compareColumnValue(row, key);
  if (key === "vci") return fmtVci(value);
  if (key === "cot_delta") return value == null ? "N/A" : fmtDelta(value);
  return fmtPct(value);
}

function compareChartValue(row, key) {
  const value = compareColumnValue(row, key);
  if (value == null || !Number.isFinite(value)) return null;
  return key === "vci" && value > 1 ? value / 100 : value;
}

function heatPct(value) {
  if (value == null || !Number.isFinite(value)) return "N/A";
  const ratio = value > 1 ? value / 100 : value;
  return `${Math.round(ratio * 100)}%`;
}

function scoreColor(value, invert = false, opacity = 1) {
  if (value == null || !Number.isFinite(value)) return "transparent";
  const normalizedValue = value > 1 ? value / 100 : value;
  const score = Math.max(
    0,
    Math.min(1, invert ? 1 - normalizedValue : normalizedValue),
  );
  const hue = 358 + score * 170;
  const lightness = 20 + score * 34;
  return `hsl(${hue} 72% ${lightness}% / ${opacity})`;
}

function FilterField({ label, children }) {
  return (
    <label className="flex min-w-0 flex-col gap-1.5 text-sm font-semibold text-foreground [&_input]:min-h-10 [&_input]:w-full [&_input]:border [&_input]:border-border-strong [&_input]:bg-surface [&_input]:px-3 [&_input]:py-2 [&_input]:text-sm [&_input]:font-normal [&_input]:outline-none [&_input]:focus:border-brand [&_input]:focus:ring-2 [&_input]:focus:ring-brand-soft [&_select]:min-h-10 [&_select]:w-full [&_select]:border [&_select]:border-border-strong [&_select]:bg-surface [&_select]:px-3 [&_select]:py-2 [&_select]:text-sm [&_select]:font-normal [&_select]:outline-none [&_select]:focus:border-brand [&_select]:focus:ring-2 [&_select]:focus:ring-brand-soft">
      <span className="text-xs uppercase text-faint">{label}</span>
      {children}
    </label>
  );
}

function SegmentControl({ label, options, value, onChange }) {
  return (
    <div className="inline-flex flex-wrap border border-border bg-surface" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={cn("min-h-10 border-0 border-r border-border bg-transparent px-3 py-2 text-sm font-semibold text-muted last:border-r-0 hover:bg-surface-subtle hover:text-foreground", value === option.value && "bg-invert-bg text-invert-text hover:bg-invert-bg-hover hover:text-invert-text")}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ControlDeck({ eyebrow, title, chips, children }) {
  return (
    <div className="mb-6 border border-border p-5">
      <div className="mb-4 flex items-start justify-between gap-5 border-b border-border pb-4">
        <div>
          <div className={ui.sectionTag}>{eyebrow}</div>
          <h3 className={ui.heading3}>{title}</h3>
        </div>
      </div>
      {children}
      <ActiveChips chips={chips} />
    </div>
  );
}

function ActiveChips({ chips }) {
  const visibleChips = chips.filter(Boolean);
  if (!visibleChips.length) return null;
  return (
    <div className="mt-4 flex flex-wrap gap-2 border-t border-border pt-4" aria-label="Active filters">
      {visibleChips.map((chip) => (
        <button
          key={chip.label}
          type="button"
          className="inline-flex min-h-8 items-center gap-2 border border-border bg-surface-subtle px-2.5 py-1 text-xs text-muted hover:border-brand-strong hover:text-foreground"
          onClick={chip.onClear}
        >
          {chip.label}
          <span aria-hidden="true">×</span>
        </button>
      ))}
    </div>
  );
}

function compareCoverageText(row) {
  return (
    [
      hasPerception(row.visual) && "P",
      hasCognition(row.visual) && "C",
      row.spatial && "S",
    ]
      .filter(Boolean)
      .join(" + ") || "N/A"
  );
}

function CompareModelPicker({
  candidateRows,
  onAdd,
  onRemove,
  onReset,
  selectedNames,
  selectedRows,
}) {
  const selectedSet = new Set(selectedNames);
  return (
    <div className="mt-4 grid gap-3 border-t border-border pt-4">
      <div className="flex items-center justify-between gap-3 max-sm:flex-col max-sm:items-start">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-faint">
          <span>Selected models</span>
          <strong>{selectedRows.length} selected</strong>
        </div>
        <button className={ui.linkButton} type="button" onClick={onReset}>
          Reset to top models
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {selectedRows.length ? (
          selectedRows.map((row) => (
            <button
              type="button"
              className="inline-flex items-center gap-2 border border-border bg-surface-subtle px-3 py-2 text-left text-sm hover:border-brand-strong"
              key={row.model_name}
              onClick={() => onRemove(row.model_name)}
            >
              <strong>{row.model_name}</strong>
              <span className="text-xs text-faint">{compareCoverageText(row)}</span>
              <em className="not-italic text-faint" aria-hidden="true">×</em>
            </button>
          ))
        ) : (
          <p className="text-sm text-muted">
            Select at least one model to populate the comparison table.
          </p>
        )}
      </div>
      <div
        className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-2"
        aria-label="Available models to compare"
      >
        {candidateRows.length ? (
          candidateRows.map((row) => {
            const selected = selectedSet.has(row.model_name);
            return (
              <button
                className="flex min-w-0 flex-col items-start border border-border bg-surface px-3 py-2 text-left text-sm hover:border-brand-strong disabled:cursor-not-allowed disabled:opacity-50"
                type="button"
                key={row.model_name}
                disabled={selected}
                onClick={() => onAdd(row.model_name)}
              >
                <span className="text-xs uppercase text-faint">{selected ? "Selected" : "Add"}</span>
                <strong>{row.model_name}</strong>
                <em className="not-italic text-xs text-faint">{compareCoverageText(row)}</em>
              </button>
            );
          })
        ) : (
          <p className="text-sm text-muted">
            No models match this search and benchmark scope.
          </p>
        )}
      </div>
    </div>
  );
}

function visualBenchmarkLabel(value) {
  return (
    visualBenchmarkOptions.find((option) => option.value === value)?.label ||
    "All visual benchmarks"
  );
}

function compareBenchmarkLabel(value) {
  return (
    compareBenchmarkOptions.find((option) => option.value === value)?.label ||
    "All benchmarks"
  );
}

function DashboardStats({ items }) {
  return (
    <div className="mb-6 grid grid-cols-1 border-l border-t border-border sm:grid-cols-2 lg:grid-cols-4">
      {items.map(([value, label]) => (
        <div className={cn(ui.ruledCell, "!bg-transparent")} key={label}>
          <div className="font-display text-3xl font-bold tabular-nums">{value}</div>
          <div className="mt-2 text-sm text-muted">{label}</div>
        </div>
      ))}
    </div>
  );
}

function Heatmap({ rows, columns, valueFor, displayFor, emptyText, translucent = false }) {
  if (!rows.length || !columns.length)
    return <div className="grid min-h-52 place-items-center border border-dashed border-border p-6 text-center text-sm text-muted">{emptyText}</div>;
  return (
    <div className="max-h-[460px] max-w-full overflow-auto pb-1">
      <div
        className="grid min-w-max gap-px border border-border bg-border"
        style={{
          gridTemplateColumns: `minmax(160px, 1.4fr) repeat(${columns.length}, minmax(74px, 1fr))`,
        }}
      >
        <div className="sticky left-0 top-0 z-[5] flex min-h-10 items-center bg-surface px-2.5 py-2 text-center text-xs font-bold uppercase text-faint">Model</div>
        {columns.map((column) => (
          <div className="sticky top-0 z-[4] flex min-h-10 items-center justify-center bg-surface px-2.5 py-2 text-center text-xs font-bold uppercase text-faint" key={column.id || column}>
            {column.label || prettyLabel(column)}
          </div>
        ))}
        {rows.map((row) => (
          <div className="contents" key={row.model_name}>
            <div className={cn("sticky left-0 z-[3] flex min-h-10 items-center bg-surface px-2.5 py-2 font-medium text-foreground shadow-[1px_0_0_var(--border)]", row.reference && "text-page-accent")}>{row.model_name}</div>
            {columns.map((column) => {
              const value = valueFor(row, column);
              return (
                <div
                  className={cn(
                    "flex min-h-10 items-center justify-center px-2.5 py-2 text-xs font-bold tabular-nums",
                    translucent
                      ? "text-foreground transition-[background-color] hover:brightness-110"
                      : "text-white [text-shadow:0_1px_2px_rgba(0,0,0,.38)]",
                  )}
                  key={`${row.model_name}-${column.id || column}`}
                  style={{ background: scoreColor(value, false, translucent ? 0.42 : 1) }}
                  title={`${row.model_name} · ${column.label || column}: ${displayFor(value)}`}
                >
                  {displayFor(value)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function VisualCapabilityChart({ benchmark, rows, capabilities }) {
  const chartRows = benchmark === "minds_eye"
    ? [...mindsEyePaperReferenceRows, ...rows]
    : rows;
  return (
    <>
      <Heatmap
        rows={chartRows}
        columns={capabilities}
        valueFor={(row, column) => getVisualCapability(row, column.id)}
        displayFor={heatPct}
        emptyText="Capability profiles appear when ranked models are available."
        translucent
      />
      {benchmark === "minds_eye" && rows.length > 0 && (
        <p className="mt-3 text-xs leading-relaxed text-faint">
          Paper human and random choice rows reproduce the reference values in Mind's Eye Table 2. Submitted models are scored on the leaderboard release suite.
        </p>
      )}
    </>
  );
}

function SpatialDatasetChart({ rows, datasets }) {
  return (
    <Heatmap
      rows={rows}
      columns={datasets.map((dataset) => ({ id: dataset, label: dataset }))}
      valueFor={(row, column) => spatialDatasetValue(row, column.id)}
      displayFor={heatPct}
      emptyText="Accuracy for each dataset appears when spatial rankings are available."
    />
  );
}

const NARROW_CHART_MODEL_CAP = 12;

function useFilteredModelSelection(rows, defaultCount = DEFAULT_CHART_MODELS) {
  const [selected, setSelected] = useState([]);

  useEffect(() => {
    setSelected((current) => {
      const availableNames = new Set(rows.map((row) => row.model_name));
      const retained = current.filter((name) => availableNames.has(name));
      const next = retained.length
        ? retained
        : rows.slice(0, defaultCount).map((row) => row.model_name);
      return next.length === current.length && next.every((name, index) => name === current[index])
        ? current
        : next;
    });
  }, [defaultCount, rows]);

  const toggle = (name) => {
    setSelected((current) => {
      if (current.includes(name)) {
        return current.length > 1
          ? current.filter((item) => item !== name)
          : current;
      }
      return [...current, name];
    });
  };

  const reset = () => {
    setSelected(rows.slice(0, defaultCount).map((row) => row.model_name));
  };

  return { reset, selected, toggle };
}

function ChartModelSelector({ defaultCount = DEFAULT_CHART_MODELS, label, onReset, onToggle, rows, selected }) {
  const selectedSet = new Set(selected);
  const resetCount = Math.min(defaultCount, rows.length);
  const resetLabel = resetCount === 1
    ? "Top model"
    : resetCount > 1
      ? `Top ${resetCount} models`
      : "Reset";
  const summary = selected.length === 1
    ? selected[0]
    : `${selected.length} models`;
  return (
    <details className="relative shrink-0">
      <summary className="flex min-h-9 cursor-pointer list-none items-center justify-between gap-2 border border-border-strong px-3 py-1.5 text-xs font-medium text-foreground [&::-webkit-details-marker]:hidden" aria-label={`Select models for ${label}`}>
        <span className="max-w-40 truncate">{summary || "Select model"}</span>
        <ChevronDown className="size-3.5 shrink-0 text-faint" aria-hidden="true" />
      </summary>
      <div className="absolute right-0 z-40 mt-2 w-[min(19rem,80vw)] border border-border-strong bg-background shadow-lg">
        <div className="flex items-center justify-between gap-3 border-b border-border px-3 py-2.5">
          <span className="text-xs font-semibold uppercase text-faint">Models {selected.length} selected</span>
          <button className={ui.linkButton} type="button" onClick={onReset}>{resetLabel}</button>
        </div>
        <div className="max-h-64 overflow-auto p-1.5">
          {rows.length ? rows.map((row) => {
            const checked = selectedSet.has(row.model_name);
            const disabled = checked && selected.length === 1;
            return (
              <label className={cn("flex cursor-pointer items-center gap-3 px-2.5 py-2 text-sm text-foreground hover:bg-surface-subtle", disabled && "cursor-not-allowed opacity-50")} key={row.model_name}>
                <input
                  checked={checked}
                  className="size-4 shrink-0 accent-current"
                  disabled={disabled}
                  type="checkbox"
                  onChange={() => onToggle(row.model_name)}
                />
                <span className="min-w-0 truncate">{row.model_name}</span>
              </label>
            );
          }) : <p className="px-2.5 py-3 text-sm text-muted">No models are available under the current filters.</p>}
        </div>
      </div>
    </details>
  );
}

function ChartModelNote({ shown, total }) {
  if (total <= shown) return null;
  return (
    <p className="mt-3 text-xs text-faint">
      Top {shown} of {total} models. The full ranking appears in the table above.
    </p>
  );
}

function PerceptionCognitionChart({ benchmark, rows, selected }) {
  const eligibleRows = rows.filter((row) => {
    if (benchmark === "do_you_see_me") return row.perception_accuracy != null;
    if (benchmark === "minds_eye") return cognitionAccuracy(row) != null;
    return row.perception_accuracy != null && cognitionAccuracy(row) != null;
  });
  const selectedSet = new Set(selected);
  const shownRows = eligibleRows.filter((row) => selectedSet.has(row.model_name));
  const series = benchmark === "do_you_see_me"
    ? [
        { key: "perception", label: "Perception", color: chartPalette[1], valueFor: (category) => category.row.perception_accuracy },
      ]
    : benchmark === "minds_eye"
      ? [
          { key: "cognition", label: "Cognition", color: chartPalette[4], valueFor: (category) => cognitionAccuracy(category.row) },
        ]
      : [
          { key: "perception", label: "Perception", color: chartPalette[1], valueFor: (category) => category.row.perception_accuracy },
          { key: "cognition", label: "Cognition", color: chartPalette[4], valueFor: (category) => cognitionAccuracy(category.row) },
        ];
  return (
    <BarChart
      aspectRatio="16 / 8"
      categories={shownRows.map((row) => ({ label: row.model_name, row }))}
      emptyMessage={benchmark === "all"
        ? "Select a model with both perception and cognition scores."
        : "Select a model with a score for this benchmark."}
      series={series}
    />
  );
}

function analysisAccuracy(groups, key) {
  const value = groups?.[key]?.accuracy;
  return Number.isFinite(value) ? value : null;
}

function cognitionArtGroups(row) {
  if (Object.keys(row?.cognition_art || {}).length) return row.cognition_art;

  const totals = {};
  Object.entries(cognitionGroups(row)).forEach(([capability, group]) => {
    const art = mindsEyeArtByCapability[capability];
    if (!art) return;
    const total = Number(group.total_samples) || 0;
    const correct = Number(group.correct_samples) || 0;
    const accuracy = Number(group.accuracy);
    if (!Number.isFinite(accuracy)) return;
    const bucket = totals[art] || { accuracies: [], correct: 0, total: 0 };
    bucket.accuracies.push(accuracy);
    bucket.correct += correct;
    bucket.total += total;
    totals[art] = bucket;
  });
  return Object.fromEntries(
    Object.entries(totals).map(([art, values]) => [
      art,
      {
        name: art,
        total_samples: values.total,
        correct_samples: values.correct,
        accuracy: values.accuracies.reduce((sum, value) => sum + value, 0) / values.accuracies.length,
        meta: { aggregation: "unweighted_task_mean", task_count: values.accuracies.length },
      },
    ]),
  );
}

function hasPerceptionAnalysis(row) {
  const dimensions = row?.perception_dimensions || {};
  const difficulty = row?.perception_difficulty || {};
  return (
    ["2D", "3D"].some((key) => analysisAccuracy(dimensions, key) != null) ||
    difficultyLevels.some((key) => analysisAccuracy(difficulty, key) != null)
  );
}

function hasArtAnalysis(row) {
  const groups = cognitionArtGroups(row);
  return artDimensions.some((key) => analysisAccuracy(groups, key) != null);
}

function DimensionTransferChart({ rows, selected }) {
  const selectedSet = new Set(selected);
  const points = rows
    .filter((row) => selectedSet.has(row.model_name))
    .map((row, index) => ({
      key: row.model_name,
      label: row.model_name,
      color: chartPalette[index % chartPalette.length],
      x: analysisAccuracy(row.perception_dimensions, "2D"),
      y: analysisAccuracy(row.perception_dimensions, "3D"),
    }))
    .filter((point) => point.x != null && point.y != null);

  return (
    <ScatterChart
      aspectRatio="4 / 3"
      emptyMessage="Dimension breakdowns are unavailable for the selected models."
      points={points}
      xLabel="2D accuracy"
      yLabel="3D accuracy"
    />
  );
}

function DifficultyResponseChart({ rows, selected }) {
  const selectedSet = new Set(selected);
  const visibleRows = rows.filter(
    (row) => selectedSet.has(row.model_name) && hasPerceptionAnalysis(row),
  );
  return (
    <BarChart
      aspectRatio="4 / 3"
      categories={difficultyLevels.map((level) => ({ label: prettyLabel(level), level }))}
      emptyMessage="Difficulty breakdowns are unavailable for the selected models."
      series={visibleRows.map((row, index) => ({
        key: row.model_name,
        label: row.model_name,
        color: chartPalette[index % chartPalette.length],
        valueFor: (category) => analysisAccuracy(row.perception_difficulty, category.level),
      }))}
    />
  );
}

function ArtSummaryChart({ rows, selected }) {
  const selectedSet = new Set(selected);
  const visibleRows = rows.filter(
    (row) => selectedSet.has(row.model_name) && hasArtAnalysis(row),
  );
  return (
    <BarChart
      aspectRatio="4 / 3"
      categories={artDimensions.map((dimension) => ({ label: prettyLabel(dimension), dimension }))}
      compactXLabels
      emptyMessage="Cognitive dimension breakdowns are unavailable for the selected models."
      forceHorizontalLabels
      series={visibleRows.map((row, index) => {
        const groups = cognitionArtGroups(row);
        return {
          key: row.model_name,
          label: row.model_name,
          color: chartPalette[index % chartPalette.length],
          valueFor: (category) => analysisAccuracy(groups, category.dimension),
        };
      })}
    />
  );
}

function capabilitySummaryForScope(row, scope) {
  const groups = [
    ...(scope !== "cognition" ? Object.values(row?.perception_groups || {}) : []),
    ...(scope !== "perception" ? Object.values(cognitionGroups(row)) : []),
  ];
  const values = groups.map((group) => group?.accuracy).filter(finiteNumber);
  if (!values.length) return null;
  return {
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    weakest: Math.min(...values),
  };
}

function CapabilityFloorChart({ rows, scope, selected }) {
  const selectedSet = new Set(selected);
  const points = rows
    .filter((row) => selectedSet.has(row.model_name))
    .map((row, index) => {
      const summary = capabilitySummaryForScope(row, scope);
      if (!summary) return null;
      return {
        key: row.model_name,
        label: row.model_name,
        color: chartPalette[index % chartPalette.length],
        x: summary.mean,
        y: summary.weakest,
      };
    })
    .filter(Boolean);

  return (
    <ScatterChart
      aspectRatio="4 / 3"
      emptyMessage="Capability breakdowns are unavailable for the selected models."
      points={points}
      xLabel="Average capability accuracy"
      yLabel="Weakest capability accuracy"
    />
  );
}

function CotComparisonChart({ rows }) {
  const data = rows.filter(
    (row) =>
      row.diagnostics?.standard_accuracy != null &&
      row.diagnostics?.cot_accuracy != null,
  );
  const shownRows = data.slice(0, NARROW_CHART_MODEL_CAP);
  return (
    <>
      <BarChart
        aspectRatio="16 / 8"
        categories={shownRows.map((row) => ({ label: row.model_name, row }))}
        emptyMessage="Needs main non-CoT and main CoT results."
        series={[
          { key: "standard", label: "Standard", color: chartPalette[1], valueFor: (category) => category.row.diagnostics.standard_accuracy },
          { key: "cot", label: "Chain of thought", color: chartPalette[2], valueFor: (category) => category.row.diagnostics.cot_accuracy },
        ]}
      />
      <ChartModelNote shown={NARROW_CHART_MODEL_CAP} total={data.length} />
    </>
  );
}

function RobustnessChart({ rows }) {
  const data = rows.filter(
    (row) =>
      row.diagnostics?.hallucination_resistance != null &&
      row.diagnostics?.shortcut_score != null,
  );
  const shownRows = data.slice(0, NARROW_CHART_MODEL_CAP);
  return (
    <>
      <div>
        <h4 className="font-display text-sm font-semibold text-foreground">No Image++ resistance</h4>
        <p className="mt-1 text-xs leading-relaxed text-muted">Correct abstention when visual evidence is removed. Higher is better.</p>
        <BarChart
          aspectRatio="16 / 6"
          categories={shownRows.map((row) => ({ label: row.model_name, row }))}
          emptyMessage="Needs No Image++ diagnostics."
          series={[
            { key: "hallucination", label: "No Image++ resistance", color: chartPalette[0], valueFor: (category) => category.row.diagnostics.hallucination_resistance },
          ]}
        />
      </div>
      <div className="mt-6 border-t border-border pt-6">
        <h4 className="font-display text-sm font-semibold text-foreground">No Image shortcut score</h4>
        <p className="mt-1 text-xs leading-relaxed text-muted">Accuracy without visual evidence. Lower is better.</p>
        <BarChart
          aspectRatio="16 / 6"
          categories={shownRows.map((row) => ({ label: row.model_name, row }))}
          emptyMessage="Needs No Image diagnostics."
          series={[
            { key: "shortcut", label: "Shortcut score", color: chartPalette[2], valueFor: (category) => category.row.diagnostics.shortcut_score },
          ]}
        />
      </div>
      <ChartModelNote shown={NARROW_CHART_MODEL_CAP} total={data.length} />
    </>
  );
}

function ComparisonChart({ rows, scope }) {
  const metricKeys = compareMetricKeysForScope(scope, rows, true);
  const metrics = metricKeys.map((key) => ({
    key,
    label: compareMetricConfig[key].chartLabel,
  }));
  return (
    <BarChart
      aspectRatio="16 / 7"
      categories={metrics}
      emptyMessage="No models match the comparison filters."
      series={rows.map((row, index) => ({
        key: row.model_name,
        label: row.model_name,
        color: chartPalette[index % chartPalette.length],
        valueFor: (category) => compareChartValue(row, category.key),
      }))}
    />
  );
}

function finiteNumber(value) {
  return value != null && Number.isFinite(value);
}

function groupStats(groups) {
  const values = Object.values(groups || {})
    .map((group) => group.accuracy)
    .filter(finiteNumber);
  if (!values.length) return { mean: null, std: null };
  const meanValue = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.length > 1
      ? values.reduce((sum, value) => sum + (value - meanValue) ** 2, 0) /
        (values.length - 1)
      : 0;
  return { mean: meanValue, std: Math.sqrt(variance) };
}

function groupExtremes(groups) {
  const entries = Object.entries(groups || {}).filter(([, group]) =>
    finiteNumber(group.accuracy),
  );
  if (!entries.length) return { strongest: null, weakest: null };
  const sorted = [...entries].sort(
    ([, left], [, right]) => right.accuracy - left.accuracy,
  );
  return {
    strongest: { key: sorted[0][0], ...sorted[0][1] },
    weakest: { key: sorted[sorted.length - 1][0], ...sorted[sorted.length - 1][1] },
  };
}

function AnalysisCards({ items }) {
  const visibleItems = items.filter(Boolean);
  if (!visibleItems.length) return null;
  return (
    <div className="mb-5 grid grid-cols-[repeat(auto-fit,minmax(210px,1fr))] border-l border-t border-border">
      {visibleItems.map((item) => (
        <div className="min-w-0 border-b border-r border-border p-4" key={item.label}>
          <span className="mb-2 block text-xs font-semibold uppercase text-faint">{item.label}</span>
          <strong className="block break-words font-display text-base text-foreground">{item.value}</strong>
          <p className="mt-2 text-sm leading-relaxed text-muted">{item.detail}</p>
        </div>
      ))}
    </div>
  );
}

export function ResearchLeaderboard() {
  const [tab, setTab] = useState("vc");
  const [, setStats] = useState({});
  const [, setTaskInfo] = useState({});
  const [visualRows, setVisualRows] = useState([]);
  const [spatialRows, setSpatialRows] = useState([]);
  const [reportModel, setReportModel] = useState(null);
  const [loadStatus, setLoadStatus] = useState("loading");
  const [loadError, setLoadError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [visualFilters, setVisualFilters] = useState({
    search: "",
    benchmark: "all",
    metric: "vci",
    capability: "all",
  });
  const [spatialFilters, setSpatialFilters] = useState({
    search: "",
    metric: "macro",
    dataset: "all",
    datasetType: "all",
    diagnostics: "all",
  });
  const [compareSearch, setCompareSearch] = useState("");
  const [visualSort, setVisualSort] = useState({
    key: "vci",
    direction: "desc",
  });
  const [spatialSort, setSpatialSort] = useState({
    key: "macro",
    direction: "desc",
  });
  const [compareSort, setCompareSort] = useState({
    key: "vci",
    direction: "desc",
  });
  const [compareBenchmark, setCompareBenchmark] = useState("all");
  const [selectedCompareModels, setSelectedCompareModels] = useState([]);

  useEffect(() => {
    let live = true;
    setLoadStatus("loading");
    setLoadError("");
    Promise.allSettled([
      getJSON("/api/statistics/overview"),
      getJSON("/api/tasks/do_you_see_me/info"),
      getJSON("/api/tasks/minds_eye/info"),
      getJSON("/api/tasks/spatial/info"),
      getJSON("/api/leaderboard/visual-cognition"),
      getJSON("/api/leaderboard/spatial"),
    ]).then((results) => {
      if (!live) return;
      const value = (index, fallback = {}) => results[index].status === "fulfilled" ? results[index].value : fallback;
      setStats(value(0));
      setTaskInfo({
        do_you_see_me: value(1),
        minds_eye: value(2),
        spatial: value(3),
      });
      setVisualRows(value(4, { leaderboard: [] }).leaderboard || []);
      setSpatialRows(value(5, { leaderboard: [] }).leaderboard || []);
      const failed = results
        .map((result, index) => result.status === "rejected" ? { index, error: result.reason } : null)
        .filter(Boolean);
      if (failed.length) {
        const rankingFailure = failed.find(({ index }) => index === 4 || index === 5);
        setLoadStatus(rankingFailure ? "error" : "partial");
        setLoadError(
          rankingFailure
            ? errorMessage(rankingFailure.error, "Leaderboard rankings could not be loaded.")
            : `Rankings loaded, but some summary details are unavailable. ${errorMessage(failed[0].error)}`,
        );
      } else {
        setLoadStatus("ready");
      }
    });
    return () => { live = false; };
  }, [reloadKey]);

  const capabilities = useMemo(
    () => visualCapabilities(visualRows),
    [visualRows],
  );
  const visibleCapabilities = useMemo(() => {
    if (visualFilters.benchmark === "do_you_see_me")
      return capabilities.filter(
        (capability) => capability.layer === "perception",
      );
    if (visualFilters.benchmark === "minds_eye")
      return capabilities.filter(
        (capability) => capability.layer === "imagery",
      );
    return capabilities;
  }, [capabilities, visualFilters.benchmark]);
  const datasets = useMemo(() => spatialDatasets(spatialRows), [spatialRows]);
  const hasSpatialDiagnostics = useMemo(
    () => spatialRows.some((row) => Boolean(row.diagnostics)),
    [spatialRows],
  );
  const spatialTypeOptions = useMemo(() => {
    const availableTypes = new Set(
      datasets.map((dataset) => spatialDatasetTypes[dataset]).filter(Boolean),
    );
    return allSpatialTypeOptions.filter(
      (option) => option.value === "all" || availableTypes.has(option.value),
    );
  }, [datasets]);
  const datasetsForSelectedType = useMemo(
    () => spatialFilters.datasetType === "all"
      ? datasets
      : datasets.filter(
          (dataset) => spatialDatasetTypes[dataset] === spatialFilters.datasetType,
        ),
    [datasets, spatialFilters.datasetType],
  );
  const spatialBenchmarkOptions = useMemo(
    () => [
      {
        value: "all",
        label: spatialFilters.datasetType === "all"
          ? "All spatial benchmarks"
          : `All ${spatialFilters.datasetType} benchmarks`,
      },
      ...datasetsForSelectedType.map((dataset) => ({ value: dataset, label: dataset })),
    ],
    [datasetsForSelectedType, spatialFilters.datasetType],
  );
  const spatialScopedDatasets = useMemo(
    () => getSpatialScopedDatasets(datasets, spatialFilters),
    [datasets, spatialFilters],
  );
  const spatialScopeActive =
    spatialFilters.dataset !== "all" || spatialFilters.datasetType !== "all";
  const scopedSpatialMetricOptions = useMemo(
    () => spatialMetricOptionsForScope(spatialScopeActive, hasSpatialDiagnostics),
    [hasSpatialDiagnostics, spatialScopeActive],
  );

  const filteredVisualRows = useMemo(() => {
    const query = visualFilters.search.trim().toLowerCase();
    return visualRows
      .filter(
        (row) =>
          !query ||
          modelSearchText(row).includes(query),
      )
      .filter(
        (row) =>
          visualFilters.benchmark === "all" ||
          (visualFilters.benchmark === "do_you_see_me"
            ? hasPerception(row)
            : hasCognition(row)),
      )
      .sort((left, right) => {
        const valueCompare = compareSortValues(
          visualColumnValue(left, visualSort.key, visualFilters),
          visualColumnValue(right, visualSort.key, visualFilters),
          visualSort.direction,
        );
        if (valueCompare !== 0) return valueCompare;
        return compareMetricValues(left.vci, right.vci, "desc");
      });
  }, [visualRows, visualFilters, visualSort]);

  const perceptionChartRows = useMemo(
    () => filteredVisualRows.filter((row) => {
      if (visualFilters.benchmark === "do_you_see_me")
        return row.perception_accuracy != null;
      if (visualFilters.benchmark === "minds_eye")
        return cognitionAccuracy(row) != null;
      return row.perception_accuracy != null && cognitionAccuracy(row) != null;
    }),
    [filteredVisualRows, visualFilters.benchmark],
  );
  const capabilityChartRows = useMemo(
    () => filteredVisualRows.filter((row) => {
      const hasPerceptionGroups = Object.keys(row.perception_groups || {}).length > 0;
      const hasCognitionGroups = Object.keys(cognitionGroups(row)).length > 0;
      if (visualFilters.benchmark === "do_you_see_me")
        return hasPerceptionGroups;
      if (visualFilters.benchmark === "minds_eye")
        return hasCognitionGroups;
      return hasPerceptionGroups || hasCognitionGroups;
    }),
    [filteredVisualRows, visualFilters.benchmark],
  );
  const paperAnalysisRows = useMemo(
    () => filteredVisualRows.filter((row) => {
      if (visualFilters.benchmark === "do_you_see_me") return hasPerceptionAnalysis(row);
      if (visualFilters.benchmark === "minds_eye") return hasArtAnalysis(row);
      return hasPerceptionAnalysis(row) || hasArtAnalysis(row);
    }),
    [filteredVisualRows, visualFilters.benchmark],
  );
  const perceptionChartSelection = useFilteredModelSelection(perceptionChartRows);
  const capabilityChartSelection = useFilteredModelSelection(capabilityChartRows);
  const paperAnalysisSelection = useFilteredModelSelection(paperAnalysisRows);

  const filteredSpatialRows = useMemo(() => {
    const query = spatialFilters.search.trim().toLowerCase();
    return spatialRows
      .filter(
        (row) =>
          !query ||
          modelSearchText(row).includes(query),
      )
      .filter(
        (row) =>
          spatialFilters.diagnostics === "all" || Boolean(row.diagnostics),
      )
      .filter(
        (row) =>
          !spatialScopeActive ||
          spatialScopedDatasets.some((dataset) => row.groups?.[dataset]),
      )
      .sort((left, right) => {
        const valueCompare = compareSortValues(
          spatialColumnValue(
            left,
            spatialSort.key,
            spatialFilters,
            spatialScopedDatasets,
            spatialScopeActive,
          ),
          spatialColumnValue(
            right,
            spatialSort.key,
            spatialFilters,
            spatialScopedDatasets,
            spatialScopeActive,
          ),
          spatialSort.direction,
        );
        if (valueCompare !== 0) return valueCompare;
        return compareMetricValues(left.accuracy, right.accuracy, "desc");
      });
  }, [
    spatialRows,
    spatialFilters,
    spatialScopeActive,
    spatialScopedDatasets,
    spatialSort,
  ]);

  const availableCompareBenchmarkOptions = useMemo(
    () => compareBenchmarkOptions.filter((option) => {
      if (option.value === "all") return true;
      if (option.value === "do_you_see_me") return visualRows.some(hasPerception);
      if (option.value === "minds_eye") return visualRows.some(hasCognition);
      return spatialRows.length > 0;
    }),
    [spatialRows, visualRows],
  );

  useEffect(() => {
    if (!availableCompareBenchmarkOptions.some((option) => option.value === compareBenchmark)) {
      setCompareBenchmark("all");
      setCompareSort({ key: "vci", direction: "desc" });
    }
  }, [availableCompareBenchmarkOptions, compareBenchmark]);

  const availableCompareRows = useMemo(() => {
    const byModel = new Map();
    visualRows.forEach((row) =>
      byModel.set(row.model_name, {
        model_name: row.model_name,
        model_meta: row.model_meta,
        visual: row,
      }),
    );
    spatialRows.forEach((row) => {
      const current = byModel.get(row.model_name) || {
        model_name: row.model_name,
        model_meta: row.model_meta,
      };
      byModel.set(row.model_name, {
        ...current,
        spatial: row,
        model_meta: current.model_meta || row.model_meta,
      });
    });
    return Array.from(byModel.values())
      .filter(
        (row) =>
          compareBenchmark === "all" ||
          (compareBenchmark === "do_you_see_me"
            ? hasPerception(row.visual)
            : compareBenchmark === "minds_eye"
              ? hasCognition(row.visual)
              : row.spatial),
      )
      .sort((left, right) =>
        compareMetricValues(
          left.visual?.vci ?? left.spatial?.macro_accuracy ?? left.spatial?.accuracy ?? 0,
          right.visual?.vci ?? right.spatial?.macro_accuracy ?? right.spatial?.accuracy ?? 0,
          "desc",
        ),
      );
  }, [visualRows, spatialRows, compareBenchmark]);

  useEffect(() => {
    setSelectedCompareModels((current) => {
      const availableNames = new Set(
        availableCompareRows.map((row) => row.model_name),
      );
      const retained = current
        .filter((name) => availableNames.has(name));
      if (retained.length) return retained;
      return availableCompareRows
        .slice(0, DEFAULT_COMPARE_MODELS)
        .map((row) => row.model_name);
    });
  }, [availableCompareRows]);

  const compareCandidateRows = useMemo(() => {
    const query = compareSearch.trim().toLowerCase();
    return availableCompareRows.filter(
      (row) =>
        !query ||
        modelSearchText(row).includes(query),
    );
  }, [availableCompareRows, compareSearch]);

  const compareRows = useMemo(() => {
    const byModel = new Map(
      availableCompareRows.map((row) => [row.model_name, row]),
    );
    return selectedCompareModels
      .map((name) => byModel.get(name))
      .filter(Boolean)
      .sort((left, right) => {
        const valueCompare = compareSortValues(
          compareColumnValue(left, compareSort.key),
          compareColumnValue(right, compareSort.key),
          compareSort.direction,
        );
        if (valueCompare !== 0) return valueCompare;
        return compareMetricValues(
          left.visual?.vci ?? left.spatial?.macro_accuracy ?? left.spatial?.accuracy ?? 0,
          right.visual?.vci ?? right.spatial?.macro_accuracy ?? right.spatial?.accuracy ?? 0,
          "desc",
        );
      });
  }, [availableCompareRows, selectedCompareModels, compareSort]);

  const compareMetricKeys = useMemo(
    () => compareMetricKeysForScope(compareBenchmark, compareRows),
    [compareBenchmark, compareRows],
  );

  useEffect(() => {
    if (!compareRows.length) return;
    const metadataSortKeys = new Set(["model", "org", "params"]);
    if (
      !metadataSortKeys.has(compareSort.key)
      && !compareMetricKeys.includes(compareSort.key)
    ) {
      const key = compareMetricKeys[0] || "model";
      setCompareSort({
        key,
        direction: key === "shortcut" ? "asc" : key === "model" ? "asc" : "desc",
      });
    }
  }, [compareMetricKeys, compareRows.length, compareSort.key]);

  const selectedCapability = capabilities.find(
    (capability) => capability.id === visualFilters.capability,
  );
  const scopedVisualMetricOptions = [
    ...visualMetricOptionsForBenchmark(visualFilters.benchmark),
    ...(selectedCapability
      ? [{ value: "capability", label: selectedCapability.label }]
      : []),
  ];
  const visualMetricLabel =
    visualFilters.metric === "capability" && selectedCapability
      ? selectedCapability.label
      : visualMetricOptions.find(
          (option) => option.value === visualFilters.metric,
        )?.label || "Metric";
  const spatialMetricLabel = scopedSpatialMetricOptions.find(
    (option) => option.value === spatialFilters.metric,
  )?.label || "Metric";
  const showVisualRankColumn =
    visualRankSortKey(visualFilters.metric) === "rankMetric";
  const visualMetaCols = compactTableMetaColumns(filteredVisualRows);
  const spatialMetaCols = compactTableMetaColumns(filteredSpatialRows);
  const compareMetaCols = compactTableMetaColumns(compareRows);
  const visualScopeLabel = visualBenchmarkLabel(visualFilters.benchmark);
  const visualIsCombined = visualFilters.benchmark === "all";
  const visualTableColumnCount =
    2 +
    visualMetaCols.size +
    (showVisualRankColumn ? 1 : 0) +
    (visualIsCombined
      ? 5
      : visualFilters.benchmark === "do_you_see_me"
        ? 3
        : 4);
  const visualTableTitle = visualIsCombined
    ? "Combined visual rankings"
    : `${visualScopeLabel} rankings`;
  const visualScoreChartTitle = visualFilters.benchmark === "do_you_see_me"
    ? "Perception accuracy"
    : visualFilters.benchmark === "minds_eye"
      ? "Cognition accuracy"
      : "Perception vs cognition";
  const visualCapabilityScope = visualFilters.benchmark === "do_you_see_me"
    ? "perception"
    : visualFilters.benchmark === "minds_eye"
      ? "cognition"
      : "all";
  const visualDiagnosticsDescription = visualFilters.benchmark === "do_you_see_me"
    ? "Compare 2D and 3D transfer, controlled difficulty sensitivity, and capability floors for selected perception models."
    : visualFilters.benchmark === "minds_eye"
      ? "Compare Abstraction, Relation, and Transformation performance and capability floors for selected cognition models."
      : "Compare dimensional transfer, difficulty sensitivity, cognitive capability balance, and capability floors for selected models.";
  const spatialScopeLabel =
    spatialFilters.dataset !== "all"
      ? spatialFilters.dataset
      : spatialFilters.datasetType !== "all"
        ? `${spatialFilters.datasetType} spatial benchmarks`
        : "All spatial benchmarks";
  const compareScopeLabel = compareBenchmarkLabel(compareBenchmark);
  const visualEmptyMessage = loadStatus === "error"
    ? "Rankings are unavailable. Use Retry above."
    : visualRows.length === 0
      ? "No visual benchmark submissions are published yet."
      : "No models match these filters.";
  const spatialEmptyMessage = loadStatus === "error"
    ? "Rankings are unavailable. Use Retry above."
    : spatialRows.length === 0
      ? "No spatial submissions are published yet."
      : "No models match these filters.";
  const compareEmptyMessage = loadStatus === "error"
    ? "Rankings are unavailable. Use Retry above."
    : availableCompareRows.length === 0
      ? "No models are available for this comparison yet."
      : selectedCompareModels.length === 0
        ? "Select at least one model to compare."
        : "No models match these filters.";
  const visualMeanMetric = mean(filteredVisualRows, (row) =>
    visualMetricValue(row, visualFilters.metric, visualFilters.capability),
  );
  const spatialMeanMetric = mean(filteredSpatialRows, (row) =>
    spatialMetricValue(
      row,
      spatialFilters.metric,
      spatialFilters.dataset,
      spatialScopedDatasets,
      spatialScopeActive,
    ),
  );
  const visualBestMetricPrefix =
    visualFilters.metric === "gap" || visualFilters.metric === "spread"
      ? "Smallest"
      : "Top";
  const spatialBestMetricPrefix = spatialFilters.metric === "shortcut" ? "Lowest" : "Top";

  const visualStats = [
    [filteredVisualRows.length, "Models ranked"],
    [
      visualHumanReference[visualFilters.benchmark].value,
      visualHumanReference[visualFilters.benchmark].label,
    ],
    [
      filteredVisualRows[0]
        ? visualMetricDisplay(
            filteredVisualRows[0],
            visualFilters.metric,
            visualFilters.capability,
          )
        : "N/A",
      `${visualBestMetricPrefix} ${visualMetricLabel}`,
    ],
    [
      visualMeanDisplay(visualMeanMetric, visualFilters.metric),
      `Average ${visualMetricLabel}`,
    ],
  ];
  const spatialStats = [
    [filteredSpatialRows.length, "Models ranked"],
    [
      filteredSpatialRows[0]
        ? spatialMetricDisplay(
            filteredSpatialRows[0],
            spatialFilters.metric,
            spatialFilters.dataset,
            spatialScopedDatasets,
            spatialScopeActive,
          )
        : "N/A",
      `${spatialBestMetricPrefix} ${spatialMetricLabel}`,
    ],
    [
      filteredSpatialRows.filter((row) => row.diagnostics).length,
      "Diagnostic profiles",
    ],
    [
      spatialMeanDisplay(spatialMeanMetric, spatialFilters.metric),
      `Average ${spatialMetricLabel}`,
    ],
  ];
  const compareAverage = compareBenchmark === "do_you_see_me"
    ? mean(compareRows, (row) => row.visual?.perception_accuracy)
    : compareBenchmark === "minds_eye"
      ? mean(compareRows, (row) => cognitionAccuracy(row.visual))
      : compareBenchmark === "spatial"
        ? mean(compareRows, (row) => row.spatial?.macro_accuracy ?? row.spatial?.accuracy)
        : mean(compareRows, (row) => row.visual?.vci);
  const compareAverageLabel = compareBenchmark === "do_you_see_me"
    ? "Average perception"
    : compareBenchmark === "minds_eye"
      ? "Average cognition"
      : compareBenchmark === "spatial"
        ? "Average spatial score"
        : "Average VPCI";
  const compareAverageDisplay = compareBenchmark === "all"
    ? fmtVci(compareAverage)
    : fmtPct(compareAverage);
  const compareCoverage = compareBenchmark === "do_you_see_me"
    ? compareRows.filter((row) => hasCognition(row.visual)).length
    : compareBenchmark === "minds_eye"
      ? compareRows.filter((row) => hasPerception(row.visual)).length
      : compareBenchmark === "spatial"
        ? compareRows.filter((row) => row.spatial?.diagnostics).length
        : compareRows.filter((row) => row.visual?.complete && row.spatial).length;
  const compareCoverageLabel = compareBenchmark === "do_you_see_me"
    ? "Also have cognition"
    : compareBenchmark === "minds_eye"
      ? "Also have perception"
      : compareBenchmark === "spatial"
        ? "Diagnostic profiles"
        : "Complete cross track profiles";
  const compareStats = [
    [compareRows.length, "Models selected"],
    [availableCompareRows.length, "Models available"],
    [compareAverageDisplay, compareAverageLabel],
    [compareCoverage, compareCoverageLabel],
  ];

  const handleVisualSort = (key, defaultDirection = "desc") =>
    setVisualSort((current) => nextSortState(current, key, defaultDirection));
  const handleSpatialSort = (key, defaultDirection = "desc") =>
    setSpatialSort((current) => nextSortState(current, key, defaultDirection));
  const handleCompareSort = (key, defaultDirection = "desc") =>
    setCompareSort((current) => nextSortState(current, key, defaultDirection));

  const setVisualBenchmark = (benchmark) => {
    const metric = visualDefaultMetric(benchmark);
    setVisualFilters((current) => ({
      ...current,
      benchmark,
      capability: "all",
      metric,
    }));
    setVisualSort({
      key: visualRankSortKey(metric),
      direction: metricSortDirection(metric),
    });
  };
  const setVisualMetric = (metric) => {
    setVisualFilters((current) => ({
      ...current,
      capability: metric === "capability" ? current.capability : "all",
      metric,
    }));
    setVisualSort({
      key: visualRankSortKey(metric),
      direction: metricSortDirection(metric),
    });
  };
  const setVisualCapability = (capability) => {
    const metric = capability === "all"
      ? visualDefaultMetric(visualFilters.benchmark)
      : "capability";
    setVisualFilters((current) => ({ ...current, capability, metric }));
    setVisualSort({
      key: visualRankSortKey(metric),
      direction: metricSortDirection(metric),
    });
  };
  const setSpatialMetric = (metric) => {
    setSpatialFilters((current) => ({ ...current, metric }));
    setSpatialSort({
      key: spatialRankSortKey(metric),
      direction: metricSortDirection(metric),
    });
  };
  const setSpatialBenchmark = (dataset) => {
    const metric = "accuracy";
    setSpatialFilters((current) => ({
      ...current,
      dataset,
      metric,
    }));
    setSpatialSort({
      key: spatialRankSortKey(metric),
      direction: metricSortDirection(metric),
    });
  };
  const setSpatialType = (datasetType) => {
    const metric = datasetType === "all" ? "macro" : "accuracy";
    setSpatialFilters((current) => ({
      ...current,
      datasetType,
      dataset: "all",
      metric,
    }));
    setSpatialSort({
      key: spatialRankSortKey(metric),
      direction: metricSortDirection(metric),
    });
  };
  const resetSpatialScope = () => {
    setSpatialFilters((current) => ({
      ...current,
      dataset: "all",
      datasetType: "all",
      metric: "macro",
    }));
    setSpatialSort({
      key: "macro",
      direction: "desc",
    });
  };
  const setCompareBenchmarkScope = (benchmark) => {
    setCompareBenchmark(benchmark);
    setCompareSort({
      key:
        benchmark === "do_you_see_me"
          ? "perception"
          : benchmark === "minds_eye"
            ? "imagery"
            : benchmark === "spatial"
              ? "spatial"
              : "vci",
      direction: "desc",
    });
  };
  const addCompareModel = (name) =>
    setSelectedCompareModels((current) =>
      current.includes(name) ? current : [...current, name],
    );
  const removeCompareModel = (name) =>
    setSelectedCompareModels((current) =>
      current.filter((item) => item !== name),
    );
  const resetCompareModels = () =>
    setSelectedCompareModels(
      availableCompareRows
        .slice(0, DEFAULT_COMPARE_MODELS)
        .map((row) => row.model_name),
    );

  const visualChips = [
    visualFilters.search && {
      label: `Search: ${visualFilters.search}`,
      onClear: () =>
        setVisualFilters((current) => ({ ...current, search: "" })),
    },
    visualFilters.benchmark !== "all" && {
      label: visualScopeLabel,
      onClear: () => setVisualBenchmark("all"),
    },
    visualFilters.capability !== "all" &&
      selectedCapability && {
        label: `Capability rank: ${selectedCapability.label}`,
        onClear: () => setVisualCapability("all"),
      },
  ];
  const spatialChips = [
    spatialFilters.search && {
      label: `Search: ${spatialFilters.search}`,
      onClear: () =>
        setSpatialFilters((current) => ({ ...current, search: "" })),
    },
    spatialScopeActive && {
      label: spatialScopeLabel,
      onClear: resetSpatialScope,
    },
    spatialFilters.diagnostics !== "all" && {
      label: "Diagnostics available",
      onClear: () =>
        setSpatialFilters((current) => ({ ...current, diagnostics: "all" })),
    },
  ];
  const compareChips = [
    compareBenchmark !== "all" && {
      label: compareScopeLabel,
      onClear: () => setCompareBenchmarkScope("all"),
    },
  ];

  const isSampleData = useMemo(() => {
    const names = [...visualRows, ...spatialRows].map(
      (row) => (row.model_name || "").trim(),
    );
    return names.length > 0 && names.every((name) => /[-_\s]demo$/i.test(name));
  }, [visualRows, spatialRows]);

  useEffect(() => {
    const detail = loadError
      ? {
          message: loadError,
          tone: "negative",
          action: () => setReloadKey((value) => value + 1),
          actionLabel: "Retry",
        }
      : isSampleData
        ? {
            message: "These rankings use illustrative sample entries. Submitted models will replace them as verified results become available.",
            tone: "warning",
          }
        : null;
    window.dispatchEvent(new CustomEvent("app-warning", { detail }));
  }, [isSampleData, loadError]);

  return (
    <>
      <section className={leaderboardSurfaceClasses}>
        <div className={ui.sectionFrame}>
          <div className={ui.sectionBand}>
            <div className="max-w-copy">
              <div className={ui.sectionTag}>Leaderboard</div>
              <h1 className={ui.heading1}>MS VISTA leaderboard rankings</h1>
              <p className={cn(ui.lede, "mt-4")}>
                Rank, filter, and compare models across visual perception, cognition, and spatial reasoning.
              </p>
            </div>
          </div>
          <div className="px-6 pb-2 pt-5 lg:px-8">
            <TabBar tabs={trackTabs} active={tab} onChange={setTab} />
          </div>
          <div className="px-6 pb-8 pt-3 lg:px-8 lg:pb-10 lg:pt-4">

          {loadStatus === "loading" && <div className={ui.message} role="status">Loading current rankings and benchmark details...</div>}
          {tab === "vc" && (
            <section className="tab-panel is-active">
              <DashboardStats items={visualStats} />
              <ControlDeck
                eyebrow="Visual scope"
                title={visualScopeLabel}
                chips={visualChips}
              >
                <SegmentControl
                  label="Visual benchmark scope"
                  options={visualBenchmarkOptions}
                  value={visualFilters.benchmark}
                  onChange={setVisualBenchmark}
                />
                <div className="primary-controls">
                  <FilterField label="Rank by">
                    <select
                      value={visualFilters.metric}
                      onChange={(event) => setVisualMetric(event.target.value)}
                    >
                      {scopedVisualMetricOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </FilterField>
                  <FilterField label="Rank by capability">
                    <select
                      value={visualFilters.capability}
                      onChange={(event) =>
                        setVisualCapability(event.target.value)
                      }
                    >
                      <option value="all">None</option>
                      {visibleCapabilities.map((capability) => (
                        <option key={capability.id} value={capability.id}>
                          {capability.label}
                        </option>
                      ))}
                    </select>
                  </FilterField>
                  <FilterField label="Search">
                    <input
                      value={visualFilters.search}
                      onChange={(event) =>
                        setVisualFilters((current) => ({
                          ...current,
                          search: event.target.value,
                        }))
                      }
                      placeholder="Model or organization"
                    />
                  </FilterField>
                </div>
              </ControlDeck>

              <section className="leaderboard-table border-t-2 border-foreground" aria-labelledby="visual-rankings-title">
                <LeaderboardTableHeading
                  count={filteredVisualRows.length}
                  id="visual-rankings-title"
                  metric={sortSummary(visualSort, visualMetricLabel)}
                  scope={visualScopeLabel}
                  title={visualTableTitle}
                />
                <div className="table-wrap !border-t-0">
                <table className="lb-table" aria-labelledby="visual-rankings-title">
                  <thead>
                    <tr>
                      <th className="rank-col !text-center" data-tip={RANK_DESCRIPTION} title={RANK_DESCRIPTION}>##</th>
                      <SortHeader
                        className="model-col"
                        label="Model"
                        sortKey="model"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      {visualMetaCols.has("org") && (
                        <SortHeader
                          className="w-28 min-w-28 max-w-28"
                          label="Org"
                          sortKey="org"
                          sort={visualSort}
                          onSort={handleVisualSort}
                          defaultDirection="asc"
                        />
                      )}
                      {visualMetaCols.has("params") && (
                        <SortHeader
                          label="Params"
                          sortKey="params"
                          sort={visualSort}
                          onSort={handleVisualSort}
                          defaultDirection="asc"
                        />
                      )}
                      {showVisualRankColumn && (
                        <SortHeader
                          label={visualMetricLabel}
                          sortKey="rankMetric"
                          sort={visualSort}
                          onSort={handleVisualSort}
                          className="num"
                          defaultDirection={metricSortDirection(visualFilters.metric)}
                        />
                      )}
                      {visualIsCombined ? (
                        <>
                          <SortHeader
                            label="VPCI"
                            sortKey="vci"
                            sort={visualSort}
                            onSort={handleVisualSort}
                            className="num w-[9rem] min-w-[9rem] max-w-[9rem]"
                          />
                          <SortHeader
                            label="Perception avg"
                            sortKey="perception"
                            sort={visualSort}
                            onSort={handleVisualSort}
                            className="num w-[9rem] min-w-[9rem] max-w-[9rem]"
                          />
                          <SortHeader
                            label="Cognition avg"
                            sortKey="imagery"
                            sort={visualSort}
                            onSort={handleVisualSort}
                            className="num w-[9rem] min-w-[9rem] max-w-[9rem]"
                          />
                          <SortHeader
                            label="Gap"
                            sortKey="gap"
                            sort={visualSort}
                            onSort={handleVisualSort}
                            className="num w-24 min-w-24 max-w-24"
                            defaultDirection="asc"
                          />
                          <SortHeader
                            label="Spread"
                            sortKey="spread"
                            sort={visualSort}
                            onSort={handleVisualSort}
                            className="num w-[7rem] min-w-[7rem] max-w-[7rem]"
                            defaultDirection="asc"
                          />
                        </>
                      ) : visualFilters.benchmark === "do_you_see_me" ? (
                        <>
                          <SortHeader label="Overall avg" sortKey="perception" sort={visualSort} onSort={handleVisualSort} className="num" />
                          <SortHeader label="2D avg" sortKey="perception_2d" sort={visualSort} onSort={handleVisualSort} className="num" />
                          <SortHeader label="3D avg" sortKey="perception_3d" sort={visualSort} onSort={handleVisualSort} className="num" />
                        </>
                      ) : (
                        <>
                          <SortHeader label="Overall avg" sortKey="imagery" sort={visualSort} onSort={handleVisualSort} className="num" />
                          <SortHeader label="A" sortKey="art_abstraction" sort={visualSort} onSort={handleVisualSort} className="num" />
                          <SortHeader label="R" sortKey="art_relation" sort={visualSort} onSort={handleVisualSort} className="num" />
                          <SortHeader label="T" sortKey="art_transformation" sort={visualSort} onSort={handleVisualSort} className="num" />
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredVisualRows.length ? (
                      filteredVisualRows.map((row, index) => (
                        <tr
                          className="clickable"
                          key={row.model_name}
                          onClick={() => setReportModel(row.model_name)}
                        >
                          <td className="rank-col">
                            <RankBadge rank={index + 1} />
                          </td>
                          <td className="model-col">
                            <button
                              className="w-full text-left font-semibold text-foreground outline-none hover:text-brand-strong focus-visible:underline focus-visible:underline-offset-4"
                              type="button"
                              aria-label={`View model report for ${row.model_name}`}
                              onClick={(event) => {
                                event.stopPropagation();
                                setReportModel(row.model_name);
                              }}
                            >
                              {row.model_name}
                            </button>
                          </td>
                          {visualMetaCols.has("org") && (
                            <td
                              className="w-28 min-w-28 max-w-28 break-words"
                              title={modelOrg(row.model_meta)}
                            >
                              {modelOrg(row.model_meta)}
                            </td>
                          )}
                          {visualMetaCols.has("params") && (
                            <td>{modelParams(row.model_meta)}</td>
                          )}
                          {showVisualRankColumn && (
                            <td className="num vci-val">
                              {visualMetricDisplay(
                                row,
                                visualFilters.metric,
                                visualFilters.capability,
                              )}
                            </td>
                          )}
                          {visualIsCombined ? (
                            <>
                              <td className="num w-[9rem] min-w-[9rem] max-w-[9rem]">{fmtVci(row.vci)}</td>
                              <td className="num w-[9rem] min-w-[9rem] max-w-[9rem]">
                                {fmtPct(row.perception_accuracy)}
                              </td>
                              <td className="num w-[9rem] min-w-[9rem] max-w-[9rem]">
                                {fmtPct(cognitionAccuracy(row))}
                              </td>
                              <td
                                className={`num w-24 min-w-24 max-w-24 ${(visualGap(row) ?? 0) < 0 ? "neg" : "pos"}`}
                              >
                                {visualGap(row) == null
                                  ? "N/A"
                                  : `${visualGap(row) >= 0 ? "+" : ""}${(visualGap(row) * 100).toFixed(1)}`}
                              </td>
                              <td className="num w-[7rem] min-w-[7rem] max-w-[7rem]">
                                {combinedTaskSpread(row) == null
                                  ? "-"
                                  : (combinedTaskSpread(row) * 100).toFixed(1)}
                              </td>
                            </>
                          ) : visualFilters.benchmark === "do_you_see_me" ? (
                            <>
                              <td className="num">{fmtPct(row.perception_accuracy)}</td>
                              <td className="num">{fmtPct(analysisAccuracy(row.perception_dimensions, "2D"))}</td>
                              <td className="num">{fmtPct(analysisAccuracy(row.perception_dimensions, "3D"))}</td>
                            </>
                          ) : (
                            <>
                              <td className="num">{fmtPct(cognitionAccuracy(row))}</td>
                              <td className="num">{fmtPct(analysisAccuracy(cognitionArtGroups(row), "abstraction"))}</td>
                              <td className="num">{fmtPct(analysisAccuracy(cognitionArtGroups(row), "relation"))}</td>
                              <td className="num">{fmtPct(analysisAccuracy(cognitionArtGroups(row), "transformation"))}</td>
                            </>
                          )}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan={visualTableColumnCount}
                          className="empty-row"
                        >
                          {visualEmptyMessage}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
                </div>
              </section>

              <div className="mb-5 border-t border-border" aria-hidden="true" />

              <div className="dashboard-grid two-col">
                <div className="viz-card wide">
                  <h3>Capability profile</h3>
                  <VisualCapabilityChart
                    benchmark={visualFilters.benchmark}
                    rows={filteredVisualRows}
                    capabilities={visibleCapabilities}
                  />
                </div>
                <div className="viz-card wide">
                  <div className="mb-3 flex items-start justify-between gap-4 max-sm:flex-col">
                    <h3 className="!mb-0">{visualScoreChartTitle}</h3>
                    <ChartModelSelector
                      label={visualScoreChartTitle}
                      onReset={perceptionChartSelection.reset}
                      onToggle={perceptionChartSelection.toggle}
                      rows={perceptionChartRows}
                      selected={perceptionChartSelection.selected}
                    />
                  </div>
                  <PerceptionCognitionChart
                    benchmark={visualFilters.benchmark}
                    rows={perceptionChartRows}
                    selected={perceptionChartSelection.selected}
                  />
                </div>
                <div className="viz-card wide">
                  <div className="mb-3 flex items-start justify-between gap-4 max-sm:flex-col">
                    <h3 className="!mb-0">Model capability trace</h3>
                    <ChartModelSelector
                      label="Model capability trace"
                      onReset={capabilityChartSelection.reset}
                      onToggle={capabilityChartSelection.toggle}
                      rows={capabilityChartRows}
                      selected={capabilityChartSelection.selected}
                    />
                  </div>
                  <CapabilityRadar
                    rows={capabilityChartRows}
                    scope={visualCapabilityScope}
                    selected={capabilityChartSelection.selected}
                  />
                </div>
                <div className="viz-card wide !p-0">
                  <div className="flex items-start justify-between gap-4 p-6 max-sm:flex-col">
                    <div>
                      <h3 className="!mb-0">Performance diagnostics</h3>
                      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
                        {visualDiagnosticsDescription}
                      </p>
                    </div>
                    {paperAnalysisRows.length > 0 && (
                      <ChartModelSelector
                        label="Performance diagnostics"
                        onReset={paperAnalysisSelection.reset}
                        onToggle={paperAnalysisSelection.toggle}
                        rows={paperAnalysisRows}
                        selected={paperAnalysisSelection.selected}
                      />
                    )}
                  </div>
                  <div className="grid grid-cols-1 border-t border-border md:grid-cols-2">
                    {visualFilters.benchmark !== "minds_eye" && (
                      <div className="min-w-0 p-6">
                        <h4 className="font-display text-base font-semibold text-foreground">2D and 3D transfer</h4>
                        <p className="mt-1 text-sm text-muted">Distance from the parity line shows how consistently perception transfers between dimensions.</p>
                        <DimensionTransferChart rows={paperAnalysisRows} selected={paperAnalysisSelection.selected} />
                      </div>
                    )}
                    {visualFilters.benchmark !== "minds_eye" && (
                      <div className="min-w-0 border-t border-border p-6 md:border-l md:border-t-0">
                        <h4 className="font-display text-base font-semibold text-foreground">Difficulty response</h4>
                        <p className="mt-1 text-sm text-muted">Accuracy across the controlled easy, medium, and hard perception samples.</p>
                        <DifficultyResponseChart rows={paperAnalysisRows} selected={paperAnalysisSelection.selected} />
                      </div>
                    )}
                    {visualFilters.benchmark !== "do_you_see_me" && (
                      <div className={cn(
                        "min-w-0 p-6",
                        visualFilters.benchmark === "all" && "border-t border-border",
                      )}>
                        <h4 className="font-display text-base font-semibold text-foreground">ART capability summary</h4>
                        <p className="mt-1 text-sm text-muted">A compact view of Abstraction, Relation, and Transformation performance.</p>
                        <ArtSummaryChart rows={paperAnalysisRows} selected={paperAnalysisSelection.selected} />
                      </div>
                    )}
                    <div className={cn(
                      "min-w-0 border-border p-6",
                      visualFilters.benchmark !== "minds_eye" && "border-t",
                      visualFilters.benchmark === "all" && "md:border-l",
                      visualFilters.benchmark === "minds_eye" && "border-t md:border-l md:border-t-0",
                    )}>
                      <h4 className="font-display text-base font-semibold text-foreground">Average and weakest capability</h4>
                      <p className="mt-1 text-sm text-muted">Models toward the upper right combine stronger average capability accuracy with a higher weakest score.</p>
                      <CapabilityFloorChart
                        rows={paperAnalysisRows}
                        scope={visualCapabilityScope}
                        selected={paperAnalysisSelection.selected}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {tab === "spatial" && (
            <section className="tab-panel is-active">
              <DashboardStats items={spatialStats} />
              <ControlDeck
                eyebrow="Spatial scope"
                title={spatialScopeLabel}
                chips={spatialChips}
              >
                {spatialTypeOptions.length > 2 && (
                  <SegmentControl
                    label="Spatial benchmark type"
                    options={spatialTypeOptions}
                    value={spatialFilters.datasetType}
                    onChange={setSpatialType}
                  />
                )}
                <div className="primary-controls">
                  {spatialBenchmarkOptions.length > 2 && (
                    <FilterField label="Benchmark">
                      <select
                        value={spatialFilters.dataset}
                        onChange={(event) =>
                          setSpatialBenchmark(event.target.value)
                        }
                      >
                        {spatialBenchmarkOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </FilterField>
                  )}
                  <FilterField label="Rank by">
                    <select
                      value={spatialFilters.metric}
                      onChange={(event) => setSpatialMetric(event.target.value)}
                    >
                      {scopedSpatialMetricOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </FilterField>
                  <FilterField label="Search">
                    <input
                      value={spatialFilters.search}
                      onChange={(event) =>
                        setSpatialFilters((current) => ({
                          ...current,
                          search: event.target.value,
                        }))
                      }
                      placeholder="Model or organization"
                    />
                  </FilterField>
                  {hasSpatialDiagnostics && (
                    <FilterField label="Diagnostics">
                      <select
                        value={spatialFilters.diagnostics}
                        onChange={(event) =>
                          setSpatialFilters((current) => ({
                            ...current,
                            diagnostics: event.target.value,
                          }))
                        }
                      >
                        <option value="all">All submissions</option>
                        <option value="full">Diagnostics available</option>
                      </select>
                    </FilterField>
                  )}
                </div>
              </ControlDeck>

              <section className="leaderboard-table border-t-2 border-foreground" aria-labelledby="spatial-rankings-title">
                <LeaderboardTableHeading
                  count={filteredSpatialRows.length}
                  id="spatial-rankings-title"
                  metric={sortSummary(spatialSort, spatialMetricLabel)}
                  scope={spatialScopeLabel}
                  title="Spatial model rankings"
                />
                <div className="table-wrap !border-t-0">
                <table className="lb-table" aria-labelledby="spatial-rankings-title">
                  <thead>
                    <tr>
                      <th className="rank-col !text-center" data-tip={RANK_DESCRIPTION} title={RANK_DESCRIPTION}>##</th>
                      <SortHeader
                        className="model-col"
                        label="Model"
                        sortKey="model"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      {spatialMetaCols.has("org") && (
                        <SortHeader
                          className="w-28 min-w-28 max-w-28"
                          label="Org"
                          sortKey="org"
                          sort={spatialSort}
                          onSort={handleSpatialSort}
                          defaultDirection="asc"
                        />
                      )}
                      {spatialMetaCols.has("params") && (
                        <SortHeader
                          label="Params"
                          sortKey="params"
                          sort={spatialSort}
                          onSort={handleSpatialSort}
                          defaultDirection="asc"
                        />
                      )}
                      <SortHeader
                        label={
                          spatialScopeActive
                            ? "Scope avg."
                            : "Main micro"
                        }
                        sortKey="scope"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                      />
                      <SortHeader
                        label="Macro avg"
                        sortKey="macro"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                      />
                      {hasSpatialDiagnostics && (
                        <>
                          <SortHeader
                            label="CoT delta"
                            sortKey="cot_delta"
                            sort={spatialSort}
                            onSort={handleSpatialSort}
                            className="num"
                          />
                          <SortHeader
                            label="Shortcut ↓"
                            sortKey="shortcut"
                            sort={spatialSort}
                            onSort={handleSpatialSort}
                            className="num"
                            defaultDirection="asc"
                          />
                          <SortHeader
                            label="NI++ resist. ↑"
                            sortKey="hallucination"
                            sort={spatialSort}
                            onSort={handleSpatialSort}
                            className="num"
                          />
                          <SortHeader
                            label="Diagnostics"
                            sortKey="coverage"
                            sort={spatialSort}
                            onSort={handleSpatialSort}
                          />
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSpatialRows.length ? (
                      filteredSpatialRows.map((row, index) => (
                        <tr
                          className="clickable"
                          key={row.model_name}
                          onClick={() => setReportModel(row.model_name)}
                        >
                          <td className="rank-col">
                            <RankBadge rank={index + 1} />
                          </td>
                          <td className="model-col">
                            <button
                              className="w-full text-left font-semibold text-foreground outline-none hover:text-brand-strong focus-visible:underline focus-visible:underline-offset-4"
                              type="button"
                              aria-label={`View model report for ${row.model_name}`}
                              onClick={(event) => {
                                event.stopPropagation();
                                setReportModel(row.model_name);
                              }}
                            >
                              {row.model_name}
                            </button>
                            {row.evidence_url && (
                              <a
                                className="mt-1 inline-flex items-center gap-1 text-xs text-muted underline decoration-border-strong underline-offset-4 hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand"
                                href={apiUrl(row.evidence_url)}
                                target="_blank"
                                rel="noreferrer noopener"
                                onClick={(event) => event.stopPropagation()}
                              >
                                Public evidence <ExternalLink size={12} aria-hidden="true" />
                              </a>
                            )}
                          </td>
                          {spatialMetaCols.has("org") && (
                            <td
                              className="w-28 min-w-28 max-w-28 break-words"
                              title={modelOrg(row.model_meta)}
                            >
                              {modelOrg(row.model_meta)}
                            </td>
                          )}
                          {spatialMetaCols.has("params") && (
                            <td>{modelParams(row.model_meta)}</td>
                          )}
                          <td className="num">
                            {fmtPct(
                              spatialScopeActive
                                ? spatialScopeAccuracy(
                                    row,
                                    spatialScopedDatasets,
                                  )
                                : row.accuracy,
                            )}
                          </td>
                          <td className="num">
                            {fmtPct(row.macro_accuracy)}
                          </td>
                          {hasSpatialDiagnostics && (
                            <>
                              <td
                                className={`num ${(row.diagnostics?.cot_delta ?? 0) < 0 ? "neg" : "pos"}`}
                              >
                                {row.diagnostics?.cot_delta == null
                                  ? "N/A"
                                  : `${fmtDelta(row.diagnostics.cot_delta)}`}
                              </td>
                              <td className="num">
                                {fmtPct(row.diagnostics?.shortcut_score)}
                              </td>
                              <td className="num">
                                {fmtPct(row.diagnostics?.hallucination_resistance)}
                              </td>
                              <td>
                                <DiagnosticsChip diagnostics={row.diagnostics} />
                              </td>
                            </>
                          )}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan={
                            4 + spatialMetaCols.size + (hasSpatialDiagnostics ? 4 : 0)
                          }
                          className="empty-row"
                        >
                          {spatialEmptyMessage}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
                </div>
              </section>

              <div className="mb-5 border-t border-border" aria-hidden="true" />

              <div className="dashboard-grid two-col">
                <div className="viz-card wide">
                  <h3>Accuracy by dataset</h3>
                  <SpatialDatasetChart
                    rows={filteredSpatialRows}
                    datasets={spatialScopedDatasets}
                  />
                </div>
                {hasSpatialDiagnostics && (
                  <>
                    <div className="viz-card">
                      <h3>CoT effect</h3>
                      <CotComparisonChart rows={filteredSpatialRows} />
                    </div>
                    <div className="viz-card">
                      <h3>Grounding robustness</h3>
                      <RobustnessChart rows={filteredSpatialRows} />
                    </div>
                  </>
                )}
              </div>
            </section>
          )}

          {tab === "compare" && (
            <section className="tab-panel is-active">
              <DashboardStats items={compareStats} />
              <ControlDeck
                eyebrow="Comparison scope"
                title={compareScopeLabel}
                chips={compareChips}
              >
                <SegmentControl
                  label="Comparison benchmark scope"
                  options={availableCompareBenchmarkOptions}
                  value={compareBenchmark}
                  onChange={setCompareBenchmarkScope}
                />
                <div className="primary-controls narrow">
                  <FilterField label="Find model to add">
                    <input
                      value={compareSearch}
                      onChange={(event) => setCompareSearch(event.target.value)}
                      placeholder="Model or organization"
                    />
                  </FilterField>
                </div>
                <CompareModelPicker
                  candidateRows={compareCandidateRows}
                  onAdd={addCompareModel}
                  onRemove={removeCompareModel}
                  onReset={resetCompareModels}
                  selectedNames={selectedCompareModels}
                  selectedRows={compareRows}
                />
              </ControlDeck>
              <section className="leaderboard-table border-t-2 border-foreground" aria-labelledby="compare-rankings-title">
                <LeaderboardTableHeading
                  count={compareRows.length}
                  id="compare-rankings-title"
                  metric={sortSummary(compareSort, "Selected metric")}
                  scope={compareScopeLabel}
                  title="Selected model comparison"
                />
                <div className="table-wrap !border-t-0">
                <table className="lb-table" aria-labelledby="compare-rankings-title">
                  <thead>
                    <tr>
                      <SortHeader
                        className="model-col model-col-first"
                        label="Model"
                        sortKey="model"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      {compareMetaCols.has("org") && (
                        <SortHeader
                          className="w-28 min-w-28 max-w-28"
                          label="Org"
                          sortKey="org"
                          sort={compareSort}
                          onSort={handleCompareSort}
                          defaultDirection="asc"
                        />
                      )}
                      {compareMetaCols.has("params") && (
                        <SortHeader
                          label="Params"
                          sortKey="params"
                          sort={compareSort}
                          onSort={handleCompareSort}
                          defaultDirection="asc"
                        />
                      )}
                      {compareMetricKeys.map((key) => (
                        <SortHeader
                          className="num"
                          defaultDirection={key === "shortcut" ? "asc" : "desc"}
                          key={key}
                          label={compareMetricConfig[key].label}
                          onSort={handleCompareSort}
                          sort={compareSort}
                          sortKey={key}
                        />
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {compareRows.length ? (
                      compareRows.map((row) => (
                        <tr key={row.model_name}>
                          <td className="model-col model-col-first">
                            <strong>{row.model_name}</strong>
                          </td>
                          {compareMetaCols.has("org") && (
                            <td
                              className="w-28 min-w-28 max-w-28 break-words"
                              title={modelOrg(row.model_meta)}
                            >
                              {modelOrg(row.model_meta)}
                            </td>
                          )}
                          {compareMetaCols.has("params") && (
                            <td>{modelParams(row.model_meta)}</td>
                          )}
                          {compareMetricKeys.map((key) => {
                            const value = compareColumnValue(row, key);
                            return (
                              <td
                                className={cn(
                                  "num",
                                  key === "vci" && "vci-val",
                                  key === "cot_delta" && value != null
                                    && (value < 0 ? "neg" : "pos"),
                                )}
                                key={key}
                              >
                                {compareMetricDisplay(row, key)}
                              </td>
                            );
                          })}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan={1 + compareMetaCols.size + compareMetricKeys.length}
                          className="empty-row"
                        >
                          {compareEmptyMessage}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
                </div>
              </section>
              <div className="dashboard-grid">
                <div className="viz-card wide">
                  <h3>{compareBenchmark === "all" ? "Profile matrix across available tracks" : `${compareScopeLabel} profile matrix`}</h3>
                  <ComparisonChart rows={compareRows.slice(0, 14)} scope={compareBenchmark} />
                </div>
              </div>
            </section>
          )}
          </div>
        </div>
      </section>
      <ReportModalShim
        model={reportModel}
        onClose={() => setReportModel(null)}
      />
    </>
  );
}

function GroupsTable({ groups }) {
  const keys = Object.keys(groups || {});
  if (!keys.length) return <p className="text-sm text-muted">N/A</p>;
  return (
    <table className={cn(ui.table, "min-w-[420px]")}>
      <thead>
        <tr>
          <th>Capability / dataset</th>
          <th className={ui.tableNumber}>Accuracy</th>
          <th className={ui.tableNumber}>Correct / total</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((key) => (
          <tr key={key}>
            <td>{prettyLabel(key)}</td>
            <td className={ui.tableNumber}>{fmtPct(groups[key].accuracy)}</td>
            <td className={ui.tableNumber}>
              {groups[key].correct_samples}/{groups[key].total_samples}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReportInsightSummary({ tasks }) {
  const perceptionStats = groupStats(tasks.do_you_see_me?.groups);
  const cognitionStats = groupStats(tasks.minds_eye?.groups);
  const perceptionExtremes = groupExtremes(tasks.do_you_see_me?.groups);
  const cognitionExtremes = groupExtremes(tasks.minds_eye?.groups);
  const spatialDiagnostics = tasks.spatial?.diagnostics;
  const items = [
    tasks.do_you_see_me && {
      label: "Perception profile",
      value: fmtPct(tasks.do_you_see_me.macro_accuracy ?? perceptionStats.mean),
      detail: perceptionExtremes.weakest
        ? `Weakest capability: ${prettyLabel(perceptionExtremes.weakest.key)} at ${fmtPct(perceptionExtremes.weakest.accuracy)}. Task spread is ${fmtPct(tasks.do_you_see_me.task_spread ?? perceptionStats.std)}.`
        : "Per-capability accuracy is available below.",
    },
    tasks.minds_eye && {
      label: "Cognition profile",
      value: fmtPct(tasks.minds_eye.macro_accuracy ?? cognitionStats.mean),
      detail: cognitionExtremes.weakest
        ? `Weakest capability: ${prettyLabel(cognitionExtremes.weakest.key)} at ${fmtPct(cognitionExtremes.weakest.accuracy)}. Task spread is ${fmtPct(tasks.minds_eye.task_spread ?? cognitionStats.std)}.`
        : "Per-capability accuracy is available below.",
    },
    spatialDiagnostics && {
      label: "Spatial robustness profile",
      value: `${fmtDelta(spatialDiagnostics.cot_delta)} pts CoT change`,
      detail: `No image shortcut score ${fmtPct(spatialDiagnostics.shortcut_score)}; no image plus resistance ${fmtPct(spatialDiagnostics.hallucination_resistance)}.`,
    },
  ];
  return <AnalysisCards items={items} />;
}

function ReportMeta({ meta }) {
  if (!meta) return null;
  const safeUrl = (value) => (/^https?:\/\//i.test(String(value || "").trim()) ? String(value).trim() : null);
  const rows = [
    ["Organization", modelOrg(meta)],
    ["Access", modelType(meta)],
    ["Parameters", modelParams(meta)],
    ["Base model", modelBase(meta)],
  ].filter(([, value]) => value && value !== "\u2014");
  const paperUrl = safeUrl(meta.paper_url);
  const blocks = [
    ["Method", meta.method_description],
    ["Training data", meta.training_data],
  ].filter(([, value]) => value && String(value).trim());
  if (!rows.length && !paperUrl && !blocks.length) return null;
  return (
    <div className="mt-6">
      <h3 className={ui.heading3}>Model details</h3>
      {(rows.length > 0 || paperUrl) && (
        <table className={cn(ui.table, "mt-3 min-w-0")}>
          <tbody>
            {rows.map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{value}</td>
              </tr>
            ))}
            {paperUrl && (
              <tr>
                <td>Paper / report</td>
                <td>
                  <a className="underline" href={paperUrl} target="_blank" rel="noreferrer noopener">
                    {paperUrl}
                  </a>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}
      {blocks.map(([key, value]) => (
        <div className="mt-4 border-t border-border pt-4" key={key}>
          <h4 className="font-display font-bold">{key}</h4>
          <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-muted">{value}</p>
        </div>
      ))}
    </div>
  );
}

function ReportRunMeta({ task, benchmark }) {
  const meta = task?.model_meta;
  if (!meta) return null;
  const rows = [
    ["CoT used", modelCot(meta)],
    ["Method", meta.method_description],
    ["Prompt template", meta.prompt_template],
    ["Changes from previous submission", meta.changes_from_previous],
  ].filter(([, value]) => value && value !== "N/A" && String(value).trim());
  if (!rows.length) return null;
  return (
    <div className="mt-6 border-t border-border pt-5">
      <h3 className={ui.heading3}>{benchmark} run details</h3>
      <dl className="mt-3 grid gap-x-6 gap-y-4 sm:grid-cols-2">
        {rows.map(([key, value]) => (
          <div key={key} className="min-w-0">
            <dt className="text-xs font-semibold uppercase text-faint">{key}</dt>
            <dd className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-muted">
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function SpatialEvidenceLinks({ task }) {
  const evidence = task?.metadata?.public_evidence;
  if (!evidence?.available || !evidence?.url) return null;
  const linkClass = "inline-flex min-h-10 items-center gap-2 border border-border-strong bg-surface px-3 py-2 text-sm font-semibold text-foreground transition-colors hover:bg-surface-subtle focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand";
  return (
    <section className="mt-6 border-t border-border pt-5" aria-labelledby="public-evidence-title">
      <h3 className={ui.heading3} id="public-evidence-title">Public submission evidence</h3>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
        {evidence.notice || "Inspect the retained per sample outputs, aggregate report, and package hashes for this spatial submission."}
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        <a className={linkClass} href={apiUrl(evidence.url)} target="_blank" rel="noreferrer noopener">
          <ExternalLink size={16} aria-hidden="true" /> Evidence record
        </a>
        {evidence.answers_url && (
          <a className={linkClass} href={apiUrl(evidence.answers_url)} target="_blank" rel="noreferrer noopener">
            <ExternalLink size={16} aria-hidden="true" /> Per sample results
          </a>
        )}
        {evidence.archive_url && (
          <a className={linkClass} href={apiUrl(evidence.archive_url)}>
            <Download size={16} aria-hidden="true" /> Original package
          </a>
        )}
      </div>
    </section>
  );
}

function ReportModalShim({ model, onClose }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const dialogRef = useRef(null);
  const closeButtonRef = useRef(null);
  const previouslyFocusedRef = useRef(null);
  const titleId = useId();

  useEffect(() => {
    if (!model) return;
    setReport(null);
    setError("");
    getJSON(`/api/model/${encodeURIComponent(model)}/report`)
      .then(setReport)
      .catch((err) => setError(errorMessage(err, "The model report could not be loaded.")));
  }, [model, reloadKey]);

  useEffect(() => {
    if (!model) return undefined;
    previouslyFocusedRef.current = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(
        dialogRef.current?.querySelectorAll(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ) || [],
      ).filter((element) => !element.hasAttribute("hidden"));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previouslyFocusedRef.current?.focus?.();
    };
  }, [model, onClose]);

  if (!model) return null;
  const visual = report?.visual_cognition || {};
  const tasks = report?.tasks || {};
  return (
    <div className={ui.modal}>
      <div className={ui.modalBackdrop} onClick={onClose} aria-hidden="true" />
      <div
        className={ui.modalCard}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-busy={!report && !error}
        ref={dialogRef}
      >
        <button
          className={cn(ui.iconButton, "absolute right-3 top-3")}
          type="button"
          onClick={onClose}
          aria-label="Close model report"
          ref={closeButtonRef}
        >
          <X size={18} aria-hidden="true" />
        </button>
        <h2 className={ui.heading2} id={titleId}>{report?.model_name || model}</h2>
        {!report && !error && <p className="text-muted">Loading…</p>}
        {error && (
          <p className={cn(ui.message, ui.messageError)} role="alert">{error} <button type="button" className={ui.linkButton} onClick={() => setReloadKey((value) => value + 1)}>Retry</button></p>
        )}
        {report && (
          <>
            <div className={ui.kpiGrid}>
              <div className={ui.kpi}>
                <div className={ui.kpiLabel}>VPCI score</div>
                <div className={ui.kpiValue}>{fmtVci(visual.vci)}</div>
              </div>
              <div className={ui.kpi}>
                <div className={ui.kpiLabel}>Perception accuracy</div>
                <div className={ui.kpiValue}>
                  {fmtPct(visual.perception_accuracy)}
                </div>
              </div>
              <div className={ui.kpi}>
                <div className={ui.kpiLabel}>Cognition accuracy</div>
                <div className={ui.kpiValue}>{fmtPct(cognitionAccuracy(visual))}</div>
              </div>
              <div className={ui.kpi}>
                <div className={ui.kpiLabel}>Spatial standard accuracy</div>
                <div className={ui.kpiValue}>
                  {tasks.spatial ? fmtPct(tasks.spatial.macro_accuracy ?? tasks.spatial.accuracy) : "N/A"}
                </div>
              </div>
            </div>
            <ReportInsightSummary tasks={tasks} />
            <ReportMeta meta={report.model_meta} />
            {tasks.do_you_see_me && (
              <>
                <ReportRunMeta task={tasks.do_you_see_me} benchmark="Do You See Me" />
                <h3 className={cn(ui.heading3, "mt-6")}>Do You See Me capabilities</h3>
                <div className={ui.tableWrap}><GroupsTable groups={tasks.do_you_see_me.groups} /></div>
              </>
            )}
            {tasks.minds_eye && (
              <>
                <ReportRunMeta task={tasks.minds_eye} benchmark="Mind's Eye" />
                <h3 className={cn(ui.heading3, "mt-6")}>Mind's Eye capabilities</h3>
                <div className={ui.tableWrap}><GroupsTable groups={tasks.minds_eye.groups} /></div>
              </>
            )}
            {tasks.spatial && (
              <>
                <ReportRunMeta task={tasks.spatial} benchmark="Spatial" />
                <SpatialEvidenceLinks task={tasks.spatial} />
                <h3 className={cn(ui.heading3, "mt-6")}>Spatial datasets</h3>
                <div className={ui.tableWrap}><GroupsTable groups={tasks.spatial.groups} /></div>
                {tasks.spatial.diagnostics && (
                  <div className={ui.kpiGrid}>
                    <div className={ui.kpi}>
                      <div className={ui.kpiLabel}>CoT accuracy change</div>
                      <div
                        className={cn(ui.kpiValue, tasks.spatial.diagnostics.cot_delta < 0 ? "text-negative" : "text-positive")}
                      >
                        {fmtDelta(tasks.spatial.diagnostics.cot_delta)} pts
                      </div>
                    </div>
                    <div className={ui.kpi}>
                      <div className={ui.kpiLabel}>No image shortcut score</div>
                      <div className={ui.kpiValue}>
                        {fmtPct(tasks.spatial.diagnostics.shortcut_score)}
                      </div>
                    </div>
                    <div className={ui.kpi}>
                      <div className={ui.kpiLabel}>No image plus resistance</div>
                      <div className={ui.kpiValue}>
                        {fmtPct(
                          tasks.spatial.diagnostics.hallucination_resistance,
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
