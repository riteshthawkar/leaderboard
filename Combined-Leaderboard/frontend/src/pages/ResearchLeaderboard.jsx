import { useEffect, useMemo, useState } from "react";
import { BarChart, CapabilityRadar, chartPalette } from "@/components/Charts";
import { PageHero } from "@/components/Hero";
import { TabBar } from "@/components/ui/tabs";
import { getJSON } from "@/lib/api";
import { fmtDelta, fmtPct, fmtVci, modelType, prettyLabel } from "@/lib/utils";

const trackTabs = [
  { id: "vc", label: "Visual Perception and Cognition Index" },
  { id: "spatial", label: "Spatial Reasoning Index" },
  { id: "compare", label: "Compare Models" },
];

const MAX_COMPARE_MODELS = 4;

const visualMetricOptions = [
  { value: "vci", label: "VCI" },
  { value: "perception", label: "Perception" },
  { value: "imagery", label: "Imagery" },
  { value: "gap", label: "Layer gap" },
  { value: "capability", label: "Selected capability" },
];

const visualBenchmarkOptions = [
  { value: "all", label: "All visual benchmarks" },
  { value: "do_you_see_me", label: "Do-You-See-Me" },
  { value: "minds_eye", label: "Mind's-Eye" },
];

const spatialTypeOptions = [
  { value: "all", label: "All spatial" },
  { value: "2D", label: "2D" },
  { value: "3D", label: "3D" },
  { value: "dynamic", label: "Dynamic" },
];

const compareBenchmarkOptions = [
  { value: "all", label: "All benchmarks" },
  { value: "do_you_see_me", label: "Do-You-See-Me" },
  { value: "minds_eye", label: "Mind's-Eye" },
  { value: "spatial", label: "Spatial" },
];

const spatialMetricOptions = [
  { value: "accuracy", label: "Standard accuracy" },
  { value: "macro", label: "Macro accuracy" },
  { value: "dataset", label: "Selected dataset" },
  { value: "cot_delta", label: "CoT delta" },
  { value: "shortcut", label: "Shortcut score" },
  { value: "hallucination", label: "Hallucination resistance" },
];

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
const cotFilterOptions = [
  { value: "all", label: "All CoT modes" },
  { value: "yes", label: "Yes" },
  { value: "no", label: "No" },
  { value: "mixed", label: "Mixed / task-specific" },
  { value: "unspecified", label: "Not specified" },
];

function RankBadge({ rank }) {
  return (
    <span className={`rank-badge ${rank <= 3 ? `rank-${rank}` : ""}`}>
      {rank}
    </span>
  );
}

function DiagnosticsChip({ diagnostics }) {
  if (!diagnostics) return <span className="chip">Standard only</span>;
  const count = diagnostics.conditions_present?.length || 1;
  return <span className="chip layer-spatial">{count}/4 conditions</span>;
}

function asPercentPoint(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}`;
}

function mean(rows, resolveValue) {
  const values = rows
    .map(resolveValue)
    .filter((value) => value != null && Number.isFinite(value));
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

function groupSampleCount(groups) {
  return Object.values(groups || {}).reduce(
    (total, group) => total + (group.total_samples || 0),
    0,
  );
}

function totalVisualSamples(row) {
  return (
    groupSampleCount(row.perception_groups) +
    groupSampleCount(row.imagery_groups)
  );
}

function modelOrg(meta) {
  return meta?.organization || meta?.org || "—";
}

function modelParams(meta) {
  if (!meta) return "—";
  if (meta.parameter_count) return meta.parameter_count;
  if (meta.params) return meta.params;
  if (meta.params_b != null) return `${meta.params_b}B`;
  return "—";
}

function modelBase(meta) {
  return meta?.base_model || meta?.family || "—";
}

function modelCot(meta) {
  const value = meta?.cot_used;
  if (value == null || value === "") return "—";
  if (value === true) return "Yes";
  if (value === false) return "No";
  return prettyLabel(value);
}
function modelCotMode(meta) {
  const value = meta?.cot_used;
  if (value == null || value === "") return "unspecified";
  if (value === true) return "yes";
  if (value === false) return "no";
  const normalized = String(value).trim().toLowerCase();
  if (["yes", "true", "1"].includes(normalized)) return "yes";
  if (["no", "false", "0"].includes(normalized)) return "no";
  if (normalized.includes("mixed") || normalized.includes("task"))
    return "mixed";
  return normalized;
}
function cotFilterLabel(value) {
  return (
    cotFilterOptions.find((option) => option.value === value)?.label ||
    prettyLabel(value)
  );
}

function modelSearchText(row) {
  return [
    row.model_name,
    modelOrg(row.model_meta),
    modelType(row.model_meta),
    modelParams(row.model_meta),
    modelBase(row.model_meta),
    modelCot(row.model_meta),
  ]
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
        });
      }
    });
    Object.keys(row.imagery_groups || {}).forEach((key) => {
      const id = `imagery:${key}`;
      if (!seen.has(id)) {
        seen.add(id);
        capabilities.push({
          id,
          key,
          layer: "imagery",
          label: `I · ${prettyLabel(key)}`,
        });
      }
    });
  });
  return capabilities;
}

function getVisualCapability(row, capabilityId) {
  if (!capabilityId || capabilityId === "all") return null;
  const [layer, key] = capabilityId.split(":");
  const groups =
    layer === "perception" ? row.perception_groups : row.imagery_groups;
  return groups?.[key]?.accuracy ?? null;
}

function visualGap(row) {
  if (row.perception_accuracy == null || row.imagery_accuracy == null)
    return null;
  return row.perception_accuracy - row.imagery_accuracy;
}

function visualMetricValue(row, metric, capabilityId) {
  if (metric === "perception") return row.perception_accuracy;
  if (metric === "imagery") return row.imagery_accuracy;
  if (metric === "gap") {
    const gap = visualGap(row);
    return gap == null ? null : Math.abs(gap);
  }
  if (metric === "capability") return getVisualCapability(row, capabilityId);
  return row.vci;
}

function visualMetricDisplay(row, metric, capabilityId) {
  if (metric === "gap") {
    const gap = visualGap(row);
    return gap == null
      ? "—"
      : `${gap >= 0 ? "+" : ""}${(gap * 100).toFixed(1)} pts`;
  }
  const value = visualMetricValue(row, metric, capabilityId);
  return metric === "vci" ? fmtVci(value) : fmtPct(value);
}

function visualMeanDisplay(value, metric) {
  if (value == null) return "—";
  if (metric === "vci") return fmtVci(value);
  if (metric === "gap") return `${(value * 100).toFixed(1)} pts`;
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
  dataset,
  scopedDatasets = [],
  scopeActive = false,
) {
  if (metric === "macro") return row.macro_accuracy;
  if (metric === "dataset") return spatialDatasetValue(row, dataset);
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
    return value == null ? "—" : `${fmtDelta(value)} pts`;
  return fmtPct(value);
}

function spatialMeanDisplay(value, metric) {
  if (value == null) return "—";
  if (metric === "cot_delta") return `${fmtDelta(value)} pts`;
  return fmtPct(value);
}

function metricSortDirection(metric, explicitDirection) {
  if (explicitDirection !== "auto") return explicitDirection;
  return metric === "shortcut" ? "asc" : "desc";
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
  type: "Model access type — e.g. open weights, proprietary API, or research preview.",
  params: "Approximate parameter count (model size).",
  base: "Base model or backbone the system is built on.",
  cot: "Whether chain-of-thought prompting was used during evaluation.",
  rankMetric: "Score currently used to rank models in this view.",
  vci: "Visual Cognition Index — combined 0–100 score across perception and imagery.",
  perception:
    "Accuracy on Do-You-See-Me perception tasks (low-level visual understanding).",
  imagery: "Accuracy on Mind's-Eye mental-imagery and visualization tasks.",
  spatial: "Overall accuracy on spatial-reasoning benchmarks.",
  gap: "Perception minus imagery accuracy (positive = stronger perception).",
  samples: "Number of scored question instances.",
  scope: "Accuracy over the currently selected spatial benchmark scope.",
  macro: "Macro-averaged accuracy across datasets, shown with its standard deviation.",
  cot_delta: "Accuracy change from chain-of-thought vs. standard prompting.",
  shortcut:
    "Shortcut reliance — accuracy when the image is withheld (lower is better).",
  hallucination:
    "Hallucination resistance — robustness to absent or misleading visuals (higher is better).",
  coverage: "Share of datasets / conditions the model was evaluated on.",
};

function SortHeader({
  label,
  sortKey,
  sort,
  onSort,
  className = "",
  defaultDirection = "desc",
}) {
  const active = sort.key === sortKey;
  const description = COLUMN_DESCRIPTIONS[sortKey];
  return (
    <th
      className={`${className} sortable-th ${active ? "is-active" : ""}`.trim()}
      data-tip={description || undefined}
      aria-sort={
        active
          ? sort.direction === "asc"
            ? "ascending"
            : "descending"
          : "none"
      }
    >
      <button type="button" onClick={() => onSort(sortKey, defaultDirection)}>
        <span>{label}</span>
        <span aria-hidden="true" className="sort-mark">
          {active ? (sort.direction === "asc" ? "↑" : "↓") : "↕"}
        </span>
      </button>
    </th>
  );
}

function visualColumnValue(row, sortKey, filters, capabilities) {
  if (sortKey === "model") return row.model_name;
  if (sortKey === "org") return modelOrg(row.model_meta);
  if (sortKey === "type") return modelType(row.model_meta);
  if (sortKey === "params") return modelParams(row.model_meta);
  if (sortKey === "base") return modelBase(row.model_meta);
  if (sortKey === "cot") return modelCot(row.model_meta);
  if (sortKey === "rankMetric")
    return visualMetricValue(row, filters.metric, filters.capability);
  if (sortKey === "vci") return row.vci;
  if (sortKey === "perception") return row.perception_accuracy;
  if (sortKey === "imagery") return row.imagery_accuracy;
  if (sortKey === "gap")
    return visualGap(row) == null ? null : Math.abs(visualGap(row));
  if (sortKey === "samples") return totalVisualSamples(row);
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
  if (sortKey === "type") return modelType(row.model_meta);
  if (sortKey === "params") return modelParams(row.model_meta);
  if (sortKey === "base") return modelBase(row.model_meta);
  if (sortKey === "cot") return modelCot(row.model_meta);
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
  if (sortKey === "samples") return row.total_samples;
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
  if (sortKey === "type") return modelType(row.model_meta);
  if (sortKey === "params") return modelParams(row.model_meta);
  if (sortKey === "base") return modelBase(row.model_meta);
  if (sortKey === "cot") return modelCot(row.model_meta);
  if (sortKey === "vci") return row.visual?.vci;
  if (sortKey === "perception") return row.visual?.perception_accuracy;
  if (sortKey === "imagery") return row.visual?.imagery_accuracy;
  if (sortKey === "spatial") return row.spatial?.accuracy;
  if (sortKey === "cot_delta") return row.spatial?.diagnostics?.cot_delta;
  if (sortKey === "shortcut") return row.spatial?.diagnostics?.shortcut_score;
  if (sortKey === "hallucination")
    return row.spatial?.diagnostics?.hallucination_resistance;
  if (sortKey === "coverage")
    return (
      Number(Boolean(row.visual?.has_perception)) +
      Number(Boolean(row.visual?.has_imagery)) +
      Number(Boolean(row.spatial))
    );
  return row.visual?.vci ?? row.spatial?.accuracy;
}

function heatPct(value) {
  if (value == null || !Number.isFinite(value)) return "—";
  const ratio = value > 1 ? value / 100 : value;
  return `${Math.round(ratio * 100)}%`;
}

function scoreColor(value, invert = false) {
  if (value == null || !Number.isFinite(value)) return "transparent";
  const normalizedValue = value > 1 ? value / 100 : value;
  const score = Math.max(
    0,
    Math.min(1, invert ? 1 - normalizedValue : normalizedValue),
  );
  const hue = 358 + score * 170;
  const lightness = 20 + score * 34;
  return `hsl(${hue} 72% ${lightness}%)`;
}

function FilterField({ label, children }) {
  return (
    <label className="control-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function SegmentControl({ label, options, value, onChange }) {
  return (
    <div className="segment-control" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={value === option.value ? "is-active" : ""}
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
    <div className="control-deck">
      <div className="control-deck-head">
        <div>
          <div className="deck-eyebrow">{eyebrow}</div>
          <h3>{title}</h3>
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
    <div className="active-filter-row" aria-label="Active filters">
      {visibleChips.map((chip) => (
        <button
          key={chip.label}
          type="button"
          className="active-filter-chip"
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
      row.visual?.has_perception && "P",
      row.visual?.has_imagery && "I",
      row.spatial && "S",
    ]
      .filter(Boolean)
      .join(" + ") || "—"
  );
}

function CompareModelPicker({
  candidateRows,
  max,
  onAdd,
  onRemove,
  onReset,
  selectedNames,
  selectedRows,
}) {
  const selectedSet = new Set(selectedNames);
  return (
    <div className="compare-picker">
      <div className="compare-picker-head">
        <div>
          <span>Selected models</span>
          <strong>
            {selectedRows.length}/{max}
          </strong>
        </div>
        <button type="button" onClick={onReset}>
          Reset to top models
        </button>
      </div>
      <div className="selected-model-row">
        {selectedRows.length ? (
          selectedRows.map((row) => (
            <button
              type="button"
              className="selected-model-chip"
              key={row.model_name}
              onClick={() => onRemove(row.model_name)}
            >
              <strong>{row.model_name}</strong>
              <span>{compareCoverageText(row)}</span>
              <em aria-hidden="true">×</em>
            </button>
          ))
        ) : (
          <p className="muted small">
            Select at least one model to populate the comparison table.
          </p>
        )}
      </div>
      <div
        className="candidate-model-grid"
        aria-label="Available models to compare"
      >
        {candidateRows.length ? (
          candidateRows.slice(0, 10).map((row) => {
            const selected = selectedSet.has(row.model_name);
            const disabled = selected || selectedRows.length >= max;
            return (
              <button
                type="button"
                key={row.model_name}
                disabled={disabled}
                onClick={() => onAdd(row.model_name)}
              >
                <span>{selected ? "Selected" : "Add"}</span>
                <strong>{row.model_name}</strong>
                <em>{compareCoverageText(row)}</em>
              </button>
            );
          })
        ) : (
          <p className="muted small">
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
    <div className="stat-band cols-4 leaderboard-stats">
      {items.map(([value, label]) => (
        <div className="stat-cell" key={label}>
          <div className="v">{value}</div>
          <div className="l">{label}</div>
        </div>
      ))}
    </div>
  );
}

function Heatmap({ rows, columns, valueFor, displayFor, emptyText }) {
  if (!rows.length || !columns.length)
    return <div className="viz-empty">{emptyText}</div>;
  return (
    <div className="heatmap-wrap">
      <div
        className="heatmap-grid"
        style={{
          gridTemplateColumns: `minmax(160px, 1.4fr) repeat(${columns.length}, minmax(74px, 1fr))`,
        }}
      >
        <div className="heat-head sticky-col">Model</div>
        {columns.map((column) => (
          <div className="heat-head" key={column.id || column}>
            {column.label || prettyLabel(column)}
          </div>
        ))}
        {rows.map((row) => (
          <div className="heat-row-frag" key={row.model_name}>
            <div className="heat-label sticky-col">{row.model_name}</div>
            {columns.map((column) => {
              const value = valueFor(row, column);
              return (
                <div
                  className="heat-cell"
                  key={`${row.model_name}-${column.id || column}`}
                  style={{ background: scoreColor(value) }}
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

function VisualCapabilityChart({ rows, capabilities }) {
  return (
    <Heatmap
      rows={rows}
      columns={capabilities}
      valueFor={(row, column) => getVisualCapability(row, column.id)}
      displayFor={heatPct}
      emptyText="Capability profiles appear when ranked models are available."
    />
  );
}

function SpatialDatasetChart({ rows, datasets }) {
  return (
    <Heatmap
      rows={rows}
      columns={datasets.map((dataset) => ({ id: dataset, label: dataset }))}
      valueFor={(row, column) => spatialDatasetValue(row, column.id)}
      displayFor={heatPct}
      emptyText="Per-dataset accuracy appears when spatial rankings are available."
    />
  );
}

function PerceptionImageryChart({ rows }) {
  const completeRows = rows.filter(
    (row) => row.perception_accuracy != null && row.imagery_accuracy != null,
  );
  return (
    <BarChart
      aspectRatio="16 / 8"
      categories={completeRows.map((row) => ({ label: row.model_name, row }))}
      emptyMessage="Needs models with both perception and imagery scores."
      series={[
        { key: "perception", label: "Perception", color: chartPalette[1], valueFor: (category) => category.row.perception_accuracy },
        { key: "imagery", label: "Imagery", color: chartPalette[4], valueFor: (category) => category.row.imagery_accuracy },
      ]}
    />
  );
}

function CotComparisonChart({ rows }) {
  const data = rows.filter(
    (row) =>
      row.diagnostics?.standard_accuracy != null &&
      row.diagnostics?.cot_accuracy != null,
  );
  return (
    <BarChart
      aspectRatio="16 / 8"
      categories={data.map((row) => ({ label: row.model_name, row }))}
      emptyMessage="Needs standard and chain of thought submissions."
      series={[
        { key: "standard", label: "Standard", color: chartPalette[1], valueFor: (category) => category.row.diagnostics.standard_accuracy },
        { key: "cot", label: "Chain of thought", color: chartPalette[2], valueFor: (category) => category.row.diagnostics.cot_accuracy },
      ]}
    />
  );
}

function RobustnessChart({ rows }) {
  const data = rows.filter(
    (row) =>
      row.diagnostics?.hallucination_resistance != null &&
      row.diagnostics?.shortcut_score != null,
  );
  return (
    <BarChart
      aspectRatio="16 / 8"
      categories={data.map((row) => ({ label: row.model_name, row }))}
      emptyMessage="Needs no-image and no-image++ diagnostics."
      series={[
        { key: "accuracy", label: "Standard accuracy", color: chartPalette[1], valueFor: (category) => category.row.accuracy },
        { key: "hallucination", label: "Hallucination resistance", color: chartPalette[0], valueFor: (category) => category.row.diagnostics.hallucination_resistance },
        { key: "shortcut", label: "Shortcut score", color: chartPalette[2], valueFor: (category) => category.row.diagnostics.shortcut_score },
      ]}
    />
  );
}

function ComparisonChart({ rows }) {
  const metrics = [
    { label: "VCI", valueFor: (row) => row.visual?.vci },
    { label: "Perception", valueFor: (row) => row.visual?.perception_accuracy },
    { label: "Imagery", valueFor: (row) => row.visual?.imagery_accuracy },
    { label: "Spatial", valueFor: (row) => row.spatial?.accuracy },
    { label: "Hallucination", valueFor: (row) => row.spatial?.diagnostics?.hallucination_resistance },
  ];
  return (
    <BarChart
      aspectRatio="16 / 7"
      categories={metrics}
      emptyMessage="No models match the comparison filters."
      series={rows.map((row, index) => ({
        key: row.model_name,
        label: row.model_name,
        color: chartPalette[index % chartPalette.length],
        valueFor: (category) => category.valueFor(row),
      }))}
    />
  );
}

export function ResearchLeaderboard() {
  const [tab, setTab] = useState("vc");
  const [stats, setStats] = useState({});
  const [taskInfo, setTaskInfo] = useState({});
  const [visualRows, setVisualRows] = useState([]);
  const [spatialRows, setSpatialRows] = useState([]);
  const [reportModel, setReportModel] = useState(null);
  const [visualFilters, setVisualFilters] = useState({
    search: "",
    type: "all",
    benchmark: "all",
    cot: "all",
    metric: "vci",
    capability: "all",
    direction: "auto",
  });
  const [spatialFilters, setSpatialFilters] = useState({
    search: "",
    type: "all",
    metric: "accuracy",
    dataset: "all",
    datasetType: "all",
    diagnostics: "all",
    direction: "auto",
    cot: "all",
  });
  const [compareSearch, setCompareSearch] = useState("");
  const [visualSort, setVisualSort] = useState({
    key: "rankMetric",
    direction: "desc",
  });
  const [spatialSort, setSpatialSort] = useState({
    key: "rankMetric",
    direction: "desc",
  });
  const [compareSort, setCompareSort] = useState({
    key: "vci",
    direction: "desc",
  });
  const [compareBenchmark, setCompareBenchmark] = useState("all");
  const [selectedCompareModels, setSelectedCompareModels] = useState([]);

  useEffect(() => {
    Promise.all([
      getJSON("/api/statistics/overview").catch(() => ({})),
      ...["do_you_see_me", "minds_eye", "spatial"].map((id) =>
        getJSON(`/api/tasks/${id}/info`).catch(() => ({})),
      ),
    ]).then(([overview, dysm, minds, spatial]) => {
      setStats(overview);
      setTaskInfo({ do_you_see_me: dysm, minds_eye: minds, spatial });
    });
    getJSON("/api/leaderboard/visual-cognition")
      .then((data) => setVisualRows(data.leaderboard || []))
      .catch(() => setVisualRows([]));
    getJSON("/api/leaderboard/spatial")
      .then((data) => setSpatialRows(data.leaderboard || []))
      .catch(() => setSpatialRows([]));
  }, []);

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
  const spatialBenchmarkOptions = useMemo(
    () => [
      { value: "all", label: "All spatial benchmarks" },
      ...datasets.map((dataset) => ({ value: dataset, label: dataset })),
    ],
    [datasets],
  );
  const spatialScopedDatasets = useMemo(
    () => getSpatialScopedDatasets(datasets, spatialFilters),
    [datasets, spatialFilters],
  );
  const spatialScopeActive =
    spatialFilters.dataset !== "all" || spatialFilters.datasetType !== "all";
  const visualTypes = useMemo(
    () =>
      Array.from(
        new Set(visualRows.map((row) => modelType(row.model_meta))),
      ).sort(),
    [visualRows],
  );
  const spatialTypes = useMemo(
    () =>
      Array.from(
        new Set(spatialRows.map((row) => modelType(row.model_meta))),
      ).sort(),
    [spatialRows],
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
            ? row.has_perception
            : row.has_imagery),
      )
      .filter(
        (row) =>
          visualFilters.type === "all" ||
          modelType(row.model_meta) === visualFilters.type,
      )
      .filter(
        (row) =>
          visualFilters.cot === "all" ||
          modelCotMode(row.model_meta) === visualFilters.cot,
      )
      .sort((left, right) => {
        const valueCompare = compareSortValues(
          visualColumnValue(left, visualSort.key, visualFilters, capabilities),
          visualColumnValue(right, visualSort.key, visualFilters, capabilities),
          visualSort.direction,
        );
        if (valueCompare !== 0) return valueCompare;
        return compareMetricValues(left.vci, right.vci, "desc");
      });
  }, [visualRows, visualFilters, visualSort, capabilities]);

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
          spatialFilters.type === "all" ||
          modelType(row.model_meta) === spatialFilters.type,
      )
      .filter(
        (row) =>
          spatialFilters.cot === "all" ||
          modelCotMode(row.model_meta) === spatialFilters.cot,
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
            ? row.visual?.has_perception
            : compareBenchmark === "minds_eye"
              ? row.visual?.has_imagery
              : row.spatial),
      )
      .sort((left, right) =>
        compareMetricValues(
          left.visual?.vci ?? left.spatial?.accuracy ?? 0,
          right.visual?.vci ?? right.spatial?.accuracy ?? 0,
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
        .filter((name) => availableNames.has(name))
        .slice(0, MAX_COMPARE_MODELS);
      if (retained.length) return retained;
      return availableCompareRows
        .slice(0, Math.min(3, MAX_COMPARE_MODELS))
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
          left.visual?.vci ?? left.spatial?.accuracy ?? 0,
          right.visual?.vci ?? right.spatial?.accuracy ?? 0,
          "desc",
        );
      });
  }, [availableCompareRows, selectedCompareModels, compareSort]);

  const selectedCapability = capabilities.find(
    (capability) => capability.id === visualFilters.capability,
  );
  const visualMetricLabel =
    visualFilters.metric === "capability" && selectedCapability
      ? selectedCapability.label
      : visualMetricOptions.find(
          (option) => option.value === visualFilters.metric,
        )?.label || "Metric";
  const spatialMetricLabel =
    spatialFilters.metric === "dataset" && spatialFilters.dataset !== "all"
      ? `${spatialFilters.dataset} accuracy`
      : spatialScopeActive && spatialFilters.metric === "accuracy"
        ? `${spatialFilters.datasetType === "all" ? spatialFilters.dataset : spatialFilters.datasetType} accuracy`
        : spatialMetricOptions.find(
            (option) => option.value === spatialFilters.metric,
          )?.label || "Metric";
  const visualScopeLabel = visualBenchmarkLabel(visualFilters.benchmark);
  const spatialScopeLabel =
    spatialFilters.dataset !== "all"
      ? spatialFilters.dataset
      : spatialFilters.datasetType !== "all"
        ? `${spatialFilters.datasetType} spatial benchmarks`
        : "All spatial benchmarks";
  const compareScopeLabel = compareBenchmarkLabel(compareBenchmark);
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

  const visualStats = [
    [filteredVisualRows.length, "Models ranked"],
    [
      filteredVisualRows.filter((row) => row.complete).length,
      "Complete VCI profiles",
    ],
    [
      filteredVisualRows[0]
        ? visualMetricDisplay(
            filteredVisualRows[0],
            visualFilters.metric,
            visualFilters.capability,
          )
        : "—",
      `Top ${visualMetricLabel}`,
    ],
    [
      visualMeanDisplay(visualMeanMetric, visualFilters.metric),
      `Average ${visualMetricLabel}`,
    ],
  ];
  const topVisualSelection = filteredVisualRows
    .slice(0, 3)
    .map((row) => row.model_name);

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
        : "—",
      `Top ${spatialMetricLabel}`,
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
  const handleVisualSort = (key, defaultDirection = "desc") =>
    setVisualSort((current) => nextSortState(current, key, defaultDirection));
  const handleSpatialSort = (key, defaultDirection = "desc") =>
    setSpatialSort((current) => nextSortState(current, key, defaultDirection));
  const handleCompareSort = (key, defaultDirection = "desc") =>
    setCompareSort((current) => nextSortState(current, key, defaultDirection));

  const setVisualBenchmark = (benchmark) => {
    const metric =
      benchmark === "do_you_see_me"
        ? "perception"
        : benchmark === "minds_eye"
          ? "imagery"
          : visualFilters.metric;
    setVisualFilters((current) => ({
      ...current,
      benchmark,
      capability: "all",
      metric,
    }));
    setVisualSort({
      key: "rankMetric",
      direction: metricSortDirection(metric, visualFilters.direction),
    });
  };
  const setVisualMetric = (metric) => {
    setVisualFilters((current) => ({ ...current, metric }));
    setVisualSort({
      key: "rankMetric",
      direction: metricSortDirection(metric, visualFilters.direction),
    });
  };
  const setVisualCapability = (capability) => {
    const metric = capability === "all" ? visualFilters.metric : "capability";
    setVisualFilters((current) => ({ ...current, capability, metric }));
    setVisualSort({
      key: "rankMetric",
      direction: metricSortDirection(metric, visualFilters.direction),
    });
  };
  const setVisualDirection = (direction) => {
    setVisualFilters((current) => ({ ...current, direction }));
    setVisualSort({
      key: "rankMetric",
      direction: metricSortDirection(visualFilters.metric, direction),
    });
  };
  const setSpatialMetric = (metric) => {
    setSpatialFilters((current) => ({ ...current, metric }));
    setSpatialSort({
      key: "rankMetric",
      direction: metricSortDirection(metric, spatialFilters.direction),
    });
  };
  const setSpatialBenchmark = (dataset) => {
    const metric = dataset === "all" ? spatialFilters.metric : "dataset";
    setSpatialFilters((current) => ({
      ...current,
      dataset,
      datasetType: "all",
      metric,
    }));
    setSpatialSort({
      key: "rankMetric",
      direction: metricSortDirection(metric, spatialFilters.direction),
    });
  };
  const setSpatialType = (datasetType) => {
    setSpatialFilters((current) => ({
      ...current,
      datasetType,
      dataset: "all",
      metric: "accuracy",
    }));
    setSpatialSort({ key: "rankMetric", direction: "desc" });
  };
  const setSpatialDirection = (direction) => {
    setSpatialFilters((current) => ({ ...current, direction }));
    setSpatialSort({
      key: "rankMetric",
      direction: metricSortDirection(spatialFilters.metric, direction),
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
      current.includes(name) || current.length >= MAX_COMPARE_MODELS
        ? current
        : [...current, name],
    );
  const removeCompareModel = (name) =>
    setSelectedCompareModels((current) =>
      current.filter((item) => item !== name),
    );
  const resetCompareModels = () =>
    setSelectedCompareModels(
      availableCompareRows
        .slice(0, Math.min(3, MAX_COMPARE_MODELS))
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
    visualFilters.type !== "all" && {
      label: `Type: ${visualFilters.type}`,
      onClear: () =>
        setVisualFilters((current) => ({ ...current, type: "all" })),
    },
    visualFilters.cot !== "all" && {
      label: `CoT: ${cotFilterLabel(visualFilters.cot)}`,
      onClear: () =>
        setVisualFilters((current) => ({ ...current, cot: "all" })),
    },
    visualFilters.capability !== "all" &&
      selectedCapability && {
        label: selectedCapability.label,
        onClear: () => setVisualCapability("all"),
      },
    visualFilters.direction !== "auto" && {
      label: visualFilters.direction === "asc" ? "Low to high" : "High to low",
      onClear: () => setVisualDirection("auto"),
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
      onClear: () => {
        setSpatialBenchmark("all");
        setSpatialType("all");
      },
    },
    spatialFilters.type !== "all" && {
      label: `Type: ${spatialFilters.type}`,
      onClear: () =>
        setSpatialFilters((current) => ({ ...current, type: "all" })),
    },
    spatialFilters.cot !== "all" && {
      label: `CoT: ${cotFilterLabel(spatialFilters.cot)}`,
      onClear: () =>
        setSpatialFilters((current) => ({ ...current, cot: "all" })),
    },
    spatialFilters.diagnostics !== "all" && {
      label: "Diagnostics only",
      onClear: () =>
        setSpatialFilters((current) => ({ ...current, diagnostics: "all" })),
    },
    spatialFilters.direction !== "auto" && {
      label: spatialFilters.direction === "asc" ? "Low to high" : "High to low",
      onClear: () => setSpatialDirection("auto"),
    },
  ];
  const compareChips = [
    compareSearch && {
      label: `Search: ${compareSearch}`,
      onClear: () => setCompareSearch(""),
    },
    compareBenchmark !== "all" && {
      label: compareScopeLabel,
      onClear: () => setCompareBenchmarkScope("all"),
    },
  ];

  return (
    <>
      <PageHero
        eyebrow="Leaderboard"
        title="Model rankings"
        subtitle="Rank, filter, and compare models across visual cognition and spatial reasoning."
      />
      <section className="section leaderboard-section">
        <div className="container">
          <TabBar tabs={trackTabs} active={tab} onChange={setTab} />

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
                      {visualMetricOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </FilterField>
                  <FilterField label="Capability">
                    <select
                      value={visualFilters.capability}
                      onChange={(event) =>
                        setVisualCapability(event.target.value)
                      }
                    >
                      <option value="all">All capabilities</option>
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
                      placeholder="Model, org, source, base, or CoT"
                    />
                  </FilterField>
                </div>
                <details className="advanced-controls">
                  <summary>Advanced filters</summary>
                  <div className="advanced-grid">
                    <FilterField label="Model type">
                      <select
                        value={visualFilters.type}
                        onChange={(event) =>
                          setVisualFilters((current) => ({
                            ...current,
                            type: event.target.value,
                          }))
                        }
                      >
                        <option value="all">All types</option>
                        {visualTypes.map((type) => (
                          <option key={type} value={type}>
                            {type}
                          </option>
                        ))}
                      </select>
                    </FilterField>
                    <FilterField label="CoT">
                      <select
                        value={visualFilters.cot}
                        onChange={(event) =>
                          setVisualFilters((current) => ({
                            ...current,
                            cot: event.target.value,
                          }))
                        }
                      >
                        {cotFilterOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </FilterField>
                    <FilterField label="Direction">
                      <select
                        value={visualFilters.direction}
                        onChange={(event) =>
                          setVisualDirection(event.target.value)
                        }
                      >
                        <option value="auto">Metric default</option>
                        <option value="desc">High to low</option>
                        <option value="asc">Low to high</option>
                      </select>
                    </FilterField>
                  </div>
                </details>
              </ControlDeck>

              <div className="table-wrap leaderboard-table">
                <table className="lb-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <SortHeader
                        label="Model"
                        sortKey="model"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Org"
                        sortKey="org"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Source"
                        sortKey="type"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Params"
                        sortKey="params"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Base"
                        sortKey="base"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="CoT"
                        sortKey="cot"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label={visualMetricLabel}
                        sortKey="rankMetric"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        className="num"
                        defaultDirection={metricSortDirection(
                          visualFilters.metric,
                          "auto",
                        )}
                      />
                      <SortHeader
                        label="VCI"
                        sortKey="vci"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        className="num"
                      />
                      <SortHeader
                        label="Perception"
                        sortKey="perception"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        className="num"
                      />
                      <SortHeader
                        label="Imagery"
                        sortKey="imagery"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        className="num"
                      />
                      <SortHeader
                        label="P-I gap"
                        sortKey="gap"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        className="num"
                      />
                      <SortHeader
                        label="n"
                        sortKey="samples"
                        sort={visualSort}
                        onSort={handleVisualSort}
                        className="num"
                      />
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
                          <td>
                            <RankBadge rank={index + 1} />
                          </td>
                          <td>
                            <strong>{row.model_name}</strong>
                          </td>
                          <td>{modelOrg(row.model_meta)}</td>
                          <td>{modelType(row.model_meta)}</td>
                          <td>{modelParams(row.model_meta)}</td>
                          <td>{modelBase(row.model_meta)}</td>
                          <td>{modelCot(row.model_meta)}</td>
                          <td className="num vci-val">
                            {visualMetricDisplay(
                              row,
                              visualFilters.metric,
                              visualFilters.capability,
                            )}
                          </td>
                          <td className="num">{fmtVci(row.vci)}</td>
                          <td className="num">
                            {fmtPct(row.perception_accuracy)}
                          </td>
                          <td className="num">
                            {fmtPct(row.imagery_accuracy)}
                          </td>
                          <td
                            className={`num ${(visualGap(row) ?? 0) < 0 ? "neg" : "pos"}`}
                          >
                            {visualGap(row) == null
                              ? "—"
                              : `${visualGap(row) >= 0 ? "+" : ""}${(visualGap(row) * 100).toFixed(1)}`}
                          </td>
                          <td className="num">
                            {totalVisualSamples(row) || "—"}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="13" className="empty-row">
                          No models match these filters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="dashboard-grid two-col">
                <div className="viz-card wide">
                  <h3>Capability profile</h3>
                  <VisualCapabilityChart
                    rows={filteredVisualRows}
                    capabilities={visibleCapabilities}
                  />
                </div>
                <div className="viz-card">
                  <h3>Perception vs imagery</h3>
                  <PerceptionImageryChart rows={filteredVisualRows} />
                </div>
                <div className="viz-card">
                  <h3>Top-model capability trace</h3>
                  <CapabilityRadar
                    rows={filteredVisualRows}
                    selected={topVisualSelection}
                  />
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
                <SegmentControl
                  label="Spatial benchmark type"
                  options={spatialTypeOptions}
                  value={spatialFilters.datasetType}
                  onChange={setSpatialType}
                />
                <div className="primary-controls">
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
                  <FilterField label="Rank by">
                    <select
                      value={spatialFilters.metric}
                      onChange={(event) => setSpatialMetric(event.target.value)}
                    >
                      {spatialMetricOptions.map((option) => (
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
                      placeholder="Model, org, source, base, or CoT"
                    />
                  </FilterField>
                </div>
                <details className="advanced-controls">
                  <summary>Advanced filters</summary>
                  <div className="advanced-grid">
                    <FilterField label="Model type">
                      <select
                        value={spatialFilters.type}
                        onChange={(event) =>
                          setSpatialFilters((current) => ({
                            ...current,
                            type: event.target.value,
                          }))
                        }
                      >
                        <option value="all">All types</option>
                        {spatialTypes.map((type) => (
                          <option key={type} value={type}>
                            {type}
                          </option>
                        ))}
                      </select>
                    </FilterField>
                    <FilterField label="CoT">
                      <select
                        value={spatialFilters.cot}
                        onChange={(event) =>
                          setSpatialFilters((current) => ({
                            ...current,
                            cot: event.target.value,
                          }))
                        }
                      >
                        {cotFilterOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </FilterField>
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
                        <option value="all">All rows</option>
                        <option value="full">Diagnostics only</option>
                      </select>
                    </FilterField>
                    <FilterField label="Direction">
                      <select
                        value={spatialFilters.direction}
                        onChange={(event) =>
                          setSpatialDirection(event.target.value)
                        }
                      >
                        <option value="auto">Metric default</option>
                        <option value="desc">High to low</option>
                        <option value="asc">Low to high</option>
                      </select>
                    </FilterField>
                  </div>
                </details>
              </ControlDeck>

              <div className="table-wrap leaderboard-table">
                <table className="lb-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <SortHeader
                        label="Model"
                        sortKey="model"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Org"
                        sortKey="org"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Source"
                        sortKey="type"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Params"
                        sortKey="params"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Base"
                        sortKey="base"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label={spatialMetricLabel}
                        sortKey="rankMetric"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                        defaultDirection={metricSortDirection(
                          spatialFilters.metric,
                          "auto",
                        )}
                      />
                      <SortHeader
                        label="Scope accuracy"
                        sortKey="scope"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                      />
                      <SortHeader
                        label="Macro ± σ"
                        sortKey="macro"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                      />
                      <SortHeader
                        label="n"
                        sortKey="samples"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                      />
                      <SortHeader
                        label="CoT"
                        sortKey="cot"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="CoT Δ"
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
                        label="Halluc. ↑"
                        sortKey="hallucination"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                        className="num"
                      />
                      <SortHeader
                        label="Coverage"
                        sortKey="coverage"
                        sort={spatialSort}
                        onSort={handleSpatialSort}
                      />
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
                          <td>
                            <RankBadge rank={index + 1} />
                          </td>
                          <td>
                            <strong>{row.model_name}</strong>
                          </td>
                          <td>{modelOrg(row.model_meta)}</td>
                          <td>{modelType(row.model_meta)}</td>
                          <td>{modelParams(row.model_meta)}</td>
                          <td>{modelBase(row.model_meta)}</td>
                          <td className="num vci-val">
                            {spatialMetricDisplay(
                              row,
                              spatialFilters.metric,
                              spatialFilters.dataset,
                              spatialScopedDatasets,
                              spatialScopeActive,
                            )}
                          </td>
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
                            {row.macro_accuracy == null
                              ? "—"
                              : `${fmtPct(row.macro_accuracy)} ± ${fmtPct(row.accuracy_std)}`}
                          </td>
                          <td className="num">{row.total_samples || 0}</td>
                          <td>{modelCot(row.model_meta)}</td>
                          <td
                            className={`num ${(row.diagnostics?.cot_delta ?? 0) < 0 ? "neg" : "pos"}`}
                          >
                            {row.diagnostics?.cot_delta == null
                              ? "—"
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
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="15" className="empty-row">
                          No models match these filters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="dashboard-grid two-col">
                <div className="viz-card wide">
                  <h3>Per-dataset accuracy</h3>
                  <SpatialDatasetChart
                    rows={filteredSpatialRows}
                    datasets={spatialScopedDatasets}
                  />
                </div>
                <div className="viz-card">
                  <h3>CoT effect</h3>
                  <CotComparisonChart rows={filteredSpatialRows} />
                </div>
                <div className="viz-card">
                  <h3>Grounding robustness</h3>
                  <RobustnessChart rows={filteredSpatialRows} />
                </div>
              </div>
            </section>
          )}

          {tab === "compare" && (
            <section className="tab-panel is-active">
              <DashboardStats
                items={[
                  [compareRows.length, "Models selected"],
                  [
                    compareRows.filter(
                      (row) => row.visual?.complete && row.spatial,
                    ).length,
                    "Complete profiles",
                  ],
                  [availableCompareRows.length, "Models available"],
                  [stats.with_diagnostics ?? "—", "Diagnostic submissions"],
                ]}
              />
              <ControlDeck
                eyebrow="Comparison scope"
                title={compareScopeLabel}
                chips={compareChips}
              >
                <SegmentControl
                  label="Comparison benchmark scope"
                  options={compareBenchmarkOptions}
                  value={compareBenchmark}
                  onChange={setCompareBenchmarkScope}
                />
                <div className="primary-controls narrow">
                  <FilterField label="Find model to add">
                    <input
                      value={compareSearch}
                      onChange={(event) => setCompareSearch(event.target.value)}
                      placeholder="Model, org, source, base, or CoT"
                    />
                  </FilterField>
                </div>
                <CompareModelPicker
                  candidateRows={compareCandidateRows}
                  max={MAX_COMPARE_MODELS}
                  onAdd={addCompareModel}
                  onRemove={removeCompareModel}
                  onReset={resetCompareModels}
                  selectedNames={selectedCompareModels}
                  selectedRows={compareRows}
                />
              </ControlDeck>
              <div className="table-wrap leaderboard-table">
                <table className="lb-table">
                  <thead>
                    <tr>
                      <SortHeader
                        label="Model"
                        sortKey="model"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Org"
                        sortKey="org"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Source"
                        sortKey="type"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Params"
                        sortKey="params"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Base"
                        sortKey="base"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="CoT"
                        sortKey="cot"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="VCI"
                        sortKey="vci"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                      />
                      <SortHeader
                        label="Perception"
                        sortKey="perception"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                      />
                      <SortHeader
                        label="Imagery"
                        sortKey="imagery"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                      />
                      <SortHeader
                        label="Spatial"
                        sortKey="spatial"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                      />
                      <SortHeader
                        label="CoT Δ"
                        sortKey="cot_delta"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                      />
                      <SortHeader
                        label="Shortcut ↓"
                        sortKey="shortcut"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                        defaultDirection="asc"
                      />
                      <SortHeader
                        label="Halluc. ↑"
                        sortKey="hallucination"
                        sort={compareSort}
                        onSort={handleCompareSort}
                        className="num"
                      />
                      <SortHeader
                        label="Coverage"
                        sortKey="coverage"
                        sort={compareSort}
                        onSort={handleCompareSort}
                      />
                    </tr>
                  </thead>
                  <tbody>
                    {compareRows.length ? (
                      compareRows.map((row) => (
                        <tr key={row.model_name}>
                          <td>
                            <strong>{row.model_name}</strong>
                          </td>
                          <td>{modelOrg(row.model_meta)}</td>
                          <td>{modelType(row.model_meta)}</td>
                          <td>{modelParams(row.model_meta)}</td>
                          <td>{modelBase(row.model_meta)}</td>
                          <td>{modelCot(row.model_meta)}</td>
                          <td className="num vci-val">
                            {fmtVci(row.visual?.vci)}
                          </td>
                          <td className="num">
                            {fmtPct(row.visual?.perception_accuracy)}
                          </td>
                          <td className="num">
                            {fmtPct(row.visual?.imagery_accuracy)}
                          </td>
                          <td className="num">
                            {fmtPct(row.spatial?.accuracy)}
                          </td>
                          <td
                            className={`num ${(row.spatial?.diagnostics?.cot_delta ?? 0) < 0 ? "neg" : "pos"}`}
                          >
                            {row.spatial?.diagnostics?.cot_delta == null
                              ? "—"
                              : fmtDelta(row.spatial.diagnostics.cot_delta)}
                          </td>
                          <td className="num">
                            {fmtPct(row.spatial?.diagnostics?.shortcut_score)}
                          </td>
                          <td className="num">
                            {fmtPct(
                              row.spatial?.diagnostics
                                ?.hallucination_resistance,
                            )}
                          </td>
                          <td>
                            <span className="chip">
                              {[
                                row.visual?.has_perception && "P",
                                row.visual?.has_imagery && "I",
                                row.spatial && "S",
                              ]
                                .filter(Boolean)
                                .join(" + ") || "—"}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="14" className="empty-row">
                          No models match these filters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="dashboard-grid">
                <div className="viz-card wide">
                  <h3>Cross-track profile matrix</h3>
                  <ComparisonChart rows={compareRows.slice(0, 14)} />
                </div>
              </div>
            </section>
          )}
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
  if (!keys.length) return <p className="muted small">—</p>;
  return (
    <table className="lb-table small">
      <thead>
        <tr>
          <th>Group</th>
          <th className="num">Acc.</th>
          <th className="num">n</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((key) => (
          <tr key={key}>
            <td>{prettyLabel(key)}</td>
            <td className="num">{fmtPct(groups[key].accuracy)}</td>
            <td className="num">
              {groups[key].correct_samples}/{groups[key].total_samples}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReportMeta({ meta }) {
  if (!meta) return null;
  const safeUrl = (value) => (/^https?:\/\//i.test(String(value || "").trim()) ? String(value).trim() : null);
  const rows = [
    ["Organization", modelOrg(meta)],
    ["Access", modelType(meta)],
    ["Parameters", modelParams(meta)],
    ["Base model", modelBase(meta)],
    ["CoT used", modelCot(meta)],
  ].filter(([, value]) => value && value !== "\u2014");
  const paperUrl = safeUrl(meta.paper_url);
  const blocks = [
    ["Method", meta.method_description],
    ["Training data", meta.training_data],
    ["Prompt template", meta.prompt_template],
    ["Changes from previous submission", meta.changes_from_previous],
  ].filter(([, value]) => value && String(value).trim());
  if (!rows.length && !paperUrl && !blocks.length) return null;
  return (
    <div className="report-meta">
      <h3>Submission details</h3>
      {(rows.length > 0 || paperUrl) && (
        <table className="lb-table report-meta-table">
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
                  <a href={paperUrl} target="_blank" rel="noreferrer noopener">
                    {paperUrl}
                  </a>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}
      {blocks.map(([key, value]) => (
        <div className="report-meta-block" key={key}>
          <h4>{key}</h4>
          <p>{value}</p>
        </div>
      ))}
    </div>
  );
}

function ReportModalShim({ model, onClose }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!model) return;
    setReport(null);
    setError("");
    getJSON(`/api/model/${encodeURIComponent(model)}/report`)
      .then(setReport)
      .catch((err) => setError(err.message));
  }, [model]);
  if (!model) return null;
  const visual = report?.visual_cognition || {};
  const tasks = report?.tasks || {};
  return (
    <div className="modal">
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal-card">
        <button
          className="icon-btn modal-close"
          type="button"
          onClick={onClose}
          aria-label="Close"
        >
          ×
        </button>
        {!report && !error && <p className="muted">Loading…</p>}
        {error && (
          <p className="form-msg err">Failed to load report: {error}</p>
        )}
        {report && (
          <>
            <h2>{report.model_name}</h2>
            <div className="kpi-row">
              <div className="kpi">
                <div className="kpi-label">VCI</div>
                <div className="kpi-val">{fmtVci(visual.vci)}</div>
              </div>
              <div className="kpi">
                <div className="kpi-label">Perception</div>
                <div className="kpi-val">
                  {fmtPct(visual.perception_accuracy)}
                </div>
              </div>
              <div className="kpi">
                <div className="kpi-label">Imagery</div>
                <div className="kpi-val">{fmtPct(visual.imagery_accuracy)}</div>
              </div>
              <div className="kpi">
                <div className="kpi-label">Spatial</div>
                <div className="kpi-val">
                  {tasks.spatial ? fmtPct(tasks.spatial.accuracy) : "—"}
                </div>
              </div>
            </div>
            <ReportMeta meta={report.model_meta} />
            {tasks.do_you_see_me && (
              <>
                <h3>Do-You-See-Me capabilities</h3>
                <GroupsTable groups={tasks.do_you_see_me.groups} />
              </>
            )}
            {tasks.minds_eye && (
              <>
                <h3>Mind's-Eye capabilities</h3>
                <GroupsTable groups={tasks.minds_eye.groups} />
              </>
            )}
            {tasks.spatial && (
              <>
                <h3>Spatial datasets</h3>
                <GroupsTable groups={tasks.spatial.groups} />
                {tasks.spatial.diagnostics && (
                  <div className="kpi-row">
                    <div className="kpi">
                      <div className="kpi-label">CoT Δ</div>
                      <div
                        className={`kpi-val ${tasks.spatial.diagnostics.cot_delta < 0 ? "neg" : "pos"}`}
                      >
                        {fmtDelta(tasks.spatial.diagnostics.cot_delta)} pts
                      </div>
                    </div>
                    <div className="kpi">
                      <div className="kpi-label">Shortcut ↓</div>
                      <div className="kpi-val">
                        {fmtPct(tasks.spatial.diagnostics.shortcut_score)}
                      </div>
                    </div>
                    <div className="kpi">
                      <div className="kpi-label">Halluc. ↑</div>
                      <div className="kpi-val">
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
