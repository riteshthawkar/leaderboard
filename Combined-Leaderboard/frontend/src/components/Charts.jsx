import { useCallback, useMemo, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear } from "@visx/scale";
import { prettyLabel } from "@/lib/utils";

export const chartPalette = [
  "#14b8a6",
  "#6366f1",
  "#f97316",
  "#e11d48",
  "#a855f7",
  "#0ea5e9",
  "#84cc16",
  "#eab308",
];

function niceMax(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  const max = finite.length ? Math.max(...finite) : 0;
  if (max <= 0) return 1;
  if (max <= 1) return Math.min(1, Math.max(0.2, Math.ceil((max + 1e-6) / 0.1) * 0.1));
  const pow = Math.pow(10, Math.floor(Math.log10(max)));
  return Math.ceil(max / pow) * pow;
}

function topRoundedPath(x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height));
  return `M${x},${y + height} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + width - r},${y} Q${x + width},${y} ${x + width},${y + r} L${x + width},${y + height} Z`;
}

function bottomRoundedPath(x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height));
  return `M${x},${y} L${x + width},${y} L${x + width},${y + height - r} Q${x + width},${y + height} ${x + width - r},${y + height} L${x + r},${y + height} Q${x},${y + height} ${x},${y + height - r} Z`;
}

function formatPct(value, { scale = 100, suffix = "%", digits = 1 } = {}) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * scale).toFixed(digits)}${suffix}`;
}

function ChartTip({ tip }) {
  if (!tip) return null;
  return (
    <div className="viz-tooltip" style={{ left: tip.left, top: tip.top }}>
      <div className="viz-tooltip-title">{tip.title}</div>
      {tip.rows.map((row) => (
        <div className="viz-tooltip-row" key={row.label}>
          <span className="viz-tooltip-key">
            <span className="viz-tooltip-dot" style={{ background: row.color }} />
            {row.label}
          </span>
          <span className="viz-tooltip-val">{row.value}</span>
        </div>
      ))}
    </div>
  );
}

function ChartLegend({ series }) {
  return (
    <div className="radar-legend">
      {series.map((entry) => (
        <span className="legend-item" key={entry.key}>
          <span className="legend-dot" style={{ background: entry.color }} />
          {entry.label}
        </span>
      ))}
    </div>
  );
}

const EMPTY_BARS = [0.42, 0.6, 0.34, 0.7, 0.5, 0.58, 0.46];

export function EmptyChart({ message, aspectRatio = "16 / 9" }) {
  return (
    <div className="bklit-chart-wrap is-empty">
      <div className="viz-chart" style={{ aspectRatio }}>
        <ParentSize debounceTime={10}>
          {({ width, height }) => {
            if (width < 10 || height < 10) return null;
            const left = 40;
            const right = 16;
            const top = 16;
            const bottom = 26;
            const innerWidth = width - left - right;
            const innerHeight = height - top - bottom;
            const xScale = scaleBand({ domain: EMPTY_BARS.map((_, i) => String(i)), range: [0, innerWidth], padding: 0.36 });
            const yScale = scaleLinear({ domain: [0, 1], range: [innerHeight, 0] });
            return (
              <svg height={height} width={width}>
                <g transform={`translate(${left},${top})`}>
                  {EMPTY_BARS.map((value, index) => {
                    const barY = yScale(value);
                    return (
                      <path
                        className="viz-bar-empty"
                        d={topRoundedPath(xScale(String(index)), barY, xScale.bandwidth(), innerHeight - barY, 4)}
                        key={index}
                      />
                    );
                  })}
                  <line className="viz-axis-line" x1={0} x2={innerWidth} y1={innerHeight} y2={innerHeight} />
                </g>
              </svg>
            );
          }}
        </ParentSize>
        <div className="chart-empty-overlay">{message}</div>
      </div>
    </div>
  );
}

/**
 * Interactive categorical bar chart (single bars or grouped) built on @visx.
 *
 * @param categories `[{ label, ...meta }]` — one group per x-axis slot.
 * @param series     `[{ key, label, color, valueFor }]` where
 *                   `valueFor(category, index)` returns a 0..1 ratio (or null).
 */
export function BarChart({
  categories,
  series,
  aspectRatio = "16 / 9",
  valueScale = 100,
  valueSuffix = "%",
  valueDigits = 0,
  emptyMessage = "No data available yet.",
  showLegend = true,
}) {
  const [tip, setTip] = useState(null);
  const clearTip = useCallback(() => setTip(null), []);
  const ready = Boolean(categories?.length && series?.length);

  const maxValue = useMemo(() => {
    if (!ready) return 1;
    const values = [];
    categories.forEach((category, index) =>
      series.forEach((entry) => values.push(entry.valueFor(category, index))),
    );
    return niceMax(values);
  }, [categories, series, ready]);

  const hasValue = useMemo(() => {
    if (!ready) return false;
    return categories.some((category, index) =>
      series.some((entry) => {
        const value = entry.valueFor(category, index);
        return value != null && Number.isFinite(value);
      }),
    );
  }, [categories, series, ready]);

  const minValue = useMemo(() => {
    if (!ready) return 0;
    const values = [];
    categories.forEach((category, index) => series.forEach((entry) => values.push(entry.valueFor(category, index))));
    const finite = values.filter((value) => Number.isFinite(value));
    const min = finite.length ? Math.min(...finite) : 0;
    if (min >= 0) return 0;
    return -niceMax(finite.filter((value) => value < 0).map((value) => Math.abs(value)));
  }, [categories, series, ready]);

  if (!ready || !hasValue) return <EmptyChart aspectRatio={aspectRatio} message={emptyMessage} />;

  const grouped = series.length > 1;

  return (
    <div className="bklit-chart-wrap">
      <div className="viz-chart" onMouseLeave={clearTip} style={{ aspectRatio }}>
        <ParentSize debounceTime={10}>
          {({ width, height }) => (
            <BarChartSvg
              categories={categories}
              grouped={grouped}
              height={height}
              maxValue={maxValue}
              minValue={minValue}
              series={series}
              setTip={setTip}
              valueScale={valueScale}
              valueSuffix={valueSuffix}
              valueDigits={valueDigits}
              width={width}
            />
          )}
        </ParentSize>
        <ChartTip tip={tip} />
      </div>
      {showLegend && grouped && <ChartLegend series={series} />}
    </div>
  );
}

function BarChartSvg({ categories, grouped, height, maxValue, minValue = 0, series, setTip, valueScale, valueSuffix, valueDigits = 0, width }) {
  if (width < 10 || height < 10) return null;
  const labels = categories.map((category) => String(category.label ?? ""));
  const left = 46;
  const right = 16;
  const top = 18;
  const innerWidthBase = width - left - right;
  const bandPer = innerWidthBase / Math.max(1, categories.length);
  const longest = labels.reduce((max, label) => Math.max(max, label.length), 0);
  const estLabelPx = longest * 6.6;
  const rotate = estLabelPx > bandPer - 10;
  const bottom = rotate ? Math.min(104, 30 + Math.round(Math.sin(Math.PI / 5) * estLabelPx)) : 34;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  if (innerWidth < 20 || innerHeight < 20) return null;

  const xScale = scaleBand({ domain: labels.map((_, i) => String(i)), range: [0, innerWidth], padding: grouped ? 0.26 : 0.34 });
  const groupScale = grouped
    ? scaleBand({ domain: series.map((entry) => entry.key), range: [0, xScale.bandwidth()], padding: 0.16 })
    : null;
  const yScale = scaleLinear({ domain: [Math.min(0, minValue), maxValue], range: [innerHeight, 0] });
  const zeroY = yScale(0);
  const yTicks = yScale.ticks(minValue < 0 ? 6 : 4);
  const totalBars = categories.length * series.length;
  const showValueLabels = totalBars <= 9;
  const tickFmt = (value) => `${+(value * valueScale).toFixed(1)}${valueSuffix}`;
  const tooltipRows = (category, index) =>
    series.map((entry) => ({
      color: entry.color,
      label: entry.label,
      value: formatPct(entry.valueFor(category, index), { scale: valueScale, suffix: valueSuffix }),
    }));

  return (
    <svg height={height} role="img" width={width}>
      <g transform={`translate(${left},${top})`}>
        {yTicks.map((tick) => (
          <g key={tick} transform={`translate(0,${yScale(tick)})`}>
            <line className="viz-grid-line" x1={0} x2={innerWidth} />
            <text className="viz-axis-text" dy="0.32em" textAnchor="end" x={-10} y={0}>
              {tickFmt(tick)}
            </text>
          </g>
        ))}
        <line className="viz-axis-line" x1={0} x2={innerWidth} y1={zeroY} y2={zeroY} />
        {categories.map((category, index) => {
          const groupX = xScale(String(index)) ?? 0;
          return (
            <g key={index} transform={`translate(${groupX},0)`}>
              {series.map((entry) => {
                const value = entry.valueFor(category, index);
                const has = value != null && Number.isFinite(value);
                if (!has) return null;
                const clamped = Math.max(Math.min(0, minValue), Math.min(value, maxValue));
                const barWidth = grouped ? groupScale.bandwidth() : xScale.bandwidth();
                const barX = grouped ? groupScale(entry.key) ?? 0 : 0;
                const negative = clamped < 0;
                const barTop = yScale(Math.max(0, clamped));
                const barBottom = yScale(Math.min(0, clamped));
                const barHeight = barBottom - barTop;
                const fill = entry.colorFor ? entry.colorFor(category, index) : entry.color;
                const centerX = left + groupX + barX + barWidth / 2;
                const rows = tooltipRows(category, index);
                const onHover = () => setTip({ left: centerX, top: top + barTop - 10, title: category.label, rows });
                return (
                  <g key={entry.key}>
                    {barHeight > 0 && (
                      <path
                        className="viz-bar"
                        d={negative
                          ? bottomRoundedPath(barX, barTop, barWidth, barHeight, Math.min(5, barWidth / 2))
                          : topRoundedPath(barX, barTop, barWidth, barHeight, Math.min(5, barWidth / 2))}
                        fill={fill}
                        onMouseEnter={onHover}
                        onMouseMove={onHover}
                      />
                    )}
                    {showValueLabels && (
                      <text className="viz-bar-label" textAnchor="middle" x={barX + barWidth / 2} y={negative ? barBottom + 13 : barTop - 6}>
                        {formatPct(value, { scale: valueScale, suffix: valueSuffix, digits: valueDigits })}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
        {categories.map((category, index) => {
          const centerX = (xScale(String(index)) ?? 0) + xScale.bandwidth() / 2;
          if (rotate) {
            return (
              <text
                className="viz-axis-text"
                key={index}
                textAnchor="end"
                transform={`translate(${centerX},${innerHeight + 12}) rotate(-32)`}
              >
                {category.label}
              </text>
            );
          }
          return (
            <text className="viz-axis-text" key={index} textAnchor="middle" x={centerX} y={innerHeight + 18}>
              {category.label}
            </text>
          );
        })}
      </g>
    </svg>
  );
}

/**
 * Interactive radar / spider chart for comparing models across several axes.
 *
 * @param categories `[{ label, ...meta }]` — one spoke per entry (>= 3).
 * @param series     `[{ key, label, color, valueFor }]` — one polygon per series.
 */
export function RadarChart({
  categories,
  series,
  aspectRatio = "1 / 1",
  valueScale = 100,
  valueSuffix = "%",
  levels = 4,
  emptyMessage = "No data available yet.",
  showLegend = true,
}) {
  const [tip, setTip] = useState(null);
  const clearTip = useCallback(() => setTip(null), []);
  const ready = Boolean(categories?.length >= 3 && series?.length);

  const maxValue = useMemo(() => {
    if (!ready) return 1;
    const values = [];
    categories.forEach((category, index) =>
      series.forEach((entry) => values.push(entry.valueFor(category, index))),
    );
    return niceMax(values);
  }, [categories, series, ready]);

  if (!ready) return <EmptyChart aspectRatio={aspectRatio} message={emptyMessage} />;

  return (
    <div className="bklit-chart-wrap">
      <div className="viz-chart" onMouseLeave={clearTip} style={{ aspectRatio }}>
        <ParentSize debounceTime={10}>
          {({ width, height }) => (
            <RadarSvg
              categories={categories}
              height={height}
              levels={levels}
              maxValue={maxValue}
              series={series}
              setTip={setTip}
              valueScale={valueScale}
              valueSuffix={valueSuffix}
              width={width}
            />
          )}
        </ParentSize>
        <ChartTip tip={tip} />
      </div>
      {showLegend && <ChartLegend series={series} />}
    </div>
  );
}

function RadarSvg({ categories, height, levels, maxValue, series, setTip, valueScale, valueSuffix, width }) {
  if (width < 10 || height < 10) return null;
  const count = categories.length;
  const padding = 72;
  const radius = Math.max(24, Math.min(width, height) / 2 - padding);
  const cx = width / 2;
  const cy = height / 2;
  const angleFor = (index) => -Math.PI / 2 + (index * 2 * Math.PI) / count;
  const pointAt = (index, r) => [cx + Math.cos(angleFor(index)) * r, cy + Math.sin(angleFor(index)) * r];
  const rings = Array.from({ length: levels }, (_, level) => (level + 1) / levels);
  const tooltipRows = (index) =>
    series.map((entry) => ({
      color: entry.color,
      label: entry.label,
      value: formatPct(entry.valueFor(categories[index], index), { scale: valueScale, suffix: valueSuffix }),
    }));

  return (
    <svg height={height} role="img" width={width}>
      {rings.map((ring, level) => (
        <polygon
          className="viz-radar-ring"
          key={level}
          points={categories.map((_, index) => pointAt(index, radius * ring).join(",")).join(" ")}
        />
      ))}
      {categories.map((category, index) => {
        const [ex, ey] = pointAt(index, radius);
        const [lx, ly] = pointAt(index, radius + 14);
        const cos = Math.cos(angleFor(index));
        const anchor = Math.abs(cos) < 0.3 ? "middle" : cos > 0 ? "start" : "end";
        return (
          <g key={index}>
            <line className="viz-radar-spoke" x1={cx} x2={ex} y1={cy} y2={ey} />
            <RadarLabel anchor={anchor} label={category.label} x={lx} y={ly} />
          </g>
        );
      })}
      {series.map((entry) => {
        const points = categories.map((category, index) => {
          const value = entry.valueFor(category, index);
          const ratio = value != null && Number.isFinite(value) ? Math.max(0, Math.min(1, value / maxValue)) : 0;
          return pointAt(index, radius * ratio);
        });
        return (
          <g key={entry.key}>
            <polygon
              className="viz-radar-area"
              fill={entry.color}
              fillOpacity={0.16}
              points={points.map((point) => point.join(",")).join(" ")}
              stroke={entry.color}
              strokeWidth={2}
            />
            {points.map((point, index) => {
              const rows = tooltipRows(index);
              const onHover = () => setTip({ left: point[0], top: point[1] - 10, title: categories[index].label, rows });
              return (
                <circle
                  className="viz-radar-dot"
                  cx={point[0]}
                  cy={point[1]}
                  fill={entry.color}
                  key={index}
                  onMouseEnter={onHover}
                  onMouseMove={onHover}
                  r={3.5}
                />
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function RadarLabel({ anchor, label, x, y }) {
  const text = String(label ?? "");
  const words = text.split(" ");
  let lines = [text];
  if (text.length > 12 && words.length > 1) {
    const mid = Math.ceil(words.length / 2);
    lines = [words.slice(0, mid).join(" "), words.slice(mid).join(" ")];
  }
  return (
    <text className="viz-axis-text" dominantBaseline="middle" textAnchor={anchor} x={x} y={y}>
      {lines.map((line, index) => (
        <tspan dy={index === 0 ? (lines.length > 1 ? "-0.3em" : "0.32em") : "1.05em"} key={index} x={x}>
          {line}
        </tspan>
      ))}
    </text>
  );
}

/**
 * Per-benchmark model performance: one bar per model showing its single
 * combined score on the benchmark.
 */
export function BenchmarkModelChart({ rows, metricFor, metricLabel, color = chartPalette[1], emptyMessage }) {
  const models = useMemo(
    () =>
      (rows || [])
        .map((row) => ({ name: row.model_name, value: metricFor(row) }))
        .filter((item) => item.value != null && Number.isFinite(item.value)),
    [rows, metricFor],
  );

  if (!models.length) {
    return (
      <EmptyChart
        aspectRatio="16 / 7"
        message={emptyMessage || "Model scores will appear once submissions are ranked."}
      />
    );
  }

  return (
    <BarChart
      aspectRatio="16 / 7"
      categories={models.map((item) => ({ label: item.name }))}
      series={[{ key: "score", label: metricLabel, color, valueFor: (_category, index) => models[index].value }]}
      showLegend={false}
    />
  );
}

export function CapabilityRadar({ rows, selected }) {
  const { categories, series } = useMemo(() => {
    const capabilityOf = (row) => {
      const caps = {};
      ["perception_groups", "imagery_groups"].forEach((key) => {
        Object.entries(row[key] || {}).forEach(([name, group]) => {
          caps[name] = group.accuracy;
        });
      });
      return caps;
    };
    const axes = new Set();
    rows.forEach((row) => Object.keys(capabilityOf(row)).forEach((key) => axes.add(key)));
    const orderedAxes = Array.from(axes).sort();
    const visible = rows.filter((row) => selected.includes(row.model_name));
    const capsByModel = new Map(visible.map((row) => [row.model_name, capabilityOf(row)]));
    return {
      categories: orderedAxes.map((axis) => ({ label: prettyLabel(axis), axis })),
      series: visible.map((row, index) => ({
        key: row.model_name,
        label: row.model_name,
        color: chartPalette[index % chartPalette.length],
        valueFor: (category) => capsByModel.get(row.model_name)?.[category.axis] ?? 0,
      })),
    };
  }, [rows, selected]);

  if (categories.length < 3 || !series.length) {
    return <EmptyChart aspectRatio="5 / 4" message="Ranked models will populate capability profiles." />;
  }

  return <RadarChart aspectRatio="5 / 4" categories={categories} series={series} />;
}
