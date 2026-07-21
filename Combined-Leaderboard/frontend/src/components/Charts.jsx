import { useCallback, useId, useMemo, useState } from "react";
import { ParentSize } from "@visx/responsive";
import { scaleBand, scaleLinear } from "@visx/scale";
import { prettyLabel } from "@/lib/utils";

export const chartPalette = [
  "var(--dysm)",
  "var(--me)",
  "var(--spatial)",
  "var(--chart-negative)",
  "var(--chart-positive)",
  "var(--chart-neutral)",
  "var(--chart-accent)",
  "var(--text-muted)",
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

function BarGradient({ color, id }) {
  return (
    <linearGradient id={id} x1="0%" x2="0%" y1="0%" y2="100%">
      <stop offset="0%" stopColor={color} stopOpacity={1} />
      <stop offset="58%" stopColor={color} stopOpacity={0.68} />
      <stop offset="100%" stopColor={color} stopOpacity={0.16} />
    </linearGradient>
  );
}

function formatPct(value, { scale = 100, suffix = "%", digits = 1 } = {}) {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return `${(value * scale).toFixed(digits)}${suffix}`;
}

export function layoutScatterPoints(points, { xScale, yScale, innerWidth, innerHeight }) {
  const positioned = points
    .map((point, index) => ({
      ...point,
      index,
      rawX: xScale(point.x),
      rawY: yScale(point.y),
    }))
    .filter((point) => Number.isFinite(point.rawX) && Number.isFinite(point.rawY));
  const clusters = [];
  positioned.forEach((point) => {
    const cluster = clusters.find((candidate) => {
      const anchor = candidate[0];
      return Math.hypot(point.rawX - anchor.rawX, point.rawY - anchor.rawY) < 9;
    });
    if (cluster) cluster.push(point);
    else clusters.push([point]);
  });
  clusters.forEach((cluster) => {
    cluster.forEach((point, index) => {
      const angle = cluster.length > 1 ? -Math.PI / 2 + (index * 2 * Math.PI) / cluster.length : 0;
      const spread = cluster.length > 1 ? Math.min(7, 3 + cluster.length) : 0;
      point.screenX = Math.max(0, Math.min(innerWidth, point.rawX + Math.cos(angle) * spread));
      point.screenY = Math.max(0, Math.min(innerHeight, point.rawY + Math.sin(angle) * spread));
    });
  });

  const placedBoxes = [];
  const offsets = [0, -16, 16, -32, 32, -48, 48, -64, 64];
  positioned
    .slice()
    .sort((left, right) => left.screenY - right.screenY || left.screenX - right.screenX)
    .forEach((point) => {
      point.alignRight = point.screenX > innerWidth * 0.76;
      point.labelX = point.screenX + (point.alignRight ? -9 : 9);
      const estimatedWidth = Math.max(42, String(point.label || "").length * 6.4);
      const boxFor = (labelY) => ({
        left: point.alignRight ? point.labelX - estimatedWidth : point.labelX,
        right: point.alignRight ? point.labelX : point.labelX + estimatedWidth,
        top: labelY - 7,
        bottom: labelY + 7,
      });
      const overlaps = (box) => placedBoxes.some((placed) => !(
        box.right + 4 < placed.left
        || box.left - 4 > placed.right
        || box.bottom + 3 < placed.top
        || box.top - 3 > placed.bottom
      ));
      let labelY = Math.max(8, Math.min(innerHeight - 8, point.screenY));
      for (const offset of offsets) {
        const candidateY = Math.max(8, Math.min(innerHeight - 8, point.screenY + offset));
        if (!overlaps(boxFor(candidateY))) {
          labelY = candidateY;
          break;
        }
      }
      point.labelY = labelY;
      placedBoxes.push(boxFor(labelY));
    });

  return positioned.sort((left, right) => left.index - right.index);
}

function ChartTip({ tip }) {
  if (!tip) return null;
  return (
    <div className="pointer-events-none absolute z-[6] mt-[-6px] min-w-32 -translate-x-1/2 -translate-y-full border border-border bg-surface px-2.5 py-2 shadow-lg" style={{ left: tip.left, top: tip.top }}>
      <div className="mb-1 text-xs font-bold text-foreground">{tip.title}</div>
      {tip.rows.map((row) => (
        <div className="flex items-center justify-between gap-4 text-xs leading-relaxed" key={row.label}>
          <span className="inline-flex items-center gap-1.5 text-muted">
            <span className="inline-block size-2 shrink-0" style={{ background: row.color }} />
            {row.label}
          </span>
          <span className="font-semibold tabular-nums text-foreground">{row.value}</span>
        </div>
      ))}
    </div>
  );
}

function ChartLegend({ series }) {
  return (
    <div className="mt-3.5 flex flex-wrap gap-3">
      {series.map((entry) => (
        <span className="inline-flex items-center gap-1.5 text-sm text-muted" key={entry.key}>
          <span className="size-2.5" style={{ background: entry.color }} />
          {entry.label}
        </span>
      ))}
    </div>
  );
}

const EMPTY_BARS = [0.42, 0.6, 0.34, 0.7, 0.5, 0.58, 0.46];

export function EmptyChart({ message, aspectRatio = "16 / 9" }) {
  return (
    <div className="relative mt-3.5 w-full opacity-95 [&_svg]:overflow-visible">
      <div className="relative w-full [&_svg]:block [&_svg]:overflow-visible" style={{ aspectRatio }}>
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
                        className="fill-surface-subtle opacity-50"
                        d={topRoundedPath(xScale(String(index)), barY, xScale.bandwidth(), innerHeight - barY, 0)}
                        key={index}
                      />
                    );
                  })}
                  <line className="stroke-[var(--border)] [shape-rendering:crispEdges]" x1={0} x2={innerWidth} y1={innerHeight} y2={innerHeight} />
                </g>
              </svg>
            );
          }}
        </ParentSize>
        <div className="pointer-events-none absolute inset-0 grid place-items-center p-5 text-center text-sm text-muted">{message}</div>
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
  bottomMargin,
  compactXLabels = false,
  valueScale = 100,
  valueSuffix = "%",
  valueDigits = 0,
  emptyMessage = "No data available yet.",
  forceHorizontalLabels = false,
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
    <div className="relative mt-3.5 w-full [&_svg]:overflow-visible">
      <div className="relative w-full [&_svg]:block [&_svg]:overflow-visible" onMouseLeave={clearTip} style={{ aspectRatio }}>
        <ParentSize debounceTime={10}>
          {({ width, height }) => (
            <BarChartSvg
              categories={categories}
              bottomMargin={bottomMargin}
              compactXLabels={compactXLabels}
              forceHorizontalLabels={forceHorizontalLabels}
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

/**
 * Compact two-variable comparison with a diagonal parity reference.
 * Points are directly labelled so the primary values do not depend on hover.
 */
export function ScatterChart({
  points,
  xLabel,
  yLabel,
  aspectRatio = "4 / 3",
  emptyMessage = "No data available yet.",
}) {
  const [tip, setTip] = useState(null);
  const clearTip = useCallback(() => setTip(null), []);
  const ready = points?.some(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
  );

  if (!ready) {
    return <EmptyChart aspectRatio={aspectRatio} message={emptyMessage} />;
  }

  return (
    <div className="relative mt-3.5 w-full [&_svg]:overflow-visible">
      <div className="relative w-full [&_svg]:block [&_svg]:overflow-visible" onMouseLeave={clearTip} style={{ aspectRatio }}>
        <ParentSize debounceTime={10}>
          {({ width, height }) => (
            <ScatterChartSvg
              height={height}
              points={points}
              setTip={setTip}
              width={width}
              xLabel={xLabel}
              yLabel={yLabel}
            />
          )}
        </ParentSize>
        <ChartTip tip={tip} />
      </div>
    </div>
  );
}

function ScatterChartSvg({ height, points, setTip, width, xLabel, yLabel }) {
  if (width < 10 || height < 10) return null;
  const left = 48;
  const right = 22;
  const top = 18;
  const bottom = 44;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  if (innerWidth < 40 || innerHeight < 40) return null;

  const xScale = scaleLinear({ domain: [0, 1], range: [0, innerWidth] });
  const yScale = scaleLinear({ domain: [0, 1], range: [innerHeight, 0] });
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const positionedPoints = layoutScatterPoints(points, {
    xScale,
    yScale,
    innerWidth,
    innerHeight,
  });

  return (
    <svg height={height} role="img" width={width}>
      <g transform={`translate(${left},${top})`}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="stroke-[var(--border)] opacity-50 [shape-rendering:crispEdges]" x1={xScale(tick)} x2={xScale(tick)} y1={0} y2={innerHeight} />
            <line className="stroke-[var(--border)] opacity-50 [shape-rendering:crispEdges]" x1={0} x2={innerWidth} y1={yScale(tick)} y2={yScale(tick)} />
            <text className="fill-muted text-xs tabular-nums" textAnchor="middle" x={xScale(tick)} y={innerHeight + 18}>{Math.round(tick * 100)}%</text>
            <text className="fill-muted text-xs tabular-nums" dy="0.32em" textAnchor="end" x={-9} y={yScale(tick)}>{Math.round(tick * 100)}%</text>
          </g>
        ))}
        <line className="stroke-[var(--text-faint)] opacity-70" strokeDasharray="5 5" x1={0} x2={innerWidth} y1={innerHeight} y2={0} />
        {positionedPoints.map((point) => {
          const x = point.screenX;
          const y = point.screenY;
          const tipRows = [
            { color: point.color, label: xLabel, value: formatPct(point.x) },
            { color: point.color, label: yLabel, value: formatPct(point.y) },
          ];
          const onHover = () => setTip({ left: left + x, top: top + y - 10, title: point.label, rows: tipRows });
          return (
            <g key={point.key || point.label}>
              {Math.abs(point.labelY - y) > 4 && (
                <line className="stroke-[var(--text-faint)] opacity-60" x1={x} x2={point.labelX} y1={y} y2={point.labelY} />
              )}
              <circle
                className="cursor-default stroke-background stroke-2"
                cx={x}
                cy={y}
                fill={point.color}
                onMouseEnter={onHover}
                onMouseMove={onHover}
                r={5}
              />
              <text className="fill-foreground text-xs font-medium" dominantBaseline="middle" textAnchor={point.alignRight ? "end" : "start"} x={point.labelX} y={point.labelY}>
                {point.label}
              </text>
            </g>
          );
        })}
        <text className="fill-muted text-xs font-medium" textAnchor="middle" x={innerWidth / 2} y={innerHeight + 38}>{xLabel}</text>
        <text className="fill-muted text-xs font-medium" textAnchor="middle" transform={`translate(${-38},${innerHeight / 2}) rotate(-90)`}>{yLabel}</text>
      </g>
    </svg>
  );
}

function BarChartSvg({ bottomMargin, categories, compactXLabels, forceHorizontalLabels, grouped, height, maxValue, minValue = 0, series, setTip, valueScale, valueSuffix, valueDigits = 0, width }) {
  const gradientPrefix = useId().replace(/[^A-Za-z0-9_-]/g, "");
  if (width < 10 || height < 10) return null;
  const labels = categories.map((category) => String(category.label ?? ""));
  const left = 46;
  const right = 16;
  const top = 18;
  const innerWidthBase = width - left - right;
  const bandPer = innerWidthBase / Math.max(1, categories.length);
  const longest = labels.reduce((max, label) => Math.max(max, label.length), 0);
  const estLabelPx = longest * 6.6;
  const rotate = !forceHorizontalLabels && estLabelPx > bandPer - 10;
  const bottom = bottomMargin ?? (rotate ? Math.min(104, 30 + Math.round(Math.sin(Math.PI / 5) * estLabelPx)) : 34);
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
  const xLabelClassName = compactXLabels ? "fill-muted text-[0.68rem]" : "fill-muted text-xs";
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
            <line className="stroke-[var(--border)] opacity-50 [shape-rendering:crispEdges]" x1={0} x2={innerWidth} />
            <text className="fill-muted text-xs tabular-nums" dy="0.32em" textAnchor="end" x={-10} y={0}>
              {tickFmt(tick)}
            </text>
          </g>
        ))}
        <line className="stroke-[var(--border)] [shape-rendering:crispEdges]" x1={0} x2={innerWidth} y1={zeroY} y2={zeroY} />
        {categories.map((category, index) => {
          const groupX = xScale(String(index)) ?? 0;
          return (
            <g key={index} transform={`translate(${groupX},0)`}>
              {series.map((entry, seriesIndex) => {
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
                const gradientId = `${gradientPrefix}-bar-${index}-${seriesIndex}`;
                const centerX = left + groupX + barX + barWidth / 2;
                const rows = tooltipRows(category, index);
                const onHover = () => setTip({ left: centerX, top: top + barTop - 10, title: category.label, rows });
                return (
                  <g key={entry.key}>
                    {barHeight > 0 && (
                      <>
                        <defs>
                          <BarGradient color={fill} id={gradientId} />
                        </defs>
                        <path
                          className="transition-opacity hover:opacity-80"
                          d={negative
                            ? bottomRoundedPath(barX, barTop, barWidth, barHeight, 0)
                            : topRoundedPath(barX, barTop, barWidth, barHeight, 0)}
                          fill={`url(#${gradientId})`}
                          onMouseEnter={onHover}
                          onMouseMove={onHover}
                        />
                      </>
                    )}
                    {showValueLabels && (
                      <text className="fill-foreground text-[0.68rem] font-semibold tabular-nums" textAnchor="middle" x={barX + barWidth / 2} y={negative ? barBottom + 13 : barTop - 6}>
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
                className={xLabelClassName}
                key={index}
                textAnchor="end"
                transform={`translate(${centerX},${innerHeight + 12}) rotate(-32)`}
              >
                {category.label}
              </text>
            );
          }
          return (
            <text className={xLabelClassName} key={index} textAnchor="middle" x={centerX} y={innerHeight + 18}>
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
  className = "",
  padding = 72,
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
    <div className="relative mt-3.5 w-full [&_svg]:overflow-visible">
      <div className={`relative w-full [&_svg]:block [&_svg]:overflow-visible ${className}`} onMouseLeave={clearTip} style={{ aspectRatio }}>
        <ParentSize debounceTime={10}>
          {({ width, height }) => (
            <RadarSvg
              categories={categories}
              height={height}
              levels={levels}
              maxValue={maxValue}
              padding={padding}
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

function RadarSvg({ categories, height, levels, maxValue, padding, series, setTip, valueScale, valueSuffix, width }) {
  if (width < 10 || height < 10) return null;
  const count = categories.length;
  const radius = Math.max(24, Math.min(width, height) / 2 - padding);
  const cx = width / 2;
  const cy = height / 2;
  const angleFor = (index) => -Math.PI / 2 + (index * 2 * Math.PI) / count;
  const pointAt = (index, r) => [cx + Math.cos(angleFor(index)) * r, cy + Math.sin(angleFor(index)) * r];
  const rings = Array.from({ length: levels }, (_, level) => (level + 1) / levels);
  const compact = width < 520;
  const maxLabelChars = compact ? 11 : width < 760 ? 14 : 18;
  const labelOffset = compact ? 12 : 18;
  const labels = categories.map((category, index) => {
    const angle = angleFor(index);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const lines = wrapRadarLabel(category.label, maxLabelChars);
    const labelHeight = lines.length * 13;
    const [ex, ey] = pointAt(index, radius);
    const centered = Math.abs(cos) < 0.08;
    const position = centered ? (sin < 0 ? "top" : "bottom") : cos > 0 ? "right" : "left";
    return {
      anchor: centered ? "middle" : position === "right" ? "start" : "end",
      ex,
      ey,
      height: labelHeight,
      index,
      lines,
      position,
      rawY: cy + sin * (radius + 20),
      x: centered ? cx : ex + (position === "right" ? labelOffset : -labelOffset),
    };
  });
  const topLabel = labels.find((label) => label.position === "top");
  const bottomLabel = labels.find((label) => label.position === "bottom");
  if (topLabel) topLabel.y = Math.max(18 + topLabel.height / 2, cy - radius - 20);
  if (bottomLabel) bottomLabel.y = Math.min(height - 18 - bottomLabel.height / 2, cy + radius + 20);
  const sideMinY = topLabel ? topLabel.y + topLabel.height / 2 + 12 : 18;
  const sideMaxY = bottomLabel ? bottomLabel.y - bottomLabel.height / 2 - 12 : height - 18;
  const sideLabels = ["left", "right"].flatMap((position) =>
    distributeRadarLabels(
      labels.filter((label) => label.position === position),
      sideMinY,
      sideMaxY,
      compact ? 6 : 8,
    ),
  );
  const positionedLabels = new Map(
    [...labels.filter((label) => label.position === "top" || label.position === "bottom"), ...sideLabels]
      .map((label) => [label.index, label]),
  );
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
          className="fill-none stroke-[var(--border)] opacity-50"
          key={level}
          points={categories.map((_, index) => pointAt(index, radius * ring).join(",")).join(" ")}
        />
      ))}
      {categories.map((category, index) => {
        const [ex, ey] = pointAt(index, radius);
        const label = positionedLabels.get(index);
        const labelEdgeX = label.position === "right"
          ? label.x - 6
          : label.position === "left"
            ? label.x + 6
            : label.x;
        return (
          <g key={index}>
            <line className="stroke-[var(--border)] [opacity:.45]" x1={cx} x2={ex} y1={cy} y2={ey} />
            {label.position === "left" || label.position === "right" ? (
              <line className="stroke-[var(--border)] [opacity:.45]" x1={ex} x2={labelEdgeX} y1={ey} y2={label.y} />
            ) : null}
            <RadarLabel anchor={label.anchor} lines={label.lines} x={label.x} y={label.y} />
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
              className="transition-[fill-opacity] hover:[fill-opacity:.26]"
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
                  className="cursor-default"
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

function wrapRadarLabel(label, maxChars) {
  const words = String(label ?? "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [""];
  return words.reduce((lines, word) => {
    const current = lines[lines.length - 1];
    if (!current || `${current} ${word}`.length > maxChars) lines.push(word);
    else lines[lines.length - 1] = `${current} ${word}`;
    return lines;
  }, []);
}

function distributeRadarLabels(items, minY, maxY, requestedGap) {
  if (!items.length) return [];
  const sorted = [...items].sort((left, right) => left.rawY - right.rawY);
  const totalHeight = sorted.reduce((sum, item) => sum + item.height, 0);
  const availableGap = sorted.length > 1
    ? (maxY - minY - totalHeight) / (sorted.length - 1)
    : 0;
  const gap = Math.max(2, Math.min(requestedGap, availableGap));
  let cursor = minY;

  sorted.forEach((item) => {
    const halfHeight = item.height / 2;
    item.y = Math.max(item.rawY, cursor + halfHeight);
    cursor = item.y + halfHeight + gap;
  });

  const overflow = sorted[sorted.length - 1].y + sorted[sorted.length - 1].height / 2 - maxY;
  if (overflow > 0) sorted.forEach((item) => { item.y -= overflow; });

  const underflow = minY - (sorted[0].y - sorted[0].height / 2);
  if (underflow > 0) sorted.forEach((item) => { item.y += underflow; });

  return sorted;
}

function RadarLabel({ anchor, lines, x, y }) {
  return (
    <text className="fill-muted text-xs" dominantBaseline="middle" textAnchor={anchor} x={x} y={y}>
      {lines.map((line, index) => (
        <tspan dy={index === 0 ? `${-0.5 * (lines.length - 1)}em` : "1em"} key={index} x={x}>
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

export function CapabilityRadar({ rows, scope = "all", selected }) {
  const { categories, series } = useMemo(() => {
    const capabilityOf = (row) => {
      const caps = {};
      const perceptionGroups = row.perception_groups || {};
      const cognitionGroups = row.cognition_groups || row.imagery_groups || {};
      const visibleGroups = scope === "perception"
        ? [perceptionGroups]
        : scope === "cognition"
          ? [cognitionGroups]
          : [perceptionGroups, cognitionGroups];
      visibleGroups.forEach((groups) => {
        Object.entries(groups).forEach(([name, group]) => {
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
  }, [rows, scope, selected]);

  if (categories.length < 3 || !series.length) {
    return <EmptyChart aspectRatio="16 / 8" message="Ranked models will populate capability profiles." />;
  }

  return (
    <RadarChart
      aspectRatio="16 / 8"
      categories={categories}
      className="min-h-[28rem] sm:min-h-[32rem]"
      padding={48}
      series={series}
    />
  );
}
