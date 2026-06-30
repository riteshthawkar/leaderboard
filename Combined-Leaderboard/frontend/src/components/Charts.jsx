import { useMemo } from "react";
import { curveCatmullRom, curveMonotoneX } from "@visx/curve";
import { Area, AreaChart } from "@/components/charts/area-chart";
import { Grid } from "@/components/charts/grid";
import { XAxis } from "@/components/charts/x-axis";
import { ChartTooltip } from "@/components/charts/tooltip";
import { fmtDelta, prettyLabel } from "@/lib/utils";

const palette = ["#14b8a6", "#f97316", "#38bdf8", "#e11d48", "#a855f7", "#84cc16"];
const positiveColor = "#22c55e";
const negativeColor = "#f43f5e";

function capabilityData(row) {
  const caps = {};
  ["perception_groups", "imagery_groups"].forEach((key) => {
    Object.entries(row[key] || {}).forEach(([name, group]) => { caps[name] = group.accuracy; });
  });
  return caps;
}

function EmptyChart({ message }) {
  const data = Array.from({ length: 8 }, (_, index) => ({
    date: new Date(2026, 0, index + 1),
    baseline: 0.24 + Math.sin(index * 0.85) * 0.08 + index * 0.035,
    comparison: 0.18 + Math.cos(index * 0.75) * 0.07 + index * 0.028,
  }));

  return (
    <div className="bklit-chart-wrap is-empty">
      <AreaChart
        animationDuration={700}
        aspectRatio="16 / 7"
        data={data}
        margin={{ top: 22, right: 18, bottom: 34, left: 18 }}
        xDataKey="date"
        yDomainTween={false}
      >
        <Grid horizontal vertical={false} />
        <Area dataKey="baseline" fill={palette[0]} fillOpacity={0.2} stroke={palette[0]} strokeWidth={2} showHighlight={false} />
        <Area dataKey="comparison" fill={palette[1]} fillOpacity={0.18} stroke={palette[1]} strokeWidth={2} showHighlight={false} />
      </AreaChart>
      <div className="chart-empty-overlay">{message}</div>
    </div>
  );
}

export function CapabilityRadar({ rows, selected }) {
  const { chartData, series } = useMemo(() => {
    const axes = new Set();
    rows.forEach((row) => Object.keys(capabilityData(row)).forEach((key) => axes.add(key)));
    const orderedAxes = Array.from(axes).sort();
    const visible = rows.filter((row) => selected.includes(row.model_name));
    const data = orderedAxes.map((axis, index) => {
      const point = { date: new Date(2026, 0, index + 1), label: prettyLabel(axis) };
      visible.forEach((row) => { point[row.model_name] = capabilityData(row)[axis] ?? 0; });
      return point;
    });
    return { chartData: data, series: visible };
  }, [rows, selected]);

  if (chartData.length < 3 || !series.length) {
    return <EmptyChart message="Ranked models will populate capability profiles." />;
  }

  return (
    <div className="bklit-chart-wrap">
      <AreaChart
        animationDuration={800}
        aspectRatio="16 / 8"
        data={chartData}
        margin={{ top: 26, right: 28, bottom: 42, left: 28 }}
        revealSignature={series.map((row) => row.model_name).join("|")}
        xDataKey="date"
        yDomainTween={false}
      >
        <Grid horizontal vertical={false} numTicksRows={5} />
        {series.map((row, index) => (
          <Area
            curve={curveCatmullRom}
            dataKey={row.model_name}
            fill={palette[index % palette.length]}
            fillOpacity={0.16}
            key={row.model_name}
            showMarkers
            stroke={palette[index % palette.length]}
            strokeWidth={2.5}
          />
        ))}
        <XAxis numTicks={Math.min(5, chartData.length)} />
        <ChartTooltip
          showDatePill={false}
          rows={(point) => series.map((row, index) => ({
            color: palette[index % palette.length],
            label: row.model_name,
            value: `${((point[row.model_name] ?? 0) * 100).toFixed(1)}%`,
          }))}
        />
      </AreaChart>
      <div className="radar-legend">
        {series.map((row, index) => (
          <span className="legend-item" key={row.model_name}>
            <span className="legend-dot" style={{ background: palette[index % palette.length] }} />
            {row.model_name}
          </span>
        ))}
      </div>
    </div>
  );
}

export function CotChart({ rows }) {
  const data = useMemo(() => rows
    .filter((row) => row.diagnostics?.cot_delta != null)
    .map((row, index) => ({
      date: new Date(2026, 0, index + 1),
      model: row.model_name,
      cotDelta: row.diagnostics.cot_delta,
      positive: Math.max(row.diagnostics.cot_delta, 0),
      negative: Math.min(row.diagnostics.cot_delta, 0),
    })), [rows]);

  if (!data.length) {
    return <EmptyChart message="Ranked spatial models will populate CoT diagnostics." />;
  }

  return (
    <div className="bklit-chart-wrap">
      <AreaChart
        animationDuration={800}
        aspectRatio="16 / 7"
        data={data}
        margin={{ top: 26, right: 28, bottom: 42, left: 28 }}
        revealSignature={data.map((row) => row.model).join("|")}
        xDataKey="date"
        yDomainTween={false}
      >
        <Grid horizontal vertical={false} rowTickValues={[0]} highlightRowValues={[0]} />
        <Area curve={curveMonotoneX} dataKey="positive" fill={positiveColor} fillOpacity={0.18} stroke={positiveColor} strokeWidth={2.5} showMarkers />
        <Area curve={curveMonotoneX} dataKey="negative" fill={negativeColor} fillOpacity={0.18} stroke={negativeColor} strokeWidth={2.5} showMarkers />
        <XAxis numTicks={Math.min(5, data.length)} />
        <ChartTooltip
          showDatePill={false}
          rows={(point) => [{
            color: point.cotDelta < 0 ? negativeColor : positiveColor,
            label: point.model,
            value: `${fmtDelta(point.cotDelta)}%`,
          }]}
        />
      </AreaChart>
      <div className="radar-legend">
        <span className="legend-item"><span className="legend-dot" style={{ background: positiveColor }} />CoT improved</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: negativeColor }} />CoT degraded</span>
      </div>
    </div>
  );
}
