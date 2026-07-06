import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { PageHero } from "@/components/Hero";
import { BenchmarkModelChart, BarChart } from "@/components/Charts";
import { Citation } from "@/components/Citation";
import { FindingGrid, Pipeline, ResultStrip, SampleGrid, ScoringBlock, SectionHead, SpecList, StatBand } from "@/components/Sections";
import { Card } from "@/components/ui/card";
import { benchmarkPages } from "@/data/benchmarks";
import { getJSON } from "@/lib/api";

const keyBySlug = { "do-you-see-me": "dysm", "minds-eye": "minds_eye", spatial: "spatial" };

const benchmarkChart = {
  dysm: { endpoint: "/api/leaderboard/visual-cognition", metricKey: "perception_accuracy", metricLabel: "Perception accuracy", color: "#6366f1" },
  minds_eye: { endpoint: "/api/leaderboard/visual-cognition", metricKey: "imagery_accuracy", metricLabel: "Imagery accuracy", color: "#a855f7" },
  spatial: { endpoint: "/api/leaderboard/spatial", metricKey: "accuracy", metricLabel: "Spatial accuracy", color: "#14b8a6" },
};

function PremiseCallout({ premise, pageId }) {
  return (
    <Card standalone className="callout bench-accent premise-callout">
      <h4>{premise.calloutTitle || (pageId === "spatial" ? "The headline result" : "The headline gap")}</h4>
      {premise.calloutItems ? <div className="insight-list">{premise.calloutItems.map(([label, title, body]) => <div className="insight-item" key={label}><span>{label}</span><strong>{title}</strong><p>{body}</p></div>)}</div> : <div className="kpi-row">{premise.kpis.map(([label, value, tone]) => <div className="kpi" key={label}><div className="kpi-label">{label}</div><div className={`kpi-val ${tone}`}>{value}</div></div>)}</div>}
      <p className="muted small">{premise.note}</p>
    </Card>
  );
}

function PremiseSection({ page }) {
  return (
    <section className={`section prose ${page.id}-premise`}>
      <div className="container">
        <div className="grid cols-2 copy-card-stack" style={{ alignItems: "start" }}>
          <div>
            <div className="section-tag" style={{ color: page.accent }}>{page.premise.tag}</div>
            <h2>{page.premise.title}</h2>
            <p className="lede">{page.premise.body}</p>
            <SpecList items={page.premise.specs} />
            {page.premise.chips && <div style={{ marginTop: 20, display: "flex", flexWrap: "wrap", gap: 8 }}>{page.premise.chips.map((chip) => <span className="chip" key={chip}>{chip}</span>)}</div>}
          </div>
          <PremiseCallout premise={page.premise} pageId={page.id} />
        </div>
      </div>
    </section>
  );
}

function TaskTableSection({ page, className = "section alt" }) {
  if (!page.taskSection) return null;
  return (
    <section className={className}>
      <div className="container">
        <SectionHead {...page.taskSection} accent={page.accent} />
        <TaskTable table={page.taskSection.table} />
      </div>
    </section>
  );
}

function TaskTable({ table }) {
  return (
    <div className="table-wrap">
      <table className="lb-table">
        <thead><tr>{table.headers.map((header) => <th key={header}>{header}</th>)}</tr></thead>
        <tbody>{table.rows.map((row, rowIndex) => <tr key={`${row[0]}-${rowIndex}`}>{row.map((cell, index) => <td key={`${row[0]}-${index}`}>{index === 0 ? <strong>{cell}</strong> : cell}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function SamplesSection({ page, className = "section alt project-samples" }) {
  if (!page.samples) return null;
  return <section className={className}><div className="container"><SectionHead {...page.samples} accent={page.accent} /><SampleGrid samples={page.samples} /></div></section>;
}

function TaxonomySection({ page }) {
  if (!page.taxonomy) return null;
  return (
    <section className="section alt">
      <div className="container">
        <SectionHead {...page.taxonomy} accent={page.accent} />
        <div className="grid cols-3 ruled">{page.taxonomy.cards.map(([number, stat, title, body]) => <Card className="finding" key={title}><span className="card-n">{number}</span><div className="stat-line">{stat}</div><h3>{title}</h3><p>{body}</p></Card>)}</div>
      </div>
    </section>
  );
}

function EvaluationSection({ page }) {
  if (!page.evaluation) return null;
  return <section className="section alt"><div className="container"><SectionHead {...page.evaluation} accent={page.accent} /><Pipeline steps={page.evaluation.steps} /></div></section>;
}

function BuildSection({ page, className = "section build-section" }) {
  if (!page.pipeline) return null;
  return <section className={className}><div className="container"><SectionHead {...page.pipeline} accent={page.accent} /><Pipeline steps={page.pipeline.steps} /></div></section>;
}

function ResultsSection({ page, className = "section result-section" }) {
  if (!page.results) return null;
  return <section className={className}><div className="container"><SectionHead {...page.results} accent={page.accent} /><ResultStrip results={page.results} /></div></section>;
}

function FindingsSection({ page, className = "section" }) {
  return <section className={className}><div className="container"><SectionHead {...page.findings} accent={page.accent} /><FindingGrid cards={page.findings.cards} /></div></section>;
}

const FIG_POS = "#14b8a6";
const FIG_NEG = "#e11d48";
const FIG_BASE = "#6366f1";

function FigureChart({ chart }) {
  const scale = chart.scale ?? 1;
  const suffix = chart.suffix ?? (chart.kind === "diverging" ? " pts" : "%");
  const aspect = chart.wide ? "24 / 7" : "16 / 10";
  const digits = chart.digits ?? (chart.kind === "diverging" ? 1 : 0);
  if (chart.kind === "grouped") {
    return (
      <BarChart
        aspectRatio={aspect}
        categories={chart.categories.map((label) => ({ label }))}
        series={chart.series.map((entry, seriesIndex) => ({ key: entry.key || `s${seriesIndex}`, label: entry.label, color: entry.color, valueFor: (_category, index) => entry.values[index] }))}
        valueScale={scale}
        valueSuffix={suffix}
        valueDigits={digits}
        showLegend
      />
    );
  }
  const diverging = chart.kind === "diverging";
  return (
    <BarChart
      aspectRatio={aspect}
      categories={chart.bars.map(([label]) => ({ label }))}
      series={[{
        key: "v",
        label: chart.unit || "",
        color: chart.color || (diverging ? FIG_POS : FIG_BASE),
        colorFor: (_category, index) => {
          const bar = chart.bars[index];
          if (diverging) return bar[1] >= 0 ? FIG_POS : FIG_NEG;
          return bar[2] || chart.color || FIG_BASE;
        },
        valueFor: (_category, index) => chart.bars[index][1],
      }]}
      valueScale={scale}
      valueSuffix={suffix}
      valueDigits={digits}
      showLegend={false}
    />
  );
}

function FindingsFiguresSection({ page, className = "section findings-figures" }) {
  if (!page.figures?.charts?.length) return null;
  const f = page.figures;
  return (
    <section className={className}>
      <div className="container">
        <SectionHead tag={f.tag} title={f.title} body={f.body} accent={page.accent} />
        <div className="figure-grid">
          {f.charts.map((chart, index) => (
            <div className={`figure-card${chart.wide ? " is-wide" : ""}`} key={chart.title || index}>
              <h3>{chart.title}</h3>
              <p className="figure-caption">{chart.caption}</p>
              <FigureChart chart={chart} />
              {chart.source && <p className="figure-source">{chart.source}</p>}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ModelPerformanceSection({ page, rows, className = "section" }) {
  const config = benchmarkChart[page.id];
  if (!config) return null;
  return (
    <section className={`${className} model-performance-section`}>
      <div className="container">
        <SectionHead
          tag="Leaderboard"
          title="Model performance on this benchmark"
          body={`How ranked models score on ${page.navLabel}, measured by ${config.metricLabel.toLowerCase()} per submitted model.`}
          accent={page.accent}
        />
        <div className="viz-card benchmark-chart-card">
          <BenchmarkModelChart
            rows={rows}
            metricFor={(row) => row[config.metricKey]}
            metricLabel={config.metricLabel}
            color={config.color}
            emptyMessage="Model scores will appear here once submissions are ranked for this benchmark."
          />
        </div>
      </div>
    </section>
  );
}

function SpatialDataSection({ page, datasets }) {
  return (
    <section className="section alt">
      <div className="container">
        <SectionHead tag="The data" title="Thirteen spatial benchmarks, one policy" body="Static 2D relations, 3D geometry, and dynamic/temporal understanding — unified under a single evaluation and scoring scheme. Datasets are downloaded from their official sources on your machine; none are redistributed here." accent={page.accent} />
        <div className="table-wrap"><table className="lb-table"><thead><tr><th>Dataset</th><th>Type</th><th className="num">Approx. n</th><th>Capabilities</th><th>License</th></tr></thead><tbody>{datasets.length ? datasets.map((dataset) => <tr key={dataset.name}><td><strong>{dataset.name}</strong></td><td>{dataset.type}</td><td className="num">{dataset.approx_n ? `~${dataset.approx_n}` : "—"}</td><td>{(dataset.tags || []).map((tag) => <span className="chip" key={tag}>{tag}</span>)}</td><td>{dataset.license || "—"}</td></tr>) : <tr><td colSpan="5" className="empty-row">Manifest unavailable.</td></tr>}</tbody></table></div>
      </div>
    </section>
  );
}

export function Benchmark() {
  const { slug } = useParams();
  const page = benchmarkPages[keyBySlug[slug]] || benchmarkPages.dysm;
  const [datasets, setDatasets] = useState([]);
  const [modelRows, setModelRows] = useState([]);
  useEffect(() => {
    if (page.id === "spatial") getJSON("/api/tasks/spatial/info").then((info) => setDatasets(info.datasets || [])).catch(() => setDatasets([]));
  }, [page.id]);
  useEffect(() => {
    const config = benchmarkChart[page.id];
    if (!config) { setModelRows([]); return; }
    getJSON(config.endpoint).then((data) => setModelRows(data.leaderboard || [])).catch(() => setModelRows([]));
  }, [page.id]);

  return (
    <>
      <PageHero {...page} />
      <StatBand stats={page.stats} />
      <PremiseSection page={page} />
      <ModelPerformanceSection page={page} rows={modelRows} className="section alt" />
      <SamplesSection page={page} />
      <TaskTableSection page={page} />
      <TaxonomySection page={page} />
      <EvaluationSection page={page} />
      <BuildSection page={page} />
      <ResultsSection page={page} />
      <FindingsSection page={page} />
      <FindingsFiguresSection page={page} />
      {page.id === "spatial" && <SpatialDataSection page={page} datasets={datasets} />}
      <ScoringBlock scoring={page.scoring} className={page.id === "spatial" ? "section" : "section alt"} />
      <Citation citation={page.citation} />
    </>
  );
}