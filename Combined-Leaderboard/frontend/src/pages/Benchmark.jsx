import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { PageHero } from "@/components/Hero";
import { Citation } from "@/components/Citation";
import { FindingGrid, Pipeline, ResultStrip, SampleGrid, ScoringBlock, SectionHead, SpecList, StatBand } from "@/components/Sections";
import { Card } from "@/components/ui/card";
import { benchmarkPages } from "@/data/benchmarks";
import { getJSON } from "@/lib/api";

const keyBySlug = { "do-you-see-me": "dysm", "minds-eye": "minds_eye", spatial: "spatial" };

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
  useEffect(() => {
    if (page.id === "spatial") getJSON("/api/tasks/spatial/info").then((info) => setDatasets(info.datasets || [])).catch(() => setDatasets([]));
  }, [page.id]);

  return (
    <>
      <PageHero {...page} />
      <StatBand stats={page.stats} />
      <PremiseSection page={page} />
      <SamplesSection page={page} />
      <TaskTableSection page={page} />
      <TaxonomySection page={page} />
      <EvaluationSection page={page} />
      <BuildSection page={page} />
      <ResultsSection page={page} />
      <FindingsSection page={page} />
      {page.id === "spatial" && <SpatialDataSection page={page} datasets={datasets} />}
      <ScoringBlock scoring={page.scoring} className={page.id === "spatial" ? "section" : "section alt"} />
      <Citation citation={page.citation} />
    </>
  );
}