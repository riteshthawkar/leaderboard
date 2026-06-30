import { useState } from "react";
import { Link } from "react-router-dom";
import { X, ZoomIn } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export function SectionHead({ tag, title, body, accent }) {
  return (
    <div className="section-head">
      {tag && <div className="section-tag" style={accent ? { color: accent } : undefined}>{tag}</div>}
      <h2>{title}</h2>
      {body && <p>{body}</p>}
    </div>
  );
}

export function StatBand({ stats }) {
  return (
    <section className="stats-section">
      <div className="container">
        <div className="stat-band cols-4">
          {stats.map(([value, label], index) => (
            <div className="stat-cell" key={label}><span className="n">[{String(index + 1).padStart(2, "0")}]</span><span className="v">{value}</span><span className="l">{label}</span></div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function Pipeline({ steps }) {
  return (
    <ol className="build-pipeline" style={{ "--steps": steps.length }}>
      {steps.map(([title, body], index) => (
        <li className="build-step" key={title} style={{ "--i": index }}><span className="build-node">Step {String(index + 1).padStart(2, "0")}</span><div className="build-copy"><h4>{title}</h4><p>{body}</p></div></li>
      ))}
    </ol>
  );
}

export function FindingGrid({ cards }) {
  return (
    <div className="grid cols-3 ruled">
      {cards.map(([stat, title, body]) => (
        <Card className="finding" key={title}><div className="stat-line">{stat}</div><h3>{title}</h3><p>{body}</p></Card>
      ))}
    </div>
  );
}

function SampleVisual({ sample, onZoom }) {
  const { visual, label } = sample;
  const dots = Array.from({ length: 18 }, (_, index) => index);

  if (sample.imagePath) {
    const imageSrc = `${import.meta.env.BASE_URL}benchmark-samples/${sample.imagePath}`;

    return (
      <div className="sample-visual sample-image-frame" aria-label={`${label} benchmark image`}>
        <img src={imageSrc} alt={`${sample.title} benchmark sample`} loading="lazy" />
        <button className="sample-zoom-btn" type="button" onClick={() => onZoom({ ...sample, imageSrc })} aria-label={`Zoom ${sample.title} sample image`}>
          <ZoomIn size={16} aria-hidden="true" />
        </button>
      </div>
    );
  }

  if (sample.sampleId || sample.meta) {
    const rows = sample.meta || [
      ["Sample ID", sample.sampleId],
      ["Task", sample.task],
      ["Capability", sample.capability],
      ["Image path", sample.imagePath],
    ].filter(([, value]) => value);

    return (
      <div className="sample-visual sample-data-card" aria-label={`${label} benchmark metadata`}>
        {rows.map(([key, value]) => (
          <div className="sample-data-row" key={key}>
            <span>{key}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`sample-visual sample-${visual}`} aria-label={`${label} schematic`}>
      {visual === "dysm-closure" && <><span className="closure-seg a" /><span className="closure-seg b" /><span className="closure-seg c" /></>}
      {visual === "dysm-binding" && <><span className="shape square blue" /><span className="shape circle amber" /><span className="shape square blue" /><span className="shape triangle lime" /><span className="shape square blue" /><span className="shape square blue" /></>}
      {visual === "dysm-grid" && <><span className="grid-board" />{dots.slice(0, 9).map((dot) => <span className={`grid-token t${dot + 1}`} key={dot} />)}</>}
      {visual === "me-rotation" && <><span className="axis" /><span className="rot-object before" /><span className="rot-arrow">↷</span><span className="rot-object after" /></>}
      {visual === "me-folding" && <><span className="net-face center" /><span className="net-face top" /><span className="net-face right" /><span className="net-face bottom" /><span className="fold-line" /></>}
      {visual === "me-compose" && <><span className="part p1" /><span className="part p2" /><span className="part p3" /><span className="target-shape" /></>}
      {visual === "sp-standard" && <><span className="spatial-grid" /><span className="object cube" /><span className="object cylinder" /><span className="object sphere" /><span className="spatial-arrow" /></>}
      {visual === "sp-cot" && <><span className="trace-line one" /><span className="trace-line two" /><span className="trace-line three" /><span className="trace-warn">−3%</span></>}
      {visual === "sp-noimage" && <><span className="no-image-box" /><span className="prior-line one" /><span className="prior-line two" /></>}
      {visual === "sp-blank" && <><span className="blank-frame" /><span className="abstain-chip">Cannot determine</span></>}
    </div>
  );
}

export function SampleGrid({ samples }) {
  const [zoomedSample, setZoomedSample] = useState(null);

  return (
    <>
      <div className="sample-grid">
        {samples.cards.map((sample) => (
          <Card standalone className={`sample-card ${sample.sampleId || sample.imagePath || sample.meta ? "verified-sample" : ""}`} key={sample.sampleId || sample.label}>
            <SampleVisual sample={sample} onZoom={setZoomedSample} />
            <div className="sample-copy">
              <div className="sample-label">{sample.label}</div>
              <h3>{sample.title}</h3>
              <p className="sample-prompt">{sample.prompt}</p>
              <p>{sample.detail}</p>
              <div className="sample-chips">{sample.chips.map((chip) => <span className="chip" key={chip}>{chip}</span>)}</div>
            </div>
          </Card>
        ))}
      </div>
      {zoomedSample && (
        <div className="modal sample-lightbox" role="dialog" aria-modal="true" aria-label={`${zoomedSample.title} sample image`} tabIndex={-1}>
          <div className="modal-backdrop" onClick={() => setZoomedSample(null)} />
          <div className="modal-card sample-lightbox-card">
            <button className="icon-btn modal-close" type="button" onClick={() => setZoomedSample(null)} aria-label="Close image preview"><X size={18} aria-hidden="true" /></button>
            <figure className="sample-lightbox-figure">
              <img src={zoomedSample.imageSrc} alt={`${zoomedSample.title} benchmark sample`} />
              <figcaption>
                <span>{zoomedSample.sampleId}</span>
                <strong>{zoomedSample.title}</strong>
                <p>{zoomedSample.imagePath}</p>
              </figcaption>
            </figure>
          </div>
        </div>
      )}
    </>
  );
}

export function ResultStrip({ results }) {
  return (
    <div className="result-strip">
      {results.rows.map((row, index) => (
        <div className={`result-item ${row.tone || "base"}`} key={`${row.label}-${index}`}>
          <span className="result-label">{row.label}</span>
          <strong>{row.value}</strong>
          {typeof row.score === "number" && <div className="result-bar" aria-hidden="true"><span style={{ width: `${Math.min(Math.max(row.score, 0), 100)}%` }} /></div>}
          <p>{row.note}</p>
        </div>
      ))}
    </div>
  );
}

export function SpecList({ items }) {
  return <ul className="spec-list" style={{ marginTop: 16 }}>{items.map(([k, v]) => <li key={k}><span className="k">{k}</span><span className="v">{v}</span></li>)}</ul>;
}

export function ScoringBlock({ scoring, className = "section alt" }) {
  return (
    <section className={className}>
      <div className="container">
        <div className="grid cols-2 copy-card-stack" style={{ alignItems: "start" }}>
          <div><div className="section-tag">{scoring.tag}</div><h2>{scoring.title}</h2><p>{scoring.body}</p><SpecList items={scoring.specs} /></div>
          <Card standalone className="callout bench-accent">
            <h4>Get started</h4>
            <p className="muted">{scoring.getStarted}</p>
            <div className="scoring-actions">
              <div className="dl-row">
                <Button asChild size="sm"><a href={`/api/tasks/${scoring.taskId}/questions`}>Questions (JSON)</a></Button>
                {scoring.taskId === "spatial" && <Button asChild size="sm" variant="ghost"><a href="/api/spatial/manifest">Manifest (JSON)</a></Button>}
                <Button asChild size="sm" variant="ghost"><a href={`/api/tasks/${scoring.taskId}/template.json`}>Template (JSON)</a></Button>
                {scoring.taskId !== "spatial" && <Button asChild size="sm" variant="ghost"><a href={`/api/tasks/${scoring.taskId}/template.csv`}>Template (CSV)</a></Button>}
              </div>
              <Button asChild variant="brand"><Link to="/submit">Go to submission →</Link></Button>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
}