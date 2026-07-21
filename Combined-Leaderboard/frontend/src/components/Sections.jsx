import { useState } from "react";
import { Link } from "react-router-dom";
import { X, ZoomIn } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { apiUrl } from "@/lib/api";
import { ui } from "@/lib/styles";
import { cn } from "@/lib/utils";

export function SectionHead({ tag, title, body, accented = false, banded = false }) {
  if (banded) {
    return (
      <div className={cn(ui.sectionBand, accented && "!border-page-accent-border")}>
        <div className="max-w-copy">
          {tag && <div className={cn(ui.sectionTag, accented && "!text-page-accent")}>{tag}</div>}
          <h2 className={ui.heading2}>{title}</h2>
          {body && <p className={cn(ui.lede, "mt-4")}>{body}</p>}
        </div>
      </div>
    );
  }
  return (
    <div className="mb-10 max-w-copy max-sm:mb-6">
      {tag && <div className={cn("mb-3 text-xs font-semibold uppercase text-faint", accented && "!text-page-accent")}>{tag}</div>}
      <h2 className="mb-3 font-display text-3xl font-bold leading-tight max-sm:text-2xl">{title}</h2>
      {body && <p className="max-w-[62ch] text-base leading-relaxed text-muted">{body}</p>}
    </div>
  );
}

export function StatBand({ stats, accented = false }) {
  return (
    <section className="bg-background">
      <div className="container !px-0">
        <div className={cn("grid grid-cols-1 border-l border-t border-border sm:grid-cols-2 lg:grid-cols-4", accented && "!border-page-accent-border")}>
          {stats.map(([value, label], index) => (
            <div className={cn("flex min-w-0 flex-col border-b border-r border-border p-6", accented && "!border-page-accent-border")} key={label}>
              <span className={cn("mb-6 text-xs font-semibold uppercase text-faint", accented && "!text-page-accent")}>[{String(index + 1).padStart(2, "0")}]</span>
              <span className="font-display text-4xl font-bold tabular-nums text-foreground">{value}</span>
              <span className="mt-2 text-sm leading-relaxed text-muted">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function Pipeline({ steps, accented = false }) {
  return (
    <div className="relative">
      <ol className="relative m-0 list-none py-8 pl-0 md:py-12">
        {steps.map(([title, body], index) => {
          const isFirst = index === 0;
          const isLast = index === steps.length - 1;
          const contentSide = index % 2 === 0
            ? "md:col-start-1 md:row-start-1 md:ml-auto md:pr-10"
            : "md:col-start-3 md:row-start-1 md:pl-10";
          return (
            <li className="relative grid min-h-40 min-w-0 grid-cols-[64px_minmax(0,1fr)] md:grid-cols-[minmax(0,1fr)_72px_minmax(0,1fr)]" key={title}>
              <div className={cn("col-start-2 min-w-0 py-6 pr-6 md:max-w-[520px] md:py-8", contentSide)}>
                <h4 className="font-display text-xl font-bold leading-tight">{title}</h4>
                <p className="mt-3 text-sm leading-relaxed text-muted">{body}</p>
              </div>
              <div className="relative col-start-1 row-start-1 flex items-start justify-center pt-6 md:col-start-2 md:pt-8">
                {!isFirst && <span className={cn("absolute left-1/2 top-0 h-[44px] w-px -translate-x-1/2 bg-border md:h-[52px]", accented && "bg-page-accent-border")} aria-hidden="true" />}
                {!isLast && <span className={cn("absolute bottom-0 left-1/2 top-[44px] w-px -translate-x-1/2 bg-border md:top-[52px]", accented && "bg-page-accent-border")} aria-hidden="true" />}
                <span className={cn("relative z-10 grid size-10 place-items-center border border-border-strong bg-background text-xs font-semibold tabular-nums text-faint", accented && "border-page-accent text-page-accent")}>
                  {String(index + 1).padStart(2, "0")}
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function FindingScore({ value }) {
  const trailingOperator = value.match(/^(.*?)\s+(→|≠|↓|±)(?:\s+(.*))?$/);
  if (trailingOperator) {
    const [, primary, operator, secondary] = trailingOperator;
    return (
      <span className="inline-flex min-w-0 items-center tabular-nums">
        <span className="font-display text-3xl font-bold">{primary}</span>
        <span className="mx-2.5 shrink-0 font-sans text-3xl font-normal text-muted">{operator}</span>
        {secondary && <span className="font-display text-3xl font-bold">{secondary}</span>}
      </span>
    );
  }

  const leadingOperator = value.match(/^(≤)\s+(.*)$/);
  if (leadingOperator) {
    return (
      <span className="inline-flex min-w-0 items-center tabular-nums">
        <span className="mr-2.5 shrink-0 font-sans text-3xl font-normal text-muted">{leadingOperator[1]}</span>
        <span className="font-display text-3xl font-bold">{leadingOperator[2]}</span>
      </span>
    );
  }

  return <span className="font-display text-3xl font-bold tabular-nums">{value}</span>;
}

export function FindingGrid({ cards }) {
  return (
    <div className="divide-y divide-page-accent-border">
      {cards.map(([stat, title, body], index) => (
        <article className="grid min-w-0 grid-cols-[56px_minmax(0,1fr)] lg:grid-cols-[72px_minmax(0,1fr)]" key={title}>
          <div className="flex items-center justify-center border-r border-page-accent-border px-2 py-6 text-xs font-semibold tabular-nums text-page-accent lg:py-8">
            [{String(index + 1).padStart(2, "0")}]
          </div>
          <div className="grid min-w-0 sm:grid-cols-[minmax(220px,0.65fr)_minmax(0,1.35fr)]">
            <div className="flex min-w-0 items-center border-b border-page-accent-border px-5 py-6 sm:border-b-0 sm:border-r lg:px-7 lg:py-8">
              <FindingScore value={stat} />
            </div>
            <div className="min-w-0 px-5 py-6 lg:px-7 lg:py-8">
              <h3 className="font-display text-xl font-bold">{title}</h3>
              <p className="mt-3 max-w-[64ch] text-sm leading-relaxed text-muted">{body}</p>
            </div>
          </div>
        </article>
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
        <button className="absolute right-3 top-3 z-[2] grid size-9 cursor-pointer place-items-center border border-black/20 bg-white/90 text-black shadow-lg transition-colors hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-black" type="button" onClick={() => onZoom({ ...sample, imageSrc })} aria-label={`Zoom ${sample.title} sample image`}>
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
      {visual === "sp-cot" && <><span className="trace-line one" /><span className="trace-line two" /><span className="trace-line three" /><span className="trace-warn">3% lower</span></>}
      {visual === "sp-noimage" && <><span className="no-image-box" /><span className="prior-line one" /><span className="prior-line two" /></>}
      {visual === "sp-blank" && <><span className="blank-frame" /><span className="abstain-chip">Cannot determine</span></>}
    </div>
  );
}

export function SampleGrid({ samples }) {
  const [zoomedSample, setZoomedSample] = useState(null);

  return (
    <>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3">
        {samples.cards.map((sample) => {
          const sampleKey = sample.sampleId || sample.label;
          return (
            <div className="relative min-w-0" key={sampleKey}>
              <span className="pointer-events-none absolute -left-1 -right-1 top-0 z-10 border-t border-border-strong" aria-hidden="true" />
              <span className="pointer-events-none absolute -left-1 -right-1 bottom-0 z-10 border-b border-border-strong" aria-hidden="true" />
              <span className="pointer-events-none absolute -bottom-1 -top-1 left-0 z-10 border-l border-border-strong" aria-hidden="true" />
              <span className="pointer-events-none absolute -bottom-1 -top-1 right-0 z-10 border-r border-border-strong" aria-hidden="true" />
              <Card standalone className={cn("sample-card grid h-full grid-rows-[190px_1fr] overflow-hidden border border-border-strong p-6 !shadow-[inset_0_2px_0_var(--page-accent-line)] max-sm:grid-rows-[170px_1fr]", (sample.sampleId || sample.imagePath || sample.meta) && "verified-sample grid-rows-[280px_1fr] max-sm:grid-rows-[220px_1fr]")}>
                <SampleVisual sample={sample} onZoom={setZoomedSample} />
                <div className="flex min-w-0 flex-col pt-5">
                  <div className="mb-2 text-xs font-semibold uppercase text-page-accent">{sample.label}</div>
                  <h3 className="mb-3 font-display text-lg font-bold">{sample.title}</h3>
                  <p className="mb-3 text-sm leading-relaxed text-foreground">{sample.prompt}</p>
                  <p className="mb-4 text-sm leading-relaxed text-muted">{sample.detail}</p>
                  <div className="mt-auto flex flex-wrap gap-2">{sample.chips.map((chip) => <span className="inline-flex min-h-7 items-center border border-page-accent-chip-border bg-page-accent-chip px-2.5 py-1 text-xs text-muted" key={chip}>{chip}</span>)}</div>
                </div>
              </Card>
            </div>
          );
        })}
      </div>
      {zoomedSample && (
        <div className="fixed inset-0 z-[100] grid place-items-center p-5" role="dialog" aria-modal="true" aria-label={`${zoomedSample.title} sample image`} tabIndex={-1}>
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setZoomedSample(null)} />
          <div className="relative max-h-[92vh] w-full max-w-5xl overflow-auto border border-border bg-surface p-5 shadow-lg">
            <button className="absolute right-3 top-3 z-10 grid size-10 place-items-center border border-border-strong bg-surface text-foreground" type="button" onClick={() => setZoomedSample(null)} aria-label="Close image preview"><X size={18} aria-hidden="true" /></button>
            <figure className="m-0 grid gap-4">
              <img className="max-h-[72vh] w-full object-contain" src={zoomedSample.imageSrc} alt={`${zoomedSample.title} benchmark sample`} />
              <figcaption className="grid gap-1 border-t border-border pt-4">
                <span className="text-xs text-faint">{zoomedSample.sampleId}</span>
                <strong className="font-display text-lg">{zoomedSample.title}</strong>
                <p className="m-0 text-sm text-muted">{zoomedSample.imagePath}</p>
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
    <div className="grid grid-cols-1 border-l border-t border-border sm:grid-cols-2 lg:grid-cols-4">
      {results.rows.map((row, index) => (
        <div className="min-w-0 border-b border-r border-border bg-surface p-6" key={`${row.label}-${index}`}>
          <span className="mb-5 block text-xs font-semibold uppercase text-page-accent-muted">{row.label}</span>
          <strong className={cn("font-display text-3xl font-bold tabular-nums", row.tone === "pos" && "text-positive", row.tone === "neg" && "text-negative", row.tone === "warn" && "text-warning")}>{row.value}</strong>
          {typeof row.score === "number" && <div className="mt-4 h-1.5 overflow-hidden bg-surface-subtle" aria-hidden="true"><span className={cn("block h-full bg-page-accent", row.tone === "pos" && "bg-positive", row.tone === "neg" && "bg-negative", row.tone === "warn" && "bg-warning")} style={{ width: `${Math.min(Math.max(row.score, 0), 100)}%` }} /></div>}
          <p className="mt-4 text-sm leading-relaxed text-muted">{row.note}</p>
        </div>
      ))}
    </div>
  );
}

export function SpecList({ items }) {
  return <ul className="mt-4 grid list-none gap-2 p-0">{items.map(([k, v]) => <li className="flex gap-4 border-b border-border py-2 text-sm max-sm:flex-col max-sm:gap-1" key={k}><span className="min-w-36 font-semibold text-faint">{k}</span><span className="text-muted">{v}</span></li>)}</ul>;
}

export function ScoringBlock({ scoring, className }) {
  return (
    <section className={className}>
      <div className={cn(ui.sectionFrame, "!border-page-accent-border")}>
        <SectionHead tag={scoring.tag} title={scoring.title} body={scoring.body} accented banded />
        <dl className="grid border-b border-page-accent-border sm:grid-cols-2 xl:grid-cols-4">
          {scoring.specs.map(([label, value], index) => (
            <div className="min-w-0 border-b border-r border-page-accent-border p-6 sm:[&:nth-last-child(-n+2)]:border-b-0 xl:border-b-0 lg:p-8" key={label}>
              <dt className="flex items-center justify-between gap-3 text-xs font-semibold uppercase text-page-accent">
                <span>{label}</span>
                <span className="tabular-nums text-faint">[{String(index + 1).padStart(2, "0")}]</span>
              </dt>
              <dd className="mt-5 text-lg font-medium leading-relaxed text-muted">{value}</dd>
            </div>
          ))}
        </dl>
        <div className="grid border-b border-page-accent-border lg:grid-cols-[minmax(0,1fr)_auto]">
          <div className="min-w-0 p-6 lg:p-8">
            <span className="mb-2 block text-xs font-semibold uppercase text-page-accent">Run the evaluation</span>
            <p className="max-w-[62ch] text-sm leading-relaxed text-muted">{scoring.getStarted}</p>
          </div>
          <div className="flex min-w-0 items-center gap-2 p-6 max-sm:flex-col max-sm:items-stretch lg:p-8">
            <Button asChild size="sm"><a href={apiUrl(`/api/tasks/${scoring.taskId}/questions`)}>Questions</a></Button>
            {scoring.taskId === "spatial" && <Button asChild size="sm" variant="ghost"><a href={apiUrl("/api/spatial/manifest")}>Manifest (JSON)</a></Button>}
            <Button asChild size="sm" variant="ghost"><a href={apiUrl(`/api/tasks/${scoring.taskId}/template.jsonl`)}>Template (JSONL)</a></Button>
            <Button asChild variant="brand"><Link to="/submit">Go to submission →</Link></Button>
          </div>
        </div>
      </div>
    </section>
  );
}
