import { useEffect, useState } from "react";
import { useParams } from "react-router";
import { PageHero } from "@/components/Hero";
import { BenchmarkModelChart, BarChart } from "@/components/Charts";
import { Citation } from "@/components/Citation";
import { FindingGrid, Pipeline, ResultStrip, SampleGrid, ScoringBlock, SectionHead, StatBand } from "@/components/Sections";
import { ArrowUpRight, Eye, ScanSearch, SlidersHorizontal } from "lucide-react";
import { benchmarkPages } from "@/data/benchmarks";
import { errorMessage, getJSON } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ui } from "@/lib/styles";
import { NotFound } from "@/pages/NotFound";

const keyBySlug = { "do-you-see-me": "dysm", "minds-eye": "minds_eye", spatial: "spatial" };
const benchmarkTableClasses = "[&_thead_th]:!text-page-accent-muted [&_tbody_tr:hover]:!bg-page-accent-chip";
const taskTableClasses = "[&_thead_th]:!py-4 [&_tbody_td]:!py-5";
const benchmarkFrameClasses = "!border-page-accent-border";
const paperSections = [
  ["Overview", "overview"],
  ["Research question", "research-question"],
  ["Tasks & data", "tasks"],
  ["Method", "method"],
  ["Evidence", "results"],
  ["Live results", "live-results"],
  ["Submission", "submission"],
  ["Citation", "cite"],
];

function BenchmarkSection({ head, children, bodyClassName, className, id, padded = false }) {
  return (
    <section className={className} id={id}>
      <div className={cn(ui.sectionFrame, benchmarkFrameClasses)}>
        <SectionHead {...head} accented banded />
        <div className={cn(padded && ui.sectionBody, bodyClassName)}>{children}</div>
      </div>
    </section>
  );
}

function PaperProjectOverview({ page }) {
  const { citation, project } = page;
  return (
    <section className="scroll-mt-24 bg-background" id="overview">
      <div className={cn(ui.sectionFrame, benchmarkFrameClasses)}>
        <nav className="flex min-w-0 items-center gap-6 overflow-x-auto border-b border-page-accent-border px-6 py-4 lg:px-8" aria-label={`${page.navLabel} paper sections`}>
          <span className="shrink-0 text-xs font-semibold uppercase text-page-accent">On this page</span>
          <div className="flex min-w-max items-center gap-5">
            {paperSections.map(([label, anchor]) => (
              <a className="whitespace-nowrap text-sm text-muted transition-colors hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-page-accent" href={`#${anchor}`} key={anchor}>
                {label}
              </a>
            ))}
          </div>
        </nav>

        <div className="grid border-b border-page-accent-border lg:grid-cols-[minmax(0,1.55fr)_minmax(280px,0.45fr)]">
          <article className="min-w-0 border-b border-page-accent-border p-7 lg:border-b-0 lg:border-r lg:p-10">
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-semibold uppercase">
              <span className="text-page-accent">Research project</span>
              <span className="text-faint">{citation.venue} · {citation.year}</span>
            </div>
            <h2 className="mt-6 max-w-[30ch] font-display text-3xl font-bold leading-tight text-foreground sm:text-4xl">
              {citation.title}
            </h2>
            <p className="mt-5 max-w-[72ch] text-sm leading-relaxed text-muted sm:text-base">{citation.authors}</p>

            <div className="mt-10 border-t border-page-accent-border pt-8">
              <h3 className="text-xs font-semibold uppercase text-page-accent">Abstract</h3>
              <p className="mt-4 max-w-[76ch] text-base leading-relaxed text-foreground sm:text-lg">{project.abstract}</p>
            </div>

            <div className="mt-9 grid border-l border-t border-page-accent-border sm:grid-cols-2">
              <div className="min-w-0 border-b border-r border-page-accent-border p-5 sm:p-6">
                <span className="text-xs font-semibold uppercase text-page-accent">Research question</span>
                <p className="mt-3 text-sm leading-relaxed text-muted">{project.researchQuestion}</p>
              </div>
              <div className="min-w-0 border-b border-r border-page-accent-border p-5 sm:p-6">
                <span className="text-xs font-semibold uppercase text-page-accent">Project artifact</span>
                <p className="mt-3 text-sm leading-relaxed text-muted">{project.artifact}</p>
              </div>
            </div>
          </article>

          <aside className="min-w-0" aria-label="Paper details">
            <div className="border-b border-page-accent-border px-6 py-5 lg:px-8">
              <span className="text-xs font-semibold uppercase text-page-accent">Paper details</span>
            </div>
            <dl className="divide-y divide-page-accent-border">
              <div className="px-6 py-5 lg:px-8">
                <dt className="text-xs font-semibold uppercase text-faint">Framework role</dt>
                <dd className="mt-2 text-sm font-medium leading-relaxed text-foreground">{project.frameworkRole}</dd>
              </div>
              <div className="px-6 py-5 lg:px-8">
                <dt className="text-xs font-semibold uppercase text-faint">Study design</dt>
                <dd className="mt-2 text-sm leading-relaxed text-muted">{project.studyDesign}</dd>
              </div>
              <div className="px-6 py-5 lg:px-8">
                <dt className="text-xs font-semibold uppercase text-faint">Publication</dt>
                <dd className="mt-2 text-sm leading-relaxed text-muted">{citation.venue} · {citation.year}</dd>
              </div>
              <div className="px-6 py-5 lg:px-8">
                <dt className="text-xs font-semibold uppercase text-faint">Identifier</dt>
                <dd className="mt-2 text-sm leading-relaxed text-muted">arXiv:{citation.arxiv}</dd>
              </div>
            </dl>
            <div className="px-6 py-6 lg:px-8">
              <a className="inline-flex min-h-10 items-center gap-2 border border-page-accent-border px-4 py-2 text-sm font-semibold text-page-accent transition-colors hover:bg-page-accent-soft focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-page-accent" href={citation.paperUrl} target="_blank" rel="noreferrer">
                Read the paper <ArrowUpRight className="size-4" strokeWidth={1.75} aria-hidden="true" />
              </a>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}

function cognitionAccuracy(row) {
  return row?.cognition_accuracy ?? row?.imagery_accuracy ?? null;
}

const benchmarkChart = {
  dysm: { endpoint: "/api/leaderboard/visual-cognition", metricKey: "perception_accuracy", metricLabel: "Perception macro average", color: "var(--dysm)" },
  minds_eye: { endpoint: "/api/leaderboard/visual-cognition", metricFor: cognitionAccuracy, metricLabel: "Cognition macro average", color: "var(--me)" },
  spatial: { endpoint: "/api/leaderboard/spatial", metricKey: "macro_accuracy", metricLabel: "Spatial macro average", color: "var(--spatial)" },
};

function MindsEyePremiseSection({ page }) {
  return (
    <BenchmarkSection head={page.premise} id="research-question" className="scroll-mt-24">
      <div className="grid border-b border-page-accent-border md:grid-cols-2">
        <article className="min-w-0 border-b border-page-accent-border p-7 md:border-b-0 md:border-r lg:p-10">
          <span className="text-xs font-medium uppercase text-page-accent">Core premise</span>
          <p className="mt-8 max-w-[26ch] font-display text-xl font-medium leading-relaxed text-foreground lg:text-2xl">
            {page.premise.thesis}
          </p>
          <p className="mt-6 max-w-[54ch] text-sm leading-relaxed text-muted">{page.premise.note}</p>
        </article>

        <div className="min-w-0 p-7 lg:p-10">
          <span className="text-xs font-medium uppercase text-page-accent">{page.premise.calloutTitle}</span>
          <div className="mt-8 grid gap-8">
            {page.premise.calloutItems.map(([label, title, body]) => (
              <div className="grid min-w-0 grid-cols-[28px_minmax(0,1fr)] gap-4" key={label}>
                <span className="pt-0.5 text-xs font-medium tabular-nums text-page-accent">{label}</span>
                <div className="min-w-0">
                  <h3 className="font-display text-sm font-medium leading-snug text-foreground">{title}</h3>
                  <p className="mt-1.5 max-w-[58ch] text-sm leading-relaxed text-muted">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </BenchmarkSection>
  );
}

function SpatialPremiseSection({ page }) {
  return (
    <BenchmarkSection head={page.premise} id="research-question" className="scroll-mt-24">
      <div className="border-b border-page-accent-border px-7 py-8 lg:px-10 lg:py-10">
        <span className="text-xs font-medium uppercase text-page-accent">Diagnostic premise</span>
        <p className="mt-5 max-w-[48ch] font-display text-xl font-medium leading-relaxed text-foreground lg:text-2xl">
          {page.premise.thesis}
        </p>
      </div>

      <div className="grid border-b border-page-accent-border md:grid-cols-2">
        {page.premise.reasoningPaths.map(([label, summary, steps], pathIndex) => (
          <article className={cn("min-w-0 p-7 lg:p-10", pathIndex === 0 && "border-b border-page-accent-border md:border-b-0 md:border-r")} key={label}>
            <span className="text-xs font-medium tabular-nums text-page-accent">PATH {String(pathIndex + 1).padStart(2, "0")}</span>
            <h3 className="mt-3 font-display text-lg font-medium leading-tight text-foreground">{label}</h3>
            <p className="mt-3 max-w-[48ch] text-sm leading-relaxed text-muted">{summary}</p>
            <ol className="mt-8 grid gap-6">
              {steps.map(([title, body], stepIndex) => (
                <li className="grid min-w-0 grid-cols-[28px_minmax(0,1fr)] gap-4" key={title}>
                  <span className="pt-0.5 text-xs font-medium tabular-nums text-page-accent">{String(stepIndex + 1).padStart(2, "0")}</span>
                  <div className="min-w-0">
                    <h4 className="font-display text-sm font-medium leading-snug text-foreground">{title}</h4>
                    <p className="mt-1.5 text-sm leading-relaxed text-muted">{body}</p>
                  </div>
                </li>
              ))}
            </ol>
          </article>
        ))}
      </div>

      <div className="flex flex-col gap-3 border-b border-page-accent-border px-7 py-6 sm:flex-row sm:items-baseline sm:gap-8 lg:px-10">
        <span className="shrink-0 text-xs font-medium uppercase text-page-accent">What to take away</span>
        <p className="max-w-[78ch] text-sm leading-relaxed text-muted">{page.premise.note}</p>
      </div>
    </BenchmarkSection>
  );
}

function PremiseSection({ page }) {
  if (page.id === "dysm") {
    const contributionIcons = [Eye, SlidersHorizontal, ScanSearch];
    return (
      <BenchmarkSection head={page.premise} id="research-question" className="scroll-mt-24">
        <div className="grid border-t border-page-accent-border md:grid-cols-2 lg:grid-cols-[minmax(280px,0.82fr)_repeat(2,minmax(0,1fr))]">
          <div className="flex min-w-0 flex-col justify-between border-b border-r border-page-accent-border p-6 md:col-span-2 lg:col-span-1 lg:row-span-2 lg:p-8">
            <span className="text-xs font-semibold uppercase text-page-accent">Diagnostic premise</span>
            <p className="mt-12 max-w-[22ch] text-xl font-medium leading-relaxed text-muted lg:text-2xl">
              {page.premise.thesis}
            </p>
          </div>
          {page.premise.specs.map(([label, value]) => (
            <div className="min-w-0 border-b border-r border-page-accent-border p-6" key={label}>
              <span className="block text-xs font-semibold uppercase text-page-accent">{label}</span>
              <p className="mt-3 text-sm leading-relaxed text-muted">{value}</p>
            </div>
          ))}
        </div>

        <div className="border-b border-page-accent-border px-6 py-5 lg:px-8">
          <h3 className={ui.heading3}>{page.premise.calloutTitle}</h3>
        </div>
        <div className="grid md:grid-cols-3">
          {page.premise.calloutItems.map(([label, title, body], index) => {
            const Icon = contributionIcons[index];
            return (
              <article className="min-w-0 border-b border-r border-page-accent-border p-6 lg:p-8" key={label}>
                <div className="mb-8 flex items-center justify-between gap-4 text-page-accent">
                  <span className="text-xs font-semibold">[{label}]</span>
                  <Icon aria-hidden="true" size={20} strokeWidth={1.5} />
                </div>
                <h4 className={ui.heading3}>{title}</h4>
                <p className="mt-3 text-sm leading-relaxed text-muted">{body}</p>
              </article>
            );
          })}
        </div>
        <div className="grid border-b border-page-accent-border lg:grid-cols-[minmax(180px,0.32fr)_minmax(0,1fr)]">
          <div className="border-b border-page-accent-border px-6 py-5 lg:border-b-0 lg:border-r lg:px-8">
            <span className="text-xs font-semibold uppercase text-page-accent">What to take away</span>
          </div>
          <p className="px-6 py-5 text-base leading-relaxed text-foreground lg:px-8">{page.premise.note}</p>
        </div>
      </BenchmarkSection>
    );
  }

  if (page.id === "minds_eye") return <MindsEyePremiseSection page={page} />;
  return <SpatialPremiseSection page={page} />;
}

function TaskTableSection({ page, className }) {
  if (!page.taskSection) return null;
  return (
    <BenchmarkSection className={className} head={page.taskSection} bodyClassName="px-6 lg:px-8">
      <TaskTable table={page.taskSection.table} />
    </BenchmarkSection>
  );
}

function TaskTable({ table }) {
  return (
    <div className={ui.tableWrap}>
      <table className={`${ui.table} ${benchmarkTableClasses} ${taskTableClasses}`}>
        <thead><tr><th className="w-20">#</th>{table.headers.map((header) => <th key={header}>{header}</th>)}</tr></thead>
        <tbody>{table.rows.map((row, rowIndex) => <tr key={`${row[0]}-${rowIndex}`}><td className="w-20 text-xs font-semibold tabular-nums text-page-accent">[{String(rowIndex + 1).padStart(2, "0")}]</td>{row.map((cell, index) => <td key={`${row[0]}-${index}`}>{index === 0 ? <strong>{cell}</strong> : cell}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function SamplesSection({ page, className }) {
  if (!page.samples) return null;
  return <BenchmarkSection className={className} head={page.samples} padded><SampleGrid samples={page.samples} /></BenchmarkSection>;
}

function TaxonomySection({ page }) {
  if (!page.taxonomy) return null;
  return (
    <BenchmarkSection head={page.taxonomy}>
      <div className="grid border-l border-t border-border md:grid-cols-3">{page.taxonomy.cards.map(([badge, label, title, body]) => <div className="min-w-0 border-b border-r border-border bg-surface p-7" key={title}><span className="mb-5 inline-flex border border-page-accent-chip-border bg-page-accent-soft px-2.5 py-1 text-xs font-semibold text-page-accent">{badge}</span><span className="mb-2 block text-xs font-semibold uppercase text-page-accent">{label}</span><h3 className={ui.heading3}>{title}</h3><p className="mt-2 text-sm leading-relaxed text-muted">{body}</p></div>)}</div>
    </BenchmarkSection>
  );
}

function condKind(label) {
  const l = label.toLowerCase();
  if (l.includes("++") || l.includes("plus")) return "blank";
  if (l.includes("no-image") || l.includes("no image")) return "noimage";
  if (l.includes("cot") || l.includes("thought") || l.includes("chain")) return "cot";
  if (l.includes("judge")) return "judge";
  return "standard";
}

function ConditionGlyph({ kind }) {
  const s = { fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round", vectorEffect: "non-scaling-stroke" };
  const frame = <rect x="8" y="11" width="32" height="26" rx="2" {...s} />;
  const photo = <><circle cx="16" cy="19" r="2.4" {...s} /><path d="M9 33l10-10 7 7 5-5 8 8" {...s} /></>;
  if (kind === "judge") {
    return <svg viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="15" {...s} /><path d="M17 24l5 5 9-12" {...s} /></svg>;
  }
  if (kind === "blank") {
    return <svg viewBox="0 0 48 48" aria-hidden="true"><rect x="8" y="11" width="32" height="26" rx="2" {...s} strokeDasharray="4 3.5" /><text x="24" y="31" textAnchor="middle" fontFamily="inherit" fontSize="17" fontWeight="600" fill="currentColor" stroke="none">?</text></svg>;
  }
  if (kind === "noimage") {
    return <svg viewBox="0 0 48 48" aria-hidden="true">{frame}<line x1="12" y1="35" x2="36" y2="13" {...s} /></svg>;
  }
  if (kind === "cot") {
    return <svg viewBox="0 0 48 48" aria-hidden="true">{frame}{photo}<circle cx="30" cy="7" r="1.5" fill="currentColor" stroke="none" /><circle cx="35" cy="7" r="1.5" fill="currentColor" stroke="none" /><circle cx="40" cy="7" r="1.5" fill="currentColor" stroke="none" /></svg>;
  }
  return <svg viewBox="0 0 48 48" aria-hidden="true">{frame}{photo}</svg>;
}

function EvaluationSection({ page }) {
  if (!page.evaluation) return null;
  return (
    <BenchmarkSection head={page.evaluation}>
      <div className={cn(
        "grid border-l border-t border-border",
        page.id === "spatial" ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3" : "grid-cols-[repeat(auto-fit,minmax(150px,1fr))]",
      )}>
        {page.evaluation.steps.map(([label, body], index) => (
          <div className="min-w-0 border-b border-r border-border p-5" key={label}>
            <span className="mb-4 block text-xs font-semibold text-faint">[{String(index + 1).padStart(2, "0")}]</span>
            <div className="mb-4 size-12 text-page-accent [&_svg]:size-full"><ConditionGlyph kind={condKind(label)} /></div>
            <span className="mb-2 block font-display font-bold">{label}</span>
            <p className="m-0 text-sm leading-relaxed text-muted">{body}</p>
          </div>
        ))}
      </div>
    </BenchmarkSection>
  );
}

function BuildSection({ page, className }) {
  if (!page.pipeline) return null;
  return <BenchmarkSection className={className} head={page.pipeline}><Pipeline steps={page.pipeline.steps} accented /></BenchmarkSection>;
}

function GapMeter({ meter }) {
  if (!meter) return null;
  const gap = Math.round((meter.human - meter.model) * 10) / 10;
  const gapLabel = `${meter.gapPrefix || ""}${gap}`;
  const gapMidpoint = (meter.human + meter.model) / 2;
  return (
    <div className="mb-8 mt-1.5" style={{ "--human": `${meter.human}%`, "--model": `${meter.model}%`, "--gap-midpoint": `${gapMidpoint}%` }}>
      <div className="relative mb-4 h-28 sm:h-16">
        <div className="absolute bottom-0 left-[var(--model)] flex -translate-x-full flex-col items-end gap-1 border-r border-page-accent pr-2 text-right">
          <span className="text-xs font-semibold uppercase text-faint">{meter.modelLabel}</span>
          <strong className="font-display text-3xl font-bold leading-none tabular-nums max-sm:text-2xl">{meter.modelValue}</strong>
        </div>
        <div className="absolute left-[var(--human)] top-0 flex -translate-x-full flex-col items-end gap-1 border-r border-border-strong pr-2 text-right sm:bottom-0 sm:top-auto">
          <span className="text-xs font-semibold uppercase text-faint">{meter.humanLabel}</span>
          <strong className="font-display text-3xl font-bold leading-none tabular-nums max-sm:text-2xl">{meter.humanValue}</strong>
        </div>
      </div>
      <div className="relative h-4 overflow-hidden border border-border-strong bg-surface-subtle" role="img" aria-label={`${meter.modelLabel} ${meter.modelValue} versus ${meter.humanLabel} ${meter.humanValue}, a gap ${meter.gapPrefix ? "greater than " : "of "}${gap} points`}>
        <div className="absolute inset-y-0 left-0 w-[var(--model)] bg-foreground/25" />
        <div className="absolute inset-y-0 left-[var(--model)] w-[calc(var(--human)_-_var(--model))] bg-[repeating-linear-gradient(45deg,color-mix(in_srgb,var(--page-accent)_40%,transparent)_0_5px,transparent_5px_10px)]" />
        <div className="absolute inset-y-0 left-[var(--model)] w-0.5 bg-page-accent" />
        <div className="absolute inset-y-0 left-[var(--human)] w-0.5 bg-brand-strong" />
      </div>
      <div className="relative mt-2 h-4 text-xs tabular-nums text-faint"><span className="absolute left-0">0</span><span className="absolute left-[var(--gap-midpoint)] -translate-x-1/2 whitespace-nowrap text-center font-semibold text-page-accent">{gapLabel} point gap</span><span className="absolute right-0">100%</span></div>
      {meter.caption && <p className="mt-3.5 max-w-[64ch] text-sm leading-relaxed text-muted">{meter.caption}</p>}
    </div>
  );
}

function ResultsSection({ page, className }) {
  if (!page.results) return null;
  return <BenchmarkSection className={className} head={page.results} padded={Boolean(page.gapMeter)}><GapMeter meter={page.gapMeter} /><ResultStrip results={page.results} /></BenchmarkSection>;
}

function FindingsSection({ page, className }) {
  return <BenchmarkSection className={className} head={page.findings}><FindingGrid cards={page.findings.cards} /></BenchmarkSection>;
}

const FIG_POS = "var(--chart-positive)";
const FIG_NEG = "var(--chart-negative)";
const FIG_BASE = "var(--chart-primary)";

function AccuracyMatrix({ chart }) {
  return (
    <div className="mt-6 overflow-x-auto border border-border-strong" role="img" aria-label={`${chart.title}. Accuracy by object size and rotation angle.`}>
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr>
            <th className="border-b border-r border-border bg-surface-subtle px-4 py-3 text-left text-xs font-semibold uppercase text-faint" scope="col">Object size</th>
            {chart.columns.map((column) => (
              <th className="border-b border-r border-border bg-surface-subtle px-4 py-3 text-center text-xs font-semibold uppercase text-faint last:border-r-0" key={column} scope="col">{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {chart.rows.map((row) => (
            <tr key={row.label}>
              <th className="border-b border-r border-border px-4 py-4 text-left font-medium text-foreground last:border-b-0" scope="row">{row.label}</th>
              {row.values.map((value, index) => (
                <td
                  className={cn(
                    "border-b border-r border-border px-4 py-4 text-center font-semibold tabular-nums last:border-r-0",
                    value >= 100
                      ? "bg-page-accent text-white"
                      : value > 0
                        ? "bg-page-accent-soft text-page-accent"
                        : "bg-surface-subtle text-faint",
                  )}
                  key={`${row.label}-${chart.columns[index]}`}
                >
                  {value}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const effectLabels = {
  gain: { label: "Gain", className: "bg-positive-soft text-positive" },
  loss: { label: "Loss", className: "bg-negative-soft text-negative" },
  neutral: { label: "Near baseline", className: "bg-surface-subtle text-muted" },
};

function EffectMatrix({ chart }) {
  return (
    <div className="mt-6 overflow-x-auto border border-border-strong" role="img" aria-label={`${chart.title}. Direction of accuracy change relative to Chain of Thought.`}>
      <table className="w-full min-w-[620px] border-collapse text-sm">
        <thead>
          <tr>
            <th className="border-b border-r border-border bg-surface-subtle px-4 py-3 text-left text-xs font-semibold uppercase text-faint" scope="col">ART dimension</th>
            {chart.columns.map((column) => (
              <th className="border-b border-r border-border bg-surface-subtle px-4 py-3 text-center text-xs font-semibold uppercase text-faint last:border-r-0" key={column} scope="col">{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {chart.rows.map((row) => (
            <tr key={row.label}>
              <th className="border-b border-r border-border px-4 py-4 text-left font-medium text-foreground" scope="row">{row.label}</th>
              {row.effects.map((effect, index) => {
                const config = effectLabels[effect] || effectLabels.neutral;
                return (
                  <td className="border-b border-r border-border p-2 last:border-r-0" key={`${row.label}-${chart.columns[index]}`}>
                    <span className={cn("flex min-h-11 items-center justify-center px-3 py-2 text-center text-sm font-medium", config.className)}>{config.label}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FigureChart({ chart }) {
  const scale = chart.scale ?? 1;
  const suffix = chart.suffix ?? (chart.kind === "diverging" ? " pts" : "%");
  const aspect = chart.aspectRatio ?? (chart.wide ? "24 / 8" : "16 / 11");
  const digits = chart.digits ?? (chart.kind === "diverging" ? 1 : 0);
  if (chart.kind === "matrix") return <AccuracyMatrix chart={chart} />;
  if (chart.kind === "effectMatrix") return <EffectMatrix chart={chart} />;
  if (chart.kind === "grouped") {
    return (
      <BarChart
        aspectRatio={aspect}
        minHeight={chart.minHeight}
        bottomMargin={chart.bottomMargin ?? 80}
        categories={chart.categories.map((label) => ({ label }))}
        compactXLabels={chart.compactXLabels}
        forceHorizontalLabels={chart.forceHorizontalLabels}
        xLabelAngle={chart.xLabelAngle}
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
      minHeight={chart.minHeight}
      bottomMargin={chart.bottomMargin ?? 80}
      categories={chart.bars.map(([label]) => ({ label }))}
      compactXLabels={chart.compactXLabels}
      xLabelAngle={chart.xLabelAngle}
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

function FindingsFiguresSection({ page, className }) {
  if (!page.figures?.charts?.length) return null;
  const f = page.figures;
  return (
    <BenchmarkSection className={className} head={{ tag: f.tag, title: f.title, body: f.body }}>
      <div className="grid border-l border-t border-border md:grid-cols-2">
        {f.charts.map((chart, index) => (
          <div className={cn("min-w-0 border-b border-r border-border p-6 lg:p-8", chart.wide && "md:col-span-2")} key={chart.title || index}>
            <h3 className={ui.heading3}>{chart.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">{chart.caption}</p>
            <FigureChart chart={chart} />
            {chart.source && <p className="mt-3 text-xs text-faint">{chart.source}</p>}
          </div>
        ))}
      </div>
    </BenchmarkSection>
  );
}

function ModelPerformanceSection({ page, rows, className, id }) {
  const config = benchmarkChart[page.id];
  if (!config) return null;
  return (
    <BenchmarkSection
      className={className}
      id={id}
      head={{
        tag: "Leaderboard",
        title: "Model performance in this module",
        body: `How ranked models score on ${page.navLabel}, measured by ${config.metricLabel.toLowerCase()} per submitted model.`,
      }}
    >
      <div className="min-w-0 border border-border p-6 lg:p-8">
        <BenchmarkModelChart
          rows={rows}
          metricFor={(row) => config.metricFor?.(row) ?? row[config.metricKey]}
          metricLabel={config.metricLabel}
          color={config.color}
          emptyMessage="Model scores will appear here once submissions are ranked for this module."
        />
      </div>
    </BenchmarkSection>
  );
}

function SpatialDataSection({ datasets }) {
  return (
    <BenchmarkSection head={{ tag: "The data", title: "Thirteen spatial datasets, one policy", body: "Static 2D relations, 3D geometry, and dynamic or temporal understanding are unified under a single evaluation and scoring scheme. Datasets are downloaded from their official sources on your machine; none are redistributed here." }}>
      <div className={ui.tableWrap}><table className={`${ui.table} ${benchmarkTableClasses}`}><thead><tr><th>Dataset</th><th>Type</th><th className={ui.tableNumber}>Approx. n</th><th>Capabilities</th><th>License</th></tr></thead><tbody>{datasets.length ? datasets.map((dataset) => <tr key={dataset.name}><td><strong>{dataset.name}</strong></td><td>{dataset.type}</td><td className={ui.tableNumber}>{dataset.approx_n ? `~${dataset.approx_n}` : "N/A"}</td><td>{(dataset.tags || []).map((tag) => <span className="mr-1 inline-flex min-h-7 items-center border border-page-accent-chip-border bg-page-accent-chip px-2.5 py-1 text-xs text-muted" key={tag}>{tag}</span>)}</td><td>{dataset.license || "N/A"}</td></tr>) : <tr><td colSpan="5" className={ui.emptyRow}>Manifest unavailable.</td></tr>}</tbody></table></div>
    </BenchmarkSection>
  );
}

function BenchmarkContent({ page }) {
  const [datasets, setDatasets] = useState([]);
  const [modelRows, setModelRows] = useState([]);
  const [loadError, setLoadError] = useState("");
  useEffect(() => {
    if (page.id !== "spatial") return;
    getJSON("/api/tasks/spatial/info")
      .then((info) => setDatasets(info.datasets || []))
      .catch((error) => {
        setDatasets([]);
        setLoadError(errorMessage(error, "Spatial dataset details could not be loaded."));
      });
  }, [page.id]);
  useEffect(() => {
    const config = benchmarkChart[page.id];
    if (!config) { setModelRows([]); return; }
    setLoadError("");
    getJSON(config.endpoint)
      .then((data) => setModelRows(data.leaderboard || []))
      .catch((error) => {
        setModelRows([]);
        setLoadError(errorMessage(error, `Ranked models for ${page.navLabel} could not be loaded.`));
      });
  }, [page.id, page.navLabel]);
  useEffect(() => {
    window.dispatchEvent(new CustomEvent("app-warning", {
      detail: loadError ? { message: loadError, tone: "negative", action: () => window.location.reload(), actionLabel: "Retry" } : null,
    }));
  }, [loadError]);

  return (
    <>
      <PageHero {...page} />
      <PaperProjectOverview page={page} />
      <StatBand stats={page.stats} accented />
      <PremiseSection page={page} />
      <div className="scroll-mt-24" id="tasks">
        <TaxonomySection page={page} />
        <TaskTableSection page={page} />
        {page.id === "spatial" && <SpatialDataSection datasets={datasets} />}
        <SamplesSection page={page} />
      </div>
      <div className="scroll-mt-24" id="method">
        <EvaluationSection page={page} />
        <BuildSection page={page} />
      </div>
      <div className="scroll-mt-24" id="results">
        <ResultsSection page={page} />
        <FindingsSection page={page} />
        <FindingsFiguresSection page={page} />
      </div>
      <ModelPerformanceSection className="scroll-mt-24" id="live-results" page={page} rows={modelRows} />
      <div className="scroll-mt-24" id="submission">
        <ScoringBlock scoring={page.scoring} cardBottomBorders={page.id === "dysm" || page.id === "minds_eye"} />
      </div>
      <Citation citation={page.citation} />
    </>
  );
}

export function Benchmark() {
  const { slug } = useParams();
  const page = benchmarkPages[keyBySlug[slug]];
  return page ? <BenchmarkContent page={page} /> : <NotFound />;
}
