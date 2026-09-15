import { Link } from "react-router";
import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { HomeHero } from "@/components/Hero";
import { snapshots } from "@/data/snapshot";
import { getJSON } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ui } from "@/lib/styles";

const visualTaskSnapshotPaths = [
  "/api/tasks/do_you_see_me/info",
  "/api/tasks/minds_eye/info",
];
const snapshotVisualReleaseItems = visualTaskSnapshotPaths.reduce(
  (sum, path) =>
    sum +
    (Number.isFinite(snapshots[path]?.total_samples)
      ? snapshots[path].total_samples
      : 0),
  0,
);
const snapshotRankedModels = snapshots["/api/statistics/overview"]?.ranked_models;

const frameworkLayers = [
  {
    to: "/benchmarks/do-you-see-me",
    n: "01",
    art: "rings",
    name: "Visual Perception",
    layer: "Evaluation layer",
    body: "Measures whether a model can detect and organize visual evidence across seven perceptual skills and controlled 2D and 3D difficulty. Do You See Me supplies this framework module.",
    meta: ["Do You See Me module", "7 skills", "4,500 release"],
    linkLabel: "Explore the perception module",
    theme: {
      art: "!border-border !bg-[color-mix(in_srgb,var(--dysm)_5%,var(--surface))] !text-dysm group-hover:!border-border-strong",
      label: "!text-dysm",
      meta: "!border-border !bg-dysm-soft",
      arrow: "!text-dysm",
    },
  },
  {
    to: "/benchmarks/minds-eye",
    n: "02",
    art: "cube",
    name: "Visual Cognition",
    layer: "Evaluation layer",
    body: "Measures abstraction, mental transformation, composition, and fluid visual reasoning across eight diagnostic task families. Mind's Eye supplies this framework module.",
    meta: ["Mind's Eye module", "8 task families", "799 release"],
    linkLabel: "Explore the cognition module",
    theme: {
      art: "!border-border !bg-[color-mix(in_srgb,var(--me)_5%,var(--surface))] !text-me group-hover:!border-border-strong",
      label: "!text-me",
      meta: "!border-border !bg-me-soft",
      arrow: "!text-me",
    },
  },
  {
    to: "/benchmarks/spatial",
    n: "03",
    art: "perspective",
    name: "Reasoning Analysis",
    layer: "Evaluation layer",
    body: "Analyzes how chain-of-thought and visual-evidence interventions change model behavior across 13 spatial datasets and six controlled conditions. Spatial Reasoning & Robustness supplies this framework module.",
    meta: ["Spatial module", "13 datasets", "6 conditions"],
    linkLabel: "Explore the reasoning module",
    theme: {
      art: "!border-border !bg-[color-mix(in_srgb,var(--spatial)_5%,var(--surface))] !text-spatial group-hover:!border-border-strong",
      label: "!text-spatial",
      meta: "!border-border !bg-spatial-soft",
      arrow: "!text-spatial",
    },
  },
];

const frameworkFindings = [
  {
    score: { primary: "95.8%", operator: "Vs.", secondary: "<50%" },
    title: "Humans see; models don't",
    body: "Humans hit 95.8%; the best MLLMs average below 50%. The gap widens sharply with difficulty.",
    source: "Do You See Me",
    paperUrl: "https://arxiv.org/abs/2506.02022",
    domain: "Visual Perception",
    accent: "text-dysm",
  },
  {
    score: { primary: "29%" },
    title: "Right answer, wrong reasons",
    body: "29% of correct reasoning answers still hid fundamental perception errors. Final accuracy is misleading.",
    source: "Do You See Me",
    paperUrl: "https://arxiv.org/abs/2506.02022",
    domain: "Visual Perception",
    accent: "text-dysm",
  },
  {
    score: { primary: "23.2", operator: "→", secondary: "41.8" },
    title: "MCQ shortcuts inflate scores",
    body: "MCQ reformulation nearly doubled accuracy (23 → 42%). Models exploit answer options, not the image.",
    source: "Do You See Me",
    paperUrl: "https://arxiv.org/abs/2506.02022",
    domain: "Visual Perception",
    accent: "text-dysm",
  },
  {
    score: { primary: "3%", qualifier: "Lower\u202FAvg." },
    title: "Chain of Thought degrades vision",
    body: "CoT lowers spatial accuracy by about 3% on average and by as much as 23% for some reasoning models.",
    source: "CoT degrades spatial reasoning",
    paperUrl: "https://arxiv.org/abs/2604.16060",
    domain: "Spatial Reasoning",
    accent: "text-spatial",
  },
  {
    score: { primary: "7", operator: "/", secondary: "8" },
    title: "Reasoning models lose to backbones",
    body: "7 of 8 reasoning models failed to beat the backbone they were distilled from on spatial benchmarks.",
    source: "CoT degrades spatial reasoning",
    paperUrl: "https://arxiv.org/abs/2604.16060",
    domain: "Spatial Reasoning",
    accent: "text-spatial",
  },
  {
    score: { primary: "80%", operator: "Vs.", secondary: "<50%" },
    title: "Visual cognition trails humans most",
    body: "On Mind's Eye, humans average 80% while top models stay below 50%, with the biggest deficits on mental transformation tasks.",
    source: "Mind's Eye",
    paperUrl: "https://arxiv.org/abs/2604.16054",
    domain: "Visual Cognition",
    accent: "text-me",
  },
];

const evaluationSteps = [
  {
    phase: "Identity & runs",
    title: "Pin one model profile",
    body: "Register one canonical model identity and record its revision, prompting, decoding, and run provenance before evaluating any module.",
  },
  {
    phase: "Capability modules",
    title: "Generate framework outputs",
    body: "Run the perception and cognition modules, then execute the six controlled reasoning-analysis conditions under the same declared model profile.",
  },
  {
    phase: "Evidence contract",
    title: "Validate complete evidence",
    body: "Normalize outputs into module-specific contracts and verify identifiers, sample coverage, conditions, hashes, counts, and provenance.",
  },
  {
    phase: "Integrated report",
    title: "Build the capability profile",
    body: "Publish perception, cognition, VPCI, and reasoning diagnostics together while retaining the scientifically valid scoring and verification method for each module.",
  },
];

const verificationLevels = [
  {
    scope: "Capability evaluation",
    title: "Ground-truth scored",
    body: "The perception and cognition modules are scored by the service against private ground truth.",
  },
  {
    scope: "Reasoning analysis",
    title: "Controlled interventions",
    body: "The reasoning module compares behavior across six image-and-prompt conditions; artifact integrity, coverage, and score arithmetic are verified.",
  },
  {
    scope: "Framework record",
    title: "Unified evidence",
    body: "Every result shares a stable model identity and run provenance. Visual answers remain private; accepted reasoning artifacts are publicly auditable.",
  },
];

const faqs = [
  {
    question: "How does MS VISTA create one profile across evaluation modules?",
    answer:
      "Register the model once, then attach every module run to that canonical identity. MS VISTA presents perception, cognition, VPCI, and controlled reasoning diagnostics together as one capability profile whenever the corresponding results are available.",
  },
  {
    question: "What file should I upload for each evaluation module?",
    answer:
      "The perception and cognition modules each accept one UTF-8 JSONL file. The reasoning-analysis module accepts track3_artifact_submission.zip from the official harness. It contains manifest.json, claimed_scores.json, answers.jsonl.gz, raw_outputs.jsonl.gz, and checksums.json; upload the ZIP unchanged rather than its members.",
  },
  {
    question: "Do submissions need reasoning text or only final answers?",
    answer:
      "Only final answers are required for the visual capability modules. Each released question ID must appear exactly once. The reasoning-analysis harness records final outputs for every required condition and adds provenance in the run manifest; free form reasoning traces are not required.",
  },
  {
    question: "What does validation check before scoring?",
    answer:
      "For the visual capability modules, validation checks the file format and complete sample coverage before deterministic scoring. For reasoning analysis, it verifies the official harness version, package hashes, provenance, public sample coverage, scoring groups, and agreement between per sample correctness flags and aggregate scores. Spatial answers are not independently graded again by the server.",
  },
  {
    question: "How are leaderboard scores and rankings calculated?",
    answer:
      "Do You See Me uses a dimension-balanced task macro. Mind's Eye uses an unweighted mean across its eight tasks. VPCI is the equally weighted mean of those two visual capability scores and is shown only when both are available. Reasoning diagnostics are reported alongside that profile but do not enter VPCI because they measure intervention effects rather than the same accuracy construct. Gap compares the visual scores, while task spread summarizes variation across tasks and is better when lower.",
  },
  {
    question: "Which models appear in each framework view?",
    answer:
      "Each evaluation view includes every model with a result for that module. The combined visual view keeps all visual models visible, but VPCI is available only for models with both perception and cognition results. The integrated comparison joins available module results under the same model identity.",
  },
  {
    question: "How often can I submit, and can I delete a result?",
    answer:
      "Each verified account has one accepted submission per evaluation module in a rolling 24 hour window. The modules have independent limits. You can delete your own result from Submission history, but deletion does not restore a consumed quota slot because the audit record must remain intact.",
  },
  {
    question: "Are private answers or submitted outputs exposed?",
    answer:
      "Private ground truth is not published. Perception and cognition response exports remain available only to the account owner and administrators. The entire Track 3 evidence ZIP is public, including final answers, model outputs that may contain reasoning text, scores, provenance, and hashes. Never include personal data, credentials, or confidential material in a public submission.",
  },
];

const overviewMotionTransition = {
  duration: 0.6,
  ease: [0.22, 1, 0.36, 1],
};

function MindEyeOverviewMotif({ active = false }) {
  return (
    <svg
      viewBox="0 0 144 130"
      aria-hidden="true"
      data-overview-motif="cube"
    >
      <g fill="currentColor" stroke="none">
        <path d="M86 3L126 19L92 47L52 31Z" fillOpacity="0.08" />
        <path d="M86 3L52 31L54 75L88 47Z" fillOpacity="0.035" />
        <path d="M126 19L92 47L94 91L128 63Z" fillOpacity="0.055" />
      </g>

      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="0.8"
        strokeLinejoin="miter"
      >
        <path d="M86 3L126 19L92 47L52 31Z" />
        <path d="M86 3L88 47M126 19L128 63M92 47L94 91M52 31L54 75" />
        <path d="M54 75L88 47L128 63L94 91" opacity="0.45" />
      </g>

      <g
        className={cn(
          "[transform-box:view-box] [transform-origin:51.388889%_63.846154%] [transform-style:preserve-3d] transition-transform duration-[1100ms] ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none",
          active &&
            "[transform:rotate3d(0.928477,0.371391,0,180deg)]",
        )}
      >
        <path
          d="M54 75L94 91L62.206897 121.482759L22.206897 105.482759Z"
          fill="currentColor"
          fillOpacity="0.1"
        />
        <path
          d="M54 75L94 91L62.206897 121.482759L22.206897 105.482759Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="0.8"
          strokeLinejoin="miter"
          vectorEffect="non-scaling-stroke"
        />
      </g>
    </svg>
  );
}

function PerspectiveOverviewMotif({ active = false }) {
  const reduceMotion = useReducedMotion();
  const transition = reduceMotion
    ? { duration: 0 }
    : overviewMotionTransition;
  const middle = active
    ? { x: 29.5, y: 36, width: 64, height: 49 }
    : { x: 31, y: 37, width: 61, height: 47 };
  const inner = active
    ? { x: 43, y: 46, width: 40, height: 30 }
    : { x: 46, y: 48, width: 34, height: 26 };
  const connectorOpacity = active ? 0.7 : 0.4;
  const shapeStyle = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    vectorEffect: "non-scaling-stroke",
  };
  const connectors = [
    [16, 26, inner.x, inner.y],
    [104, 26, inner.x + inner.width, inner.y],
    [16, 94, inner.x, inner.y + inner.height],
    [104, 94, inner.x + inner.width, inner.y + inner.height],
  ];

  return (
    <svg
      viewBox="0 0 120 120"
      aria-hidden="true"
      data-overview-motif="perspective"
    >
      <rect x="16" y="26" width="88" height="68" {...shapeStyle} />
      <motion.rect
        x="31"
        y="37"
        width="61"
        height="47"
        animate={{
          attrX: middle.x,
          attrY: middle.y,
          width: middle.width,
          height: middle.height,
        }}
        transition={transition}
        {...shapeStyle}
        opacity="0.82"
      />
      <motion.rect
        x="46"
        y="48"
        width="34"
        height="26"
        animate={{
          attrX: inner.x,
          attrY: inner.y,
          width: inner.width,
          height: inner.height,
        }}
        transition={transition}
        {...shapeStyle}
        opacity="0.7"
      />
      {connectors.map(([x1, y1, x2, y2], index) => (
        <motion.line
          key={`${x1}-${y1}`}
          x1={x1}
          y1={y1}
          x2={index % 2 === 0 ? 46 : 80}
          y2={index < 2 ? 48 : 74}
          animate={{ x2, y2, opacity: connectorOpacity }}
          transition={transition}
          {...shapeStyle}
        />
      ))}
    </svg>
  );
}

function OverviewMotif({ kind, active = false }) {
  const s = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    vectorEffect: "non-scaling-stroke",
  };
  if (kind === "rings") {
    return (
      <svg
        viewBox="0 0 120 120"
        aria-hidden="true"
        data-overview-motif="rings"
      >
        <g
          className={cn(
            "transition-transform duration-[600ms] ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none",
            active && "translate-x-[6px]",
          )}
        >
          {[40, 31, 22, 13].map((r) => (
            <circle key={`a${r}`} cx="48" cy="60" r={r} {...s} />
          ))}
        </g>
        <g
          className={cn(
            "transition-transform duration-[600ms] ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none",
            active && "-translate-x-[6px]",
          )}
        >
          {[40, 31, 22, 13].map((r) => (
            <circle key={`b${r}`} cx="72" cy="60" r={r} {...s} opacity="0.5" />
          ))}
        </g>
      </svg>
    );
  }
  if (kind === "cube") {
    return <MindEyeOverviewMotif active={active} />;
  }
  return <PerspectiveOverviewMotif active={active} />;
}

function FrameworkLayerRow({ layer, index }) {
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const active = hovered || focused;

  return (
    <Link
      to={layer.to}
      className="group grid min-w-0 md:min-h-[380px] md:grid-cols-2"
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
    >
      <div
        className={cn(
          "grid h-[300px] min-w-0 w-full place-items-center overflow-hidden border-b border-border bg-[linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)] bg-[size:40px_40px] [&_svg]:block [&_svg]:h-[calc(100%-3rem)] [&_svg]:max-h-full [&_svg]:max-w-full [&_svg]:w-[calc(100%-3rem)] md:h-full md:min-h-[380px] md:border-b-0 md:border-r",
          layer.theme.art,
          index % 2 === 1 && "md:order-2 md:border-l md:border-r-0",
        )}
        aria-hidden="true"
      >
        <OverviewMotif kind={layer.art} active={active} />
      </div>
      <div className="flex min-w-0 max-w-[60ch] flex-col justify-center px-6 py-10 lg:px-8">
        <span
          className={`mb-3 block text-xs font-semibold uppercase ${layer.theme.label}`}
        >
          [{layer.n}] {layer.layer}
        </span>
        <h3 className="mb-3 font-display text-3xl font-bold">
          {layer.name}
        </h3>
        <p className="mb-4 text-sm leading-relaxed text-muted">
          {layer.body}
        </p>
        <div className="mb-5 flex flex-wrap gap-2">
          {layer.meta.map((item) => (
            <span className={cn(ui.badge, layer.theme.meta)} key={item}>
              {item}
            </span>
          ))}
        </div>
        <span
          className={`mt-3 inline-flex text-sm font-medium ${layer.theme.arrow}`}
        >
          {layer.linkLabel} →
        </span>
      </div>
    </Link>
  );
}

export function Home() {
  const [questions, setQuestions] = useState(() =>
    snapshotVisualReleaseItems > 0
      ? snapshotVisualReleaseItems.toLocaleString()
      : "Pending",
  );
  const [models, setModels] = useState(() =>
    Number.isInteger(snapshotRankedModels) && snapshotRankedModels >= 0
      ? snapshotRankedModels.toLocaleString()
      : "Pending",
  );
  useEffect(() => {
    Promise.allSettled([
      getJSON("/api/statistics/overview"),
      ...["do_you_see_me", "minds_eye"].map((id) =>
        getJSON(`/api/tasks/${id}/info`),
      ),
    ]).then((results) => {
      const stats = results[0].status === "fulfilled" ? results[0].value : {};
      const visualInfoResults = results.slice(1);
      const allVisualInfoLoaded = visualInfoResults.every(
        (result) => result.status === "fulfilled",
      );
      const total = visualInfoResults.reduce(
        (sum, result) =>
          sum +
          (result.status === "fulfilled" &&
          Number.isFinite(result.value?.total_samples)
            ? result.value.total_samples
            : 0),
        0,
      );
      setQuestions(
        allVisualInfoLoaded && total > 0
          ? total.toLocaleString()
          : snapshotVisualReleaseItems > 0
            ? snapshotVisualReleaseItems.toLocaleString()
            : "Pending",
      );
      setModels(
        Number.isInteger(stats.ranked_models) && stats.ranked_models >= 0
          ? stats.ranked_models.toLocaleString()
          : Number.isInteger(snapshotRankedModels) && snapshotRankedModels >= 0
            ? snapshotRankedModels.toLocaleString()
            : "Pending",
      );
    });
  }, []);

  return (
    <>
      <HomeHero />
      <section className="mt-6 bg-background" aria-label="Overview statistics">
        <div className="container !px-0">
          <div className="grid grid-cols-1 border-l border-t border-border sm:grid-cols-2 lg:grid-cols-3">
            {[
              [
                "01",
                "3",
                "Evaluation layers spanning perception, cognition, and reasoning analysis",
              ],
              [
                "02",
                questions,
                "Scored items across the framework's visual capability modules",
              ],
              [
                "03",
                "13",
                "Datasets used for controlled reasoning analysis",
              ],
              ["04", models, "Unique models profiled by MS VISTA"],
              [
                "05",
                "95.8%",
                "Human perception macro accuracy in the paper study",
              ],
              [
                "06",
                "80%",
                "Human visual cognition accuracy in the paper study",
              ],
            ].map(([number, value, label]) => (
              <div
                className="flex min-w-0 flex-col border-b border-r border-border bg-transparent p-6"
                key={number}
              >
                <span className="mb-6 text-xs font-semibold text-faint">
                  [{number}]
                </span>
                <span className="font-display text-4xl font-bold tabular-nums">
                  {value}
                </span>
                <span className="mt-2 text-sm leading-relaxed text-muted">
                  {label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="overview">
        <div className="container !px-0">
          <div className={ui.sectionBand}>
            <div className="max-w-copy">
              <div className={ui.sectionTag}>Framework architecture</div>
              <h2 className={ui.heading2}>
                One framework. Three evaluation layers.
              </h2>
              <p className={cn(ui.lede, "mt-4")}>
                MS VISTA evaluates visual intelligence as a connected capability
                profile: what a model perceives, what it can infer and transform
                visually, and how reasoning strategies change that behavior.
                Published research suites supply the evaluation modules; shared
                model identity, provenance, evidence, diagnostics, and reporting
                make them one framework.
              </p>
            </div>
          </div>
          <div className="flex flex-col divide-y divide-border-strong">
            {frameworkLayers.map((layer, index) => (
              <FrameworkLayerRow
                layer={layer}
                index={index}
                key={layer.name}
              />
            ))}
          </div>
        </div>
      </section>

      <section id="findings">
        <div className={ui.sectionFrame}>
          <div className={ui.sectionBand}>
            <div className="max-w-copy">
              <div className={ui.sectionTag}>Evidence across framework layers</div>
              <h2 className={ui.heading2}>What MS VISTA reveals</h2>
              <p className={cn(ui.lede, "mt-4")}>
                Connecting perception, cognition, and controlled reasoning
                diagnostics separates capability failures from reasoning-induced
                artifacts. Each statistic remains linked to its source study;
                MS VISTA does not collapse them into one pooled accuracy.
              </p>
            </div>
          </div>
          <div className="divide-y divide-border-strong">
            {frameworkFindings.map((finding, index) => (
              <article
                className="grid min-w-0 grid-cols-[56px_minmax(0,1fr)] lg:grid-cols-[72px_minmax(0,1fr)]"
                key={finding.title}
              >
                <div
                  className={cn(
                    "flex items-center justify-center border-r border-border-strong px-2 py-6 text-xs font-semibold tabular-nums lg:py-8",
                    finding.accent,
                  )}
                >
                  [{String(index + 1).padStart(2, "0")}]
                </div>
                <div className="grid min-w-0 sm:grid-cols-[minmax(260px,0.7fr)_minmax(0,1.3fr)] lg:grid-cols-[minmax(260px,0.7fr)_minmax(0,1.3fr)_220px]">
                  <div className="flex min-w-0 items-center border-b border-border-strong px-5 py-6 sm:border-r lg:border-b-0 lg:px-7 lg:py-8">
                    <span className="inline-flex min-w-0 items-center whitespace-nowrap tabular-nums">
                      <span className="font-display text-3xl font-bold">
                        {finding.score.primary}
                      </span>
                      {finding.score.operator && (
                        <span className="mx-2.5 shrink-0 font-sans text-3xl font-normal text-muted">
                          {finding.score.operator}
                        </span>
                      )}
                      {finding.score.secondary && (
                        <span className="font-display text-3xl font-bold">
                          {finding.score.secondary}
                        </span>
                      )}
                      {finding.score.qualifier && (
                        <span className="ml-2.5 font-sans text-3xl font-normal text-muted">
                          {finding.score.qualifier}
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="min-w-0 border-b border-border-strong px-5 py-6 lg:border-b-0 lg:border-r lg:px-7 lg:py-8">
                    <h3 className="font-display text-xl font-bold">
                      {finding.title}
                    </h3>
                    <p className="mt-3 text-sm leading-relaxed text-muted">
                      {finding.body}
                    </p>
                  </div>
                  <div className="flex min-w-0 items-center justify-between gap-4 px-5 py-4 sm:col-span-2 lg:col-span-1 lg:flex-col lg:items-start lg:justify-center lg:px-6 lg:py-8">
                    <span className="text-xs font-semibold uppercase text-faint">
                      {finding.domain}
                    </span>
                    <a
                      className={cn(
                        "text-right text-sm font-semibold underline decoration-border-strong underline-offset-4 hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand lg:text-left",
                        finding.accent,
                      )}
                      href={finding.paperUrl}
                      target="_blank"
                      rel="noreferrer noopener"
                      aria-label={`${finding.source} source paper`}
                    >
                      {finding.source}
                    </a>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="methodology">
        <div className={ui.sectionFrame}>
          <div className={ui.sectionBand}>
            <div className="max-w-copy">
              <div className={ui.sectionTag}>Unified evaluation pipeline</div>
              <h2 className={ui.heading2}>One framework, scientifically valid scoring</h2>
              <p className={cn(ui.lede, "mt-4")}>
                MS VISTA unifies model identity, run provenance, output
                contracts, evidence retention, diagnostics, and reporting. Each
                evaluation layer keeps the scoring method required by its source
                task, so unification does not erase scientific differences.
              </p>
            </div>
          </div>

          <dl className="grid border-b border-border-strong md:grid-cols-3">
            {verificationLevels.map((level, index) => (
              <div
                className={cn(
                  "min-w-0 px-6 py-7 lg:px-8",
                  index < verificationLevels.length - 1 &&
                    "border-b border-border-strong md:border-b-0 md:border-r",
                )}
                key={level.title}
              >
                <dt className="text-xs font-semibold uppercase text-faint">
                  {level.scope}
                </dt>
                <dd className="mt-2">
                  <div className="font-display text-lg font-bold">
                    {level.title}
                  </div>
                  <p className="mt-2 text-sm leading-relaxed text-muted">
                    {level.body}
                  </p>
                </dd>
              </div>
            ))}
          </dl>

          <ol className="grid list-none p-0 md:grid-cols-2">
            {evaluationSteps.map((step, index) => (
              <li
                className={cn(
                  "flex min-h-[220px] min-w-0 flex-col justify-center px-6 py-9 lg:min-h-[260px] lg:px-8 lg:py-11",
                  index < evaluationSteps.length - 1 &&
                    "border-b border-border-strong",
                  index === 2 && "md:border-b-0",
                  index % 2 === 0 && "md:border-r md:border-border-strong",
                )}
                key={step.title}
              >
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm font-medium text-muted">
                    {step.phase}
                  </span>
                  <span className="text-xs font-medium tabular-nums text-faint">
                    Step {String(index + 1).padStart(2, "0")}
                  </span>
                </div>
                <h3 className="mt-3 font-display text-2xl font-bold">
                  {step.title}
                </h3>
                <p className="mt-4 max-w-[58ch] text-base leading-relaxed text-muted">
                  {step.body}
                </p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section id="faq">
        <div className={ui.sectionFrame}>
          <div className={ui.sectionBand}>
            <div className="max-w-copy">
              <div className={ui.sectionTag}>Support</div>
              <h2 className={ui.heading2}>Frequently Asked Questions</h2>
              <p className={cn(ui.lede, "mt-4")}>
                Answers about model registration, upload formats, validation,
                scoring, and leaderboard visibility.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-[clamp(1.25rem,6vw,5rem)_minmax(0,1fr)_clamp(1.25rem,6vw,5rem)]">
            <div aria-hidden="true" className="border-r border-border-strong" />
            <div className="min-w-0 pb-20 lg:pb-24">
              <div className="divide-y divide-border-strong border-b border-border-strong">
                {faqs.map((faq) => (
                  <details className="group" key={faq.question}>
                    <summary className="flex min-h-20 cursor-pointer list-none items-center justify-between gap-6 px-6 py-5 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-foreground lg:px-8 [&::-webkit-details-marker]:hidden">
                      <span className="font-sans text-base font-medium">
                        {faq.question}
                      </span>
                      <ChevronDown
                        className="shrink-0 text-muted transition-transform duration-200 group-open:rotate-180"
                        size={20}
                        strokeWidth={1.5}
                        aria-hidden="true"
                      />
                    </summary>
                    <div className="border-t border-border px-6 py-6 lg:px-8 lg:py-7">
                      <p className="max-w-[72ch] text-base leading-relaxed text-muted">
                        {faq.answer}
                      </p>
                    </div>
                  </details>
                ))}
              </div>
            </div>
            <div aria-hidden="true" className="border-l border-border-strong" />
          </div>
        </div>
      </section>
    </>
  );
}
