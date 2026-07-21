import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { HomeHero } from "@/components/Hero";
import { getJSON } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ui } from "@/lib/styles";

const overviewBenchmarks = [
  {
    to: "/benchmarks/do-you-see-me",
    n: "01",
    art: "rings",
    name: "Do You See Me",
    layer: "Perception",
    body: "7 perceptual skills in 2D and 3D: shape, color, figure ground, closure, and spatial relations. Drawn from a benchmark containing 2,612 questions with parametric difficulty.",
    meta: ["7 skills", "2D & 3D", "2,612 questions"],
    theme: {
      art: "!border-[color-mix(in_srgb,var(--dysm)_28%,var(--border))] !bg-[color-mix(in_srgb,var(--dysm)_5%,var(--surface))] !text-dysm group-hover:!border-dysm",
      label: "!text-dysm",
      meta: "!border-[color-mix(in_srgb,var(--dysm)_25%,var(--border))] !bg-dysm-soft",
      arrow: "!text-dysm",
    },
  },
  {
    to: "/benchmarks/minds-eye",
    n: "02",
    art: "cube",
    name: "Mind's Eye",
    layer: "Visual Cognition",
    body: "8 visual cognition tasks covering mental rotation, paper folding, and composition while probing fluid intelligence beyond surface perception.",
    meta: ["8 tasks", "Rotation · Folding", "Fluid reasoning"],
    theme: {
      art: "!border-[color-mix(in_srgb,var(--me)_28%,var(--border))] !bg-[color-mix(in_srgb,var(--me)_5%,var(--surface))] !text-me group-hover:!border-me",
      label: "!text-me",
      meta: "!border-[color-mix(in_srgb,var(--me)_25%,var(--border))] !bg-me-soft",
      arrow: "!text-me",
    },
  },
  {
    to: "/benchmarks/spatial",
    n: "03",
    art: "perspective",
    name: "Spatial & CoT Robustness",
    layer: "Spatial Reasoning",
    body: "13 spatial datasets, one policy. CoT, shortcut, and hallucination diagnostics expose how reasoning shortcuts distort spatial scores.",
    meta: ["13 datasets", "CoT diagnostics", "4 conditions"],
    theme: {
      art: "!border-[color-mix(in_srgb,var(--spatial)_28%,var(--border))] !bg-[color-mix(in_srgb,var(--spatial)_5%,var(--surface))] !text-spatial group-hover:!border-spatial",
      label: "!text-spatial",
      meta: "!border-[color-mix(in_srgb,var(--spatial)_25%,var(--border))] !bg-spatial-soft",
      arrow: "!text-spatial",
    },
  },
];

const benchmarkFindings = [
  {
    score: { primary: "95.8%", operator: "Vs.", secondary: "<50%" },
    title: "Humans see; models don't",
    body: "Humans hit 95.8%; the best MLLMs average below 50%. The gap widens sharply with difficulty.",
    source: "Do You See Me",
    domain: "Visual Perception",
    accent: "text-dysm",
  },
  {
    score: { primary: "29%" },
    title: "Right answer, wrong reasons",
    body: "29% of correct reasoning answers still hid fundamental perception errors. Final accuracy is misleading.",
    source: "Do You See Me",
    domain: "Visual Perception",
    accent: "text-dysm",
  },
  {
    score: { primary: "23.2", operator: "→", secondary: "41.8" },
    title: "MCQ shortcuts inflate scores",
    body: "MCQ reformulation nearly doubled accuracy (23 → 42%). Models exploit answer options, not the image.",
    source: "Do You See Me",
    domain: "Visual Perception",
    accent: "text-dysm",
  },
  {
    score: { primary: "3%", qualifier: "Lower\u202FAvg." },
    title: "Chain of Thought degrades vision",
    body: "CoT lowers spatial accuracy by about 3% on average and by as much as 23% for some reasoning models.",
    source: "CoT degrades spatial reasoning",
    domain: "Spatial Reasoning",
    accent: "text-spatial",
  },
  {
    score: { primary: "7", operator: "/", secondary: "8" },
    title: "Reasoning models lose to backbones",
    body: "7 of 8 reasoning models failed to beat the backbone they were distilled from on spatial benchmarks.",
    source: "CoT degrades spatial reasoning",
    domain: "Spatial Reasoning",
    accent: "text-spatial",
  },
  {
    score: { primary: "80%", operator: "Vs.", secondary: "<50%" },
    title: "Visual cognition trails humans most",
    body: "On Mind's Eye, humans average 80% while top models stay below 50%, with the biggest deficits on mental transformation tasks.",
    source: "Mind's Eye",
    domain: "Visual Cognition",
    accent: "text-me",
  },
];

const evaluationSteps = [
  {
    phase: "Input",
    title: "Generate responses",
    body: "Run the model on released questions. For Spatial, use the harness to generate all six required evaluation conditions.",
  },
  {
    phase: "Package",
    title: "Submit predictions",
    body: "Upload one JSONL file for each visual benchmark and one ZIP package for Spatial. Evaluation remains fully offline and isolated from private answers.",
  },
  {
    phase: "Validation",
    title: "Validate coverage",
    body: "Every released question ID must appear exactly once for each required condition.",
  },
  {
    phase: "Evaluation",
    title: "Score and publish",
    body: "Compute accuracy, macro averages, random baselines, and spatial diagnostics, then publish benchmark tables and the combined Visual Perception and Cognition Index.",
  },
];

const faqs = [
  {
    question: "How do I submit the same model to multiple benchmarks?",
    answer:
      "Register the model once in the submission workspace, then select that same model identity for each benchmark upload. Scores from Do You See Me, Mind's Eye, and Spatial are attached to one model record and shown together wherever the required results are available.",
  },
  {
    question: "What file should I upload for each benchmark?",
    answer:
      "Do You See Me and Mind's Eye each accept one UTF 8 JSONL file. Spatial accepts one ZIP package produced by the official harness. That package contains submission.jsonl, run_manifest.json, and leaderboard.json, so those files should not be uploaded separately.",
  },
  {
    question: "Do submissions need reasoning text or only final answers?",
    answer:
      "Only final answers are required for the visual benchmarks. Each released question ID must appear exactly once. The Spatial harness records final outputs for every required condition and adds provenance in the run manifest; free form reasoning traces are not required.",
  },
  {
    question: "What does validation check before scoring?",
    answer:
      "For visual benchmarks, validation checks the file format and complete sample coverage before deterministic scoring. For Spatial, it verifies the official harness version, package hashes, provenance, public sample coverage, scoring groups, and agreement between per sample correctness flags and aggregate scores. Spatial answers are not independently graded again by the server.",
  },
  {
    question: "How are leaderboard scores and rankings calculated?",
    answer:
      "Do You See Me uses a dimension balanced task macro. Mind's Eye uses an unweighted mean across its eight tasks. VPCI is the equally weighted mean of the perception and cognition scores and is shown only when both are available. Gap compares those two scores, while task spread summarizes variation across tasks and is better when lower.",
  },
  {
    question: "Which models appear when I select a benchmark tab?",
    answer:
      "A benchmark tab includes every model with a score for that benchmark, including models that also have results for other benchmarks. The combined visual view keeps all visual models visible, but VPCI is available only for models with both perception and cognition results.",
  },
  {
    question: "How often can I submit, and can I delete a result?",
    answer:
      "Each verified account has one accepted submission per benchmark in a rolling 24 hour window. The three benchmarks have separate limits. You can delete your own result from Submission history, but deletion does not restore a consumed quota slot because the audit record must remain intact.",
  },
  {
    question: "Are private answers or submitted outputs exposed?",
    answer:
      "Private ground truth never leaves the evaluation service. Visual benchmark response exports remain available only to the account owner and administrators. Spatial final answer evidence, its aggregate report, manifest, hashes, and original ZIP are public so anyone can audit a published spatial score. Free form reasoning traces are not required or published.",
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

function OverviewBenchmarkRow({ benchmark, index }) {
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const active = hovered || focused;

  return (
    <Link
      to={benchmark.to}
      className="group grid min-w-0 md:min-h-[380px] md:grid-cols-2"
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
    >
      <div
        className={cn(
          "grid h-[300px] min-w-0 w-full place-items-center overflow-hidden border-b border-border bg-[linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)] bg-[size:40px_40px] [&_svg]:block [&_svg]:h-[calc(100%-3rem)] [&_svg]:max-h-full [&_svg]:max-w-full [&_svg]:w-[calc(100%-3rem)] md:h-full md:min-h-[380px] md:border-b-0 md:border-r",
          benchmark.theme.art,
          index % 2 === 1 && "md:order-2 md:border-l md:border-r-0",
        )}
        aria-hidden="true"
      >
        <OverviewMotif kind={benchmark.art} active={active} />
      </div>
      <div className="flex min-w-0 max-w-[60ch] flex-col justify-center px-6 py-10 lg:px-8">
        <span
          className={`mb-3 block text-xs font-semibold uppercase ${benchmark.theme.label}`}
        >
          [{benchmark.n}] {benchmark.layer}
        </span>
        <h3 className="mb-3 font-display text-3xl font-bold">
          {benchmark.name}
        </h3>
        <p className="mb-4 text-sm leading-relaxed text-muted">
          {benchmark.body}
        </p>
        <div className="mb-5 flex flex-wrap gap-2">
          {benchmark.meta.map((item) => (
            <span className={cn(ui.badge, benchmark.theme.meta)} key={item}>
              {item}
            </span>
          ))}
        </div>
        <span
          className={`mt-3 inline-flex text-sm font-medium ${benchmark.theme.arrow}`}
        >
          Explore benchmark →
        </span>
      </div>
    </Link>
  );
}

export function Home() {
  const [questions, setQuestions] = useState("Loading");
  const [models, setModels] = useState("Loading");
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
          : "Unavailable",
      );
      setModels(
        Number.isInteger(stats.ranked_models) && stats.ranked_models >= 0
          ? stats.ranked_models.toLocaleString()
          : "Unavailable",
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
                "Evaluation tracks covering perception, cognition, and spatial robustness",
              ],
              [
                "02",
                questions,
                "Scored items across the two visual leaderboard tracks",
              ],
              [
                "03",
                "13",
                "Spatial datasets evaluated under controlled conditions",
              ],
              ["04", models, "Unique models currently ranked across all tracks"],
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
              <div className={ui.sectionTag}>Overview</div>
              <h2 className={ui.heading2}>
                One leaderboard. Three benchmarks.
              </h2>
              <p className={cn(ui.lede, "mt-4")}>
                MS VISTA runs three complementary benchmarks that probe
                perception, visual cognition, and spatial reasoning. Together
                they pinpoint <em>where</em> and <em>how</em> each model breaks
                down under one reproducible protocol.
              </p>
            </div>
          </div>
          <div className="flex flex-col divide-y divide-border-strong">
            {overviewBenchmarks.map((benchmark, index) => (
              <OverviewBenchmarkRow
                benchmark={benchmark}
                index={index}
                key={benchmark.name}
              />
            ))}
          </div>
        </div>
      </section>

      <section id="findings">
        <div className={ui.sectionFrame}>
          <div className={ui.sectionBand}>
            <div className="max-w-copy">
              <div className={ui.sectionTag}>What the benchmarks reveal</div>
              <h2 className={ui.heading2}>Six consistent failures</h2>
              <p className={cn(ui.lede, "mt-4")}>
                One pattern recurs across all three benchmarks: techniques that
                boost text reasoning often <em>hurt</em> visual accuracy, while
                strong headline scores can hide broken perception.
              </p>
            </div>
          </div>
          <div className="divide-y divide-border-strong">
            {benchmarkFindings.map((finding, index) => (
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
                    <span
                      className={cn(
                        "text-right text-sm font-semibold lg:text-left",
                        finding.accent,
                      )}
                    >
                      {finding.source}
                    </span>
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
              <div className={ui.sectionTag}>How it works</div>
              <h2 className={ui.heading2}>How scoring works</h2>
              <p className={cn(ui.lede, "mt-4")}>
                JSONL submissions containing final answers. Matching against
                private ground truth. Three benchmarks unified into comparable
                rankings.
              </p>
            </div>
          </div>

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
