import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { HeroArt } from "@/components/HeroArt";
import { cn } from "@/lib/utils";

export function HomeHero() {
  return (
    <section className="relative isolate flex h-[var(--hero-section-h)] items-center overflow-hidden bg-black text-[#f5f7f8] max-[700px]:h-auto max-[700px]:min-h-[var(--hero-section-h)] max-[700px]:overflow-visible max-[700px]:py-20">
      <div className="home-hero-art" aria-hidden="true" />
      <div className="container relative z-10">
        <div className="w-full max-w-[550px]">
          <span className="mb-5 inline-flex items-center text-xs font-medium uppercase text-[#d3d9dd]">Microsoft Research · Multimodal Evaluation</span>
          <h1 className="max-w-[18ch] font-display text-4xl font-bold leading-[1.06] text-[#f5f7f8] sm:text-5xl lg:text-[3.55rem]">
            <span>MS VISTA:</span> Do multimodal LLMs <span>truly see</span> what they reason about?
          </h1>
          <p className="mt-3 max-w-[62ch] text-base leading-relaxed text-[#d3d9dd] sm:text-lg">
            MS VISTA is a benchmark suite and leaderboard for the visual intelligence of multimodal LLMs. It evaluates visual <strong>perception</strong>, <strong>visual cognition</strong>, and spatial <strong>reasoning</strong> under one reproducible protocol faithful to each paper.
          </p>
          <div className="mt-8 flex flex-wrap gap-3 max-sm:w-full max-sm:[&>*]:w-full">
            <Button asChild variant="brand"><Link to="/leaderboard">View the leaderboard</Link></Button>
            <Button asChild><Link to="/submit">Submit a model</Link></Button>
          </div>
        </div>
      </div>
    </section>
  );
}

export function PageHero({ eyebrow, title, subtitle, cta, paperUrl, authors, showMeta = false, id }) {
  const isBenchmark = id === "dysm" || id === "minds_eye" || id === "spatial";
  return (
    <section className={cn(
      "relative isolate flex items-stretch overflow-hidden bg-background max-[700px]:h-auto max-[700px]:overflow-visible",
      "h-[var(--hero-section-h)] max-[700px]:min-h-[var(--hero-section-h)]",
      isBenchmark && "benchmark-page-hero [&_.page-hero-art]:!text-page-accent",
      )}>
      {isBenchmark && <HeroArt variant={id} />}
      <div className="container relative z-10 flex flex-col justify-center max-[700px]:py-20">
        <div className="w-full max-w-[550px]">
          <span className={cn("mb-5 inline-flex items-center text-xs font-medium uppercase text-faint", isBenchmark && "!text-page-accent")}>{eyebrow}</span>
          <h1 className={cn("mb-0 max-w-[680px] font-display text-[2.55rem] font-bold leading-[1.04] max-sm:text-[2.05rem]", isBenchmark && "text-4xl leading-[1.05] sm:text-5xl lg:text-[4rem]")}>{title}</h1>
          {subtitle && <p className="mt-3 max-w-[62ch] text-base leading-relaxed text-muted sm:text-lg">{subtitle}</p>}
          {cta && (
            <div className="mt-6 flex flex-wrap gap-3 max-sm:w-full max-sm:[&>*]:w-full">
              <Button asChild variant="brand"><Link to="/leaderboard">{cta}</Link></Button>
              <Button asChild className={cn(isBenchmark && "!border-page-accent-border !text-page-accent-muted")}><Link to="/submit">Submit a model</Link></Button>
              {paperUrl && (
                <Button asChild variant="ghost" className={cn(isBenchmark && "!border-page-accent-border !text-page-accent-muted")}>
                  <a href={paperUrl} target="_blank" rel="noreferrer">Read paper <ArrowUpRight className="size-3.5" strokeWidth={1.75} aria-hidden="true" /></a>
                </Button>
              )}
            </div>
          )}
          {showMeta && (
            <div className="mt-5 flex flex-wrap gap-x-4 gap-y-1.5 text-sm text-muted">
              <span className="font-semibold text-foreground">{authors}</span>
              <span className="text-faint max-sm:hidden">·</span>
              <span className="text-xs uppercase text-faint">Microsoft Research</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
