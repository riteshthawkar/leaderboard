import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export function HomeHero() {
  return (
    <section className="hero">
      <div className="container">
        <span className="eyebrow">Microsoft Research · Multimodal Evaluation</span>
        <h1>
          Do multimodal LLMs <span className="grad">truly see</span> what they reason about?
        </h1>
        <p className="hero-sub">
          Three MSR benchmarks. One leaderboard. Visual <strong>perception</strong>, mental <strong>imagery</strong>, spatial <strong>reasoning</strong> — abilities humans master effortlessly that current models still fail.
        </p>
        <div className="hero-cta">
          <Button asChild variant="brand"><Link to="/leaderboard">View the leaderboard</Link></Button>
          <Button asChild><Link to="/submit">Submit a model</Link></Button>
          <Button asChild variant="ghost"><a href="#findings">Key findings ↓</a></Button>
        </div>
      </div>
    </section>
  );
}

export function PageHero({ eyebrow, title, subtitle, cta, paperUrl, authors, accent, showMeta = false }) {
  return (
    <section className="page-hero bench-accent">
      <div className="container">
        <span className="eyebrow" style={accent ? { color: accent } : undefined}>{eyebrow}</span>
        <h1>{title}</h1>
        {subtitle && <p className="hero-sub">{subtitle}</p>}
        {cta && (
          <div className="hero-cta">
            <Button asChild variant="brand"><Link to="/leaderboard">{cta}</Link></Button>
            <Button asChild><Link to="/submit">Submit a model</Link></Button>
            {paperUrl && <Button asChild variant="ghost"><a href={paperUrl} target="_blank" rel="noreferrer">Read paper ↗</a></Button>}
          </div>
        )}
        {showMeta && (
          <div className="hero-meta">
            <span className="authors">{authors}</span>
            <span className="sep">·</span>
            <span className="affil">Microsoft Research</span>
          </div>
        )}
      </div>
    </section>
  );
}