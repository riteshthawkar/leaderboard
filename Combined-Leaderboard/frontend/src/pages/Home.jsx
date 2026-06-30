import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { HomeHero } from "@/components/Hero";
import { Pipeline } from "@/components/Sections";
import { Card } from "@/components/ui/card";
import { getJSON } from "@/lib/api";

export function Home() {
  const [questions, setQuestions] = useState("—");
  const [models, setModels] = useState("—");
  useEffect(() => {
    Promise.all([
      getJSON("/api/statistics/overview").catch(() => ({})),
      Promise.all(["do_you_see_me", "minds_eye", "spatial"].map((id) => getJSON(`/api/tasks/${id}/info`).catch(() => ({})))),
    ]).then(([stats, infos]) => {
      const total = infos.reduce((sum, info) => sum + (info.total_samples || 0), 0);
      setQuestions(total ? total.toLocaleString() : "2,612+");
      setModels(Math.max(stats.visual_cognition_models || 0, stats.spatial_models || 0) || "0");
    });
  }, []);

  return (
    <>
      <HomeHero />
      <section className="stats-section"><div className="container"><div className="stat-band">
        <div className="stat-cell"><span className="n">[01]</span><span className="v">3</span><span className="l">Benchmarks unified under one scoring protocol</span></div>
        <div className="stat-cell"><span className="n">[02]</span><span className="v">{questions}</span><span className="l">Evaluation questions across all three tasks</span></div>
        <div className="stat-cell"><span className="n">[03]</span><span className="v">{models}</span><span className="l">Multimodal LLMs ranked on the leaderboard</span></div>
        <div className="stat-cell"><span className="n">[04]</span><span className="v">29</span><span className="l">Distinct task variants and diagnostic conditions</span></div>
        <div className="stat-cell"><span className="n">[05]</span><span className="v">&lt;50%</span><span className="l">Best MLLM accuracy — far below human level</span></div>
        <div className="stat-cell"><span className="n">[06]</span><span className="v">96%</span><span className="l">Human accuracy on the same perception tasks</span></div>
      </div></div></section>

      <section className="section" id="overview"><div className="container"><div className="section-head"><div className="section-tag">Overview</div><h2>One leaderboard. Three benchmarks.</h2><p className="lede">MLLMs reason well but often misread the image. Three complementary benchmarks isolate <em>where</em> visual understanding breaks down — one consistent, reproducible protocol.</p></div>
        <div className="grid cols-3 ruled">
          <Link to="/benchmarks/do-you-see-me" className="card bench-card"><span className="card-n">[01]</span><h3>Do-You-See-Me</h3><p className="muted">7 perceptual skills in 2D &amp; 3D — shape, color, figure-ground, closure, spatial. 2,612 questions with parametric difficulty.</p><span className="arrow">Explore benchmark →</span></Link>
          <Link to="/benchmarks/minds-eye" className="card bench-card"><span className="card-n">[02]</span><h3>Mind's-Eye</h3><p className="muted">8 visuo-cognitive tasks — mental rotation, paper folding, composition — probing fluid intelligence beyond surface perception.</p><span className="arrow">Explore benchmark →</span></Link>
          <Link to="/benchmarks/spatial" className="card bench-card"><span className="card-n">[03]</span><h3>Spatial &amp; CoT Robustness</h3><p className="muted">13 spatial datasets, one policy. CoT, shortcut, and hallucination diagnostics expose how reasoning shortcuts distort spatial scores.</p><span className="arrow">Explore benchmark →</span></Link>
        </div></div></section>

      <section className="section alt" id="findings"><div className="container"><div className="section-head"><div className="section-tag">What the benchmarks reveal</div><h2>Six consistent failures</h2><p>Models reason well but see poorly. Techniques that boost text reasoning often hurt visual tasks.</p></div>
        <div className="grid cols-3 ruled">
          {[
            ["96.5% vs <50%", "Humans see; models don't", "Humans hit 96.5%; the best MLLMs average below 50%. The gap widens sharply with difficulty.", "Do You See Me"],
            ["29%", "Right answer, wrong reasons", "29% of correct reasoning answers still hid fundamental perception errors. Final accuracy is misleading.", "Do You See Me"],
            ["23.2 → 41.8", "MCQ shortcuts inflate scores", "MCQ reformulation nearly doubled accuracy (23 → 42%). Models exploit answer options, not the image.", "Do You See Me"],
            ["−3% avg", "Chain-of-Thought degrades vision", "CoT lowers spatial accuracy by ~3% on average, up to −23% for some reasoning models.", "CoT Degrades Spatial · Do You See Me"],
            ["7 / 8", "Reasoning models lose to backbones", "7 of 8 reasoning models failed to beat the backbone they were distilled from on spatial benchmarks.", "CoT Degrades Spatial"],
            ["80% vs <50%", "Imagery & transformation are hardest", "Humans average 80%; top models stay below 50%, with the biggest deficits on Transformation tasks.", "Mind's Eye"],
          ].map(([stat, title, body, src]) => <Card className="finding" key={title}><div className="stat-line">{stat}</div><h3>{title}</h3><p>{body}</p><div className="src">{src}</div></Card>)}
        </div></div></section>

      <section className="section" id="methodology"><div className="container"><div className="section-head"><div className="section-tag">How it works</div><h2>How scoring works</h2><p>One GPT-4o judge. Paper-faithful scoring. Three benchmarks unified into comparable rankings.</p></div>
        <Pipeline steps={[["Generate responses", "Run your model on released questions. For Spatial, generate CoT and no-image conditions."], ["Submit predictions", "Upload a JSON or CSV file per task. Evaluation is fully offline."], ["Judge & extract", "GPT-4o extracts answers for perception and imagery, and judges correctness for spatial tasks."], ["Score & diagnose", "Accuracy, macro averages, random baselines, and spatial robustness diagnostics are computed per benchmark."], ["Rank", "Per-benchmark tables feed a combined Visual-Cognition Index (VCI)."]]} />
        <div className="grid cols-3 ruled"><Card className="callout"><span className="card-n">[A]</span><h4>Do-You-See-Me</h4><p>GPT-4o answer <strong>extractor</strong> → exact / numeric match. Score = macro average of 2D and 3D accuracy.</p></Card><Card className="callout"><span className="card-n">[B]</span><h4>Mind's-Eye</h4><p>GPT-4o answer <strong>extractor</strong> → MCQ-label match, reported against a random-choice baseline.</p></Card><Card className="callout"><span className="card-n">[C]</span><h4>Spatial</h4><p>GPT-4o <strong>as judge</strong> on restricted MCQ options. pass@1, greedy decoding, averaged over seeds.</p></Card></div>
      </div></section>
    </>
  );
}