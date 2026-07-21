# Spatial Reasoning Evaluation Harness

A **download-and-evaluate** toolkit that reproduces the protocol of **"Chain-of-Thought Degrades
Visual Spatial Reasoning Capabilities of Multimodal LLMs"** (Kancheti et al.) on the paper's
**13 spatial benchmarks**. We can't redistribute the datasets — this repo instead fetches every
one of them from its original source with a single command, then runs the paper's exact
evaluation. The official leaderboard contract always runs six conditions:

1. **Main accuracy — CoT vs Non-CoT** on real images (the paper's headline: CoT usually *hurts* spatial reasoning).
2. **No-Image** with CoT and non-CoT prompts on the frozen 100-item-per-dataset ablation set.
3. **No-Image++** with CoT and non-CoT prompts on a **blank gray image** with a *"Cannot determine from the
   image"* option appended as the correct answer. A grounded model abstains; a model that
   hallucinates from text priors keeps "seeing" things.

Scoring uses the paper's **LLM-as-judge** (Qwen3-30B-A3B-Instruct-2507, verbatim scoring prompt,
κ>0.99 vs GPT-4o in the paper). The harness is **engine-agnostic**: inference and judging both talk
to any **OpenAI-compatible endpoint** — serve the models however you like.

---

## Quickstart — what to run, and when

```bash
# 0) DEPS   [run ONCE]  (data prep + eval client; deliberately NOT the full VLMEvalKit — see KNOWN_ISSUES A-1)
pip install -r requirements.txt

# 1) DATA   [run ONCE, ~30-60 min]  fetch + build all 13 benchmarks (~9 GB) into ./LMUData
#    SpatialBench is gated: accept the terms at hf.co/datasets/RussRobin/SpatialBench first, then pass a
#    HF token that can read gated repos (or add `--skip SpatialBench`).
python prepare_data.py --lmudata ./LMUData --hf-token hf_xxx

# 2) SERVE  [ONCE PER SESSION]  your VL model + the judge, on ANY OpenAI-compatible server
#    (engine-agnostic — vLLM / SGLang / TGI / ...; see serve/serve_example.sh)
#    vllm serve <YOUR_MODEL>                      --port 8000 --max-model-len 40960 --limit-mm-per-prompt '{"image": 8}'
#    vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --served-model-name judge --port 8100 --tensor-parallel-size 2

# 3) MANIFEST [ONCE PER RELEASE] download spatial_manifest.json from the leaderboard
mv /path/to/spatial_manifest.json ./benchmark_manifest.json

# 4) EVALUATE [ONCE PER MODEL] run all six conditions and judge final answers
./run_eval.sh <PUBLIC_MODEL_NAME> http://localhost:8000/v1 http://localhost:8100/v1
# Upload results/<PUBLIC_MODEL_NAME>/spatial_reasoning_submission.zip.
```

`run_eval.sh` chains `run_track3_vllm.py` and `judge_track3.py`. You may tune throughput and
token-budget flags. Do not override datasets, modes, prompt modes, prompts, or the ablation
manifest for a leaderboard submission; the official manifest pins all of them.

The public model name and served API identifier are separate. By default they are the same. Set
`SPATIAL_ENDPOINT_MODEL` when the endpoint uses a deployment alias:

```bash
SPATIAL_ENDPOINT_MODEL=deployments/qwen-vl \
SPATIAL_MODEL_API_KEY=... \
SPATIAL_JUDGE_API_KEY=... \
./run_eval.sh "Qwen VL" https://model.example/v1 https://judge.example/v1
```

`SPATIAL_JUDGE_ENDPOINT_MODEL` changes the served judge alias. Request timeouts default to 120
seconds and can be changed with `SPATIAL_MODEL_TIMEOUT_SECONDS` and
`SPATIAL_JUDGE_TIMEOUT_SECONDS`. The OpenAI SDK's internal retries are disabled; `--max-retries`
is the single controlled retry policy used by each harness stage.

Output artifacts:

| file | purpose |
|---|---|
| `spatial_reasoning_submission.zip` | **The single upload file.** Contains all three artifacts below |
| `submission.jsonl` | Public per-sample final answer, dataset, scoring group, correctness, and judge metadata for every required condition |
| `run_manifest.json` | Public and served model names, data, prompt, judge, hash, count, and zero-error provenance |
| `leaderboard.json` | Per-dataset aggregate report checked against the public sample evidence |

The backend reads the ZIP package in memory without extracting it. It validates provenance, complete
public sample coverage, and agreement between per-sample correctness flags and aggregate scores. It
does not independently compare answers with private ground truth. Accepted packages and their three
members are retained in SQLite and made public for visible leaderboard submissions. Raw chain-of-thought
is never included in the package.
Starting a rerun clears the harness-generated files in that output directory so a failed run cannot
leave an older upload package behind. Use a new output directory when you need to retain an earlier run.
Multiple GPUs? Serve one replica per GPU (ports 8000,8001,…) and pass all URLs to `--endpoints`
(comma-separated; the client round-robins).

---

## What's faithful to the paper — and what to know

**Faithful:** question format (`Question:…\nOptions:\nA.…\nPlease select the correct answer (letter
and option text) from the options above.`); system prompts (bare non-CoT; CoT = base +
`<think>/<answer>` instruction); dataset-specific prompts for OmniSpatial & MindCube (paper Appx
A.2); **pure greedy** decoding (temperature 0, no repetition penalty); **circular evaluation** for
SpatialBench & SAT-Real — the two datasets the paper stars (Table 5): each is re-asked with options
rotated to every position and counts correct only if right in *all* rotations; No-Image++ mechanism
(same-size gray image + appended "Cannot determine" option as ground truth); judge model + verbatim
scoring prompt (Appx A.3). The CoT answer budget defaults to 16384 tokens — non-truncating for real
traces (they hit EOS far sooner); raise `--max-tokens-cot` toward the paper's 32768 only if you also
serve a context larger than 32768.

**Documented differences** (see `KNOWN_ISSUES.md`):
- All benchmarks are standardized to **text-MCQ** (required for a uniform No-Image++). That drops
  SpatialBench's 20 numeric-counting items (+2 with broken answer keys → **152/174**) and
  OmniSpatial's 229 visual-selection items with no text options (→ **1304/1533**).
- **3DSRBench version drift**: the current public VLMEvalKit TSV is a circular/flip-*expanded*
  release (11,686 rows = 3,997 base questions). We score the **base questions non-circular** to
  match the paper's non-circular treatment, but cannot reproduce their exact ~5.2K set.
- Single greedy run (the paper averages 3 seeds; greedy is deterministic, so 1 run ≡ 3).

## The 13 datasets

| Paper name | Key | Group | Source path |
|---|---|---|---|
| BLINK | `BLINK` | 2D | hosted TSV (md5-verified) |
| CV-Bench 2D / 3D | `CV-Bench-2D` / `CV-Bench-3D` | 2D/3D | hosted TSV |
| MMVP | `MMVP` | 2D | hosted TSV |
| RealWorldQA | `RealWorldQA` | 2D | hosted TSV |
| V*Bench | `VStarBench` | 2D | hosted TSV |
| MMSI-Bench | `MMSIBench_wo_circular` | 3D | hosted TSV (non-circular, 1K = paper size) |
| 3DSRBench | `3DSRBench` | 3D | hosted TSV (md5-verified) |
| VSR | `VSR_MCQ` | 2D | hosted Yes/No TSV → recast A.Yes/B.No |
| SpatialBench | `SpatialBench` | 2D | original HF repo (**gated**) via loader |
| MindCube | `MindCube` | 3D | original HF repo via loader (multi-image) |
| OmniSpatial | `OmniSpatial` | 3D | original HF repo via loader |
| SAT-Real | `SAT-Real` | 3D | original HF repo via loader (`SAT_test.parquet`, 150 real) |

`prepare_data.py` handles all of it (downloads, format conversions, loaders, verification). The
loaders live in `loaders/` and normalize every dataset to one MCQ TSV schema:
`index | image(b64, JSON list if multi-image) | question | A..H | answer`.
Production conversion is strict: malformed records stop the build, and the known intentional
text-MCQ exclusions must match the documented counts exactly. A changed count indicates upstream
dataset drift and must be reviewed before issuing a new benchmark contract.

## Fixed ablation set

The paper defines the ablation mechanisms but publishes no fixed leaderboard question list. For a
reproducible leaderboard this repo ships one frozen set used by both No-Image and No-Image++:
- `ablation_manifest.json` — the exact per-dataset indices (seeded sample, n=100/dataset).
- `noimgpp_frozen_probe.json` — a content **sha1 per selected question** + md5 per base TSV, so any
  future run can verify it is evaluating the *identical* questions (and detect upstream drift).

Custom exploratory runs may use other subsets, but they cannot produce accepted leaderboard files.

## ⚠️ Why your numbers may differ from the paper

This harness reproduces the paper's **protocol** and its **finding** (CoT usually *hurts* spatial
reasoning). But **absolute accuracy** depends on many things outside the evaluation code — treat the
*relative* signal (CoT vs non-CoT on an identical setup) as the reliable result, and expect absolute
numbers to move for any of the reasons below.

1. **Inference stack & hardware — the biggest and least-obvious factor.** The *same model weights*
   can give materially different outputs on different GPUs/drivers, or a different engine or version
   (e.g. vLLM 0.10 vs 0.11, vLLM vs 🤗transformers, CUDA vs other backends). Vision encoders are
   especially sensitive: a model can *describe* an image correctly yet miss *fine* detail (counting,
   small objects) when low-level kernels differ numerically — shifting a dataset by tens of points
   with no error raised. **If one model is far off but another is on-target, suspect this first** and
   re-run the off model on a different engine/GPU to check. In particular the **attention
   implementation** matters: in our testing, forcing eager attention (`attn_implementation="eager"`
   in 🤗transformers, or the equivalent backend flag on your server) recovered ~20 points on a
   vision model whose fused/SDPA attention kernel was numerically degrading its vision tower — worth
   trying whenever a vision-heavy model underperforms for no clear reason.
2. **Third-party dataset drift.** The 13 benchmarks are hosted by their original authors and change
   over time (rows, images, splits). Fill every pending revision in `data_versions.json` before the
   production release; `prepare_data.py`
   md5-verifies hosted TSVs where upstream publishes hashes and sanity-checks row counts; and
   `noimgpp_frozen_probe.json` catches drift on the ablation set. Example: 3DSRBench's public TSV is
   now a circular/flip-expanded release (11,686 rows = 3,997 base Qs) vs the paper's ~5.2K.
3. **Text-MCQ standardization.** A uniform No-Image++ needs text options, so non-text-MCQ items are
   dropped: SpatialBench **152/174**, OmniSpatial **1304/1533** — different denominators than the paper.
4. **Decoding settings.** We pin **pure greedy** (temperature 0, no repetition penalty). Using a
   model's own `generation_config` (e.g. `repetition_penalty 1.05`) or any sampler changes the tokens
   produced — "greedy" is not a single well-defined thing across servers.
5. **Image preprocessing.** The resolution the model actually sees (`min_pixels`/`max_pixels`, tiling)
   changes fine-grained accuracy. Match your server's image settings to whatever you compare against.
6. **CoT token budget.** Too small truncates the trace before the answer; `max_tokens` ≥ context makes
   strict servers reject the request — both silently drop items (see KNOWN_ISSUES C-10).
7. **Seeds.** The paper averages 3 seeds; greedy is deterministic so one run ≡ three — but any sampling
   reintroduces variance.
8. **The LLM judge.** Scoring uses an LLM (Qwen3-30B-A3B, greedy); a different judge model/version or
   its serving stack can map borderline free-form answers differently (the paper validated κ>0.99 vs GPT-4o).
9. **Prompt fidelity.** Any change to the question format, system prompts, or the OmniSpatial/MindCube
   dataset prompts shifts results. This repo matches the paper's — keep them for comparability.

**For a fair comparison,** hold the model, data version, decoding, image preprocessing, and judge
identical; otherwise compare CoT-vs-non-CoT *within* your own run rather than against the paper's absolutes.

## Repo map

```
run_eval.sh                # official six-condition inference + judge workflow
prepare_data.py            # fetch and normalize all 13 benchmarks
run_track3_vllm.py         # inference client (OpenAI-compatible; multi-image + circular-eval aware)
judge_track3.py            # pinned judge -> submission, run manifest, and local report
build_server_bundle.py     # administrator-only official manifest, template, and private answer key
spatial_contract.py        # shared IDs, datasets, conditions, schemas, and hashes
serve/serve_example.sh     # reference serve commands (engine-agnostic)
prompts/                   # paper system prompts (non-CoT / CoT)
loaders/                   # per-dataset converters (SpatialBench/MindCube/OmniSpatial/SAT-Real/VSR)
ablation_manifest.json     # frozen No-Image and No-Image++ indices
noimgpp_frozen_probe.json  # content hashes for the frozen set (drift detection)
KNOWN_ISSUES.md            # every gotcha we hit, with fixes — read before modifying
```

## Citation

Kancheti, Kanade, Balasubramanian, Ganu. *Chain-of-Thought Degrades Visual Spatial Reasoning
Capabilities of Multimodal LLMs.* arXiv:2604.16060.
