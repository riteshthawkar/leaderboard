# MS-VISTA Research Experiments

This package extends the combined leaderboard from descriptive ranking to
controlled experiments. It does not modify canonical submissions, benchmark
ground truth, or production scores.

## Experiment map

| Experiment | Main question | GPU required | Primary endpoint |
| --- | --- | ---: | --- |
| Model complementarity | Do models fail on the same items, and does task routing generalize? | No | Out-of-fold routed macro minus out-of-fold global-model macro |
| Causal transformation controls | Does a model react to an answer-changing visual operator while ignoring nuisance styling? | Yes | Full-triplet causal grounding rate |
| Visual fidelity | Which capabilities fail as visual information is removed? | Yes | Paired macro change from native to each fidelity level |
| Abstraction cues | Is failure reduced by a correct abstract operation cue, and worsened by a mismatched cue? | Yes | Correct-cue and mismatched-cue deltas from baseline |
| Prompt by thinking factorial | Is Qwen3.5 sensitive to the benchmark reasoning prompt, internal thinking mode, or their interaction? | Yes | Five paired factorial contrasts with one Holm correction |
| Perception-reasoning module swap | Is an error inherited from the visual state producer or the reasoner? | Yes | Crossed perception-model by reasoner-model matrix |
| Perception-focused transfer | Does matched perception supervision transfer beyond generic recognition supervision? | Yes | Curriculum minus recognition-control benchmark change |

## Recommended controlled inference queue

The production research queue uses one free 40 GB A100 without interfering
with the two-GPU Track 3 run:

```bash
GPU_ID=2 \
ANALYSIS_ROOT=/share/data/drive_3/omkar/ms-vista-analysis-v2 \
DATASET_ROOT="$PWD/evaluation/results/.cache/visual-intelligence-dataset" \
QWEN_TARGET_HF_HOME=/share/data/drive_3/hf_cache \
INTERNVL_TARGET_HF_HOME=/share/data/drive_1/omkar/ms-vista-track3-v2/internvl-hf-home \
EXTRACTOR_HF_HOME="$PWD/evaluation/results/.cache/models/qwen3-8b-extractor" \
TARGET_SLUGS=qwen3-vl-8b,internvl35-8b \
  bash evaluation/research/run_single_gpu_analysis_queue.sh
```

It runs Qwen3-VL-8B-Instruct and InternVL3.5-8B sequentially in unquantized
BF16. Each model receives the same deterministic, paired conditions:

- 200 causal triplets under direct-answer and CoT prompts
- native, 50%, and 25% visual-fidelity conditions on stratified subsets of both
  visual benchmarks
- baseline, correct abstraction cue, and mismatched abstraction cue conditions
  on the eligible Mind's Eye tasks

The model server remains loaded while all raw image inference runs. A pinned
Qwen3-8B server is loaded once afterward to make gold-blind answer-extraction
decisions over every saved response. This avoids spending target-model GPU time
on temporary extraction. Model requests omit `max_tokens`, so vLLM uses the
remaining part of the fixed 32,768-token context. Length stops and unresolved
answers are model outcomes and count as wrong; inference errors, extractor
failures, coverage drift, or provenance mismatches fail the quality gate.

The remote host does not receive the private benchmark answer keys. It scores
the generated causal experiment there, but fidelity and abstraction outputs are
downloaded and scored on the trusted local host with
`analysis/research/score_analysis_queue.py`. That scorer applies paired
bootstrap intervals, benchmark-weighted paired label-swap tests, exact McNemar
tests for equally weighted causal triplets, and subgroup Holm correction.
Cross-model interactions use the same paired label-swap design. One global Holm
correction covers every prespecified within-model and cross-model primary
endpoint.

After that queue releases GPU 2, the scheduled Qwen3.5 follow-up runs all 799
Mind's Eye items in a paired 2 by 2 factorial design:

```bash
setsid env \
  GPU_ID=2 \
  CURRENT_ANALYSIS_ROOT=/share/data/drive_3/omkar/ms-vista-analysis-v2 \
  ABLATION_ROOT=/share/data/drive_3/omkar/ms-vista-qwen35-thinking-v1 \
  DATASET_ROOT="$PWD/evaluation/results/.cache/visual-intelligence-dataset" \
  TARGET_HF_HOME=/share/data/drive_3/hf_cache \
  EXTRACTOR_HF_HOME="$PWD/evaluation/results/.cache/models/qwen3-8b-extractor" \
  bash evaluation/research/queue_qwen35_followup.sh \
  >/share/data/drive_3/omkar/ms-vista-qwen35-thinking-v1/logs/bootstrap.log \
  2>&1 </dev/null &
```

The four conditions independently vary the direct versus reasoning prompt and
Qwen3.5's disabled versus enabled internal-thinking mode. This resolves the
protocol ambiguity in the existing leaderboard comparison without adding the
result to the frozen 21-test controlled family. Its separate five-test family
is frozen in `PRESPECIFIED_QWEN35_THINKING_PLAN.md`.

## CPU experiments already run

The following commands have been run locally with seed `20260723`:

```bash
.venv/bin/python analysis/research/offline_complementarity.py --repeats 100
.venv/bin/python evaluation/research/generate_causal_transformations.py
.venv/bin/python evaluation/research/prepare_visual_fidelity.py --workers 12
.venv/bin/python evaluation/research/prepare_state_interventions.py
.venv/bin/python evaluation/research/generate_perception_curriculum.py
```

Generated evaluation inputs are intentionally ignored by Git. Recreate them on
the GPU host after pulling the source. Validate all bundles before inference:

```bash
.venv/visual-suite/bin/python evaluation/research/validate_experiment_bundle.py \
  --manifest evaluation/research/results/causal_transformations/manifest.json \
  --manifest evaluation/research/results/visual_fidelity/manifest.json \
  --manifest evaluation/research/results/state_interventions/manifest.json \
  --manifest evaluation/research/results/perception_curriculum/manifest.json
```

The full local preparation produced:

| Bundle | Size |
| --- | ---: |
| Causal transformations | 200 problem triplets, 600 questions |
| Visual fidelity | 560 source items, 1,680 condition images |
| State interventions | 199 matched Mind's Eye items |
| Perception curriculum | 5,000 training and 500 validation pairs, 11,000 total examples |

## Current CPU findings

The canonical 14-model cohort has substantial item-level complementarity:

| Track | Best single-model macro | All-model item oracle | Universally missed |
| --- | ---: | ---: | ---: |
| Do You See Me | 38.83% | 79.10% | 1,039 of 4,500, 23.09% |
| Mind's Eye | 51.98% | 92.12% | 63 of 799, 7.88% |

The all-model oracle is an upper bound, not a usable system. The leakage-free
task router is more realistic:

| Track | Mean routed gain | Split-sensitivity interval | Positive repeats |
| --- | ---: | ---: | ---: |
| Do You See Me | +11.02 points | +10.40 to +11.53 points | 100 of 100 |
| Mind's Eye | -0.51 points | -2.07 to +1.00 points | 26 of 100 |

This supports a strong perception-specialization result. The cognition routing
gain is not stable enough to claim without additional validation.

Two useful pair-oracle observations are:

- Qwen3.5-9B thinking disabled plus Qwen2.5-VL-7B-Instruct reaches a
  56.86% perception item oracle, 20.74 points above the better member.
- Qwen3.6-27B plus GLM-4.6V-Flash reaches a 63.98% cognition item oracle,
  12.00 points above the better member.

The full tables and task selection frequencies are in
`analysis/results/research_experiments/complementarity/`.

## Research interpretation rules

1. Treat item oracles as diagnostic upper bounds only.
2. Treat repeated cross-validation folds as split sensitivity, not independent
   experimental replications.
3. Call the filename-derived Mind's Eye intervention an `oracle abstraction
   cue`, not an oracle perceptual state.
4. Count unresolved model answers as wrong. Rerun infrastructure and extractor
   failures rather than counting them as model errors.
5. Use unquantized BF16 checkpoint weights. The provided GPU commands do not use
   BitsAndBytes, AWQ, GPTQ, reduced image resolution, or early-exit shortcuts.
6. Keep model revision, prompt mode, sampling parameters, extractor revision,
   and seed fixed within every paired comparison.
7. Apply multiplicity correction when reporting many task-level or model-level
   tests. The condition scorer reports the primary paired test and interval.
8. Use at least three independent training seeds for the curriculum experiment.
   A single LoRA run is exploratory, not a paper result.
9. The provided Qwen training condition uses attention LoRA on `q_proj`,
   `k_proj`, `v_proj`, and `o_proj` while freezing the vision encoder. Interpret
   it as perception-focused supervision transfer, not direct vision-encoder
   adaptation.
10. Do not treat the earlier 8,192-token causal pilot as confirmatory evidence.
    The controlled queue above supersedes it with a fixed 32,768-token context,
    separate extraction, and explicit quality gates.

See `REMOTE_GPU_RUNBOOK.md` for executable remote commands.
