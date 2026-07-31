# MS-VISTA Controlled Analysis Plan

## Status

This plan was frozen before the controlled GPU outputs were scored. Operational
monitoring may inspect process health, row coverage, finish reasons, token
counts, and infrastructure errors. It must not inspect condition accuracies or
stop a run based on an interim effect.

Analysis ID: `ms-vista-controlled-analysis-v2`

## Scope

The confirmatory queue evaluates two unquantized BF16 checkpoints:

| Model | Revision |
| --- | --- |
| Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| InternVL3.5-8B | `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79` |

Every condition uses deterministic decoding with temperature 0, top-p 1, seed
0, and a 32,768-token server context. Requests use the remaining context rather
than an artificial completion cap. A separate pinned Qwen3-8B checkpoint
(`b968826d9c46dd6066d109eabc6255188de91218`) performs gold-blind text-only
answer extraction after all target-model inference is complete.

## Confirmatory Experiments

### Causal visual transformations

Each model receives 200 matched triplets under direct-answer and CoT prompts:

1. A base image.
2. An answer-changing causal edit.
3. An answer-preserving nuisance-style edit.

The primary endpoint is full-triplet success. A triplet succeeds only when the
base, causal, and nuisance answers are all correct and the base and nuisance
commitments remain invariant. The primary prompt effect is CoT minus direct
full-triplet success.

### Visual fidelity

The same items are evaluated at native fidelity and after 50% and 25%
downsampling followed by restoration to the original canvas size. Do You See
Me uses its direct-answer prompt, and Mind's Eye uses its reasoning prompt.

The primary endpoints are the benchmark-macro changes from native fidelity at
each degradation level. The design uses 360 stratified Do You See Me items and
200 task-balanced Mind's Eye items.

### Abstraction cues

The 199 eligible Mind's Eye items are evaluated under:

1. Baseline prompt.
2. Correct answer-blind abstraction cue.
3. Deterministically mismatched cue from another item in the same task.

The primary endpoints are correct-cue minus baseline and mismatched-cue minus
baseline benchmark-macro accuracy. These are filename-derived operation or
concept cues, not oracle perceptual states.

## Cross-Model Interactions

For every primary endpoint, a paired difference-in-differences compares the
condition effect in Qwen3-VL-8B-Instruct with the effect in InternVL3.5-8B.
This distinguishes effects shared across model families from architecture-
specific sensitivity. Model labels are swapped within matched items for the
randomization test.

## Quality Gate

Each condition must have exact question, diagnostic, and submission coverage.
The raw response must be nonempty and free of inference errors. Finish reason
`stop` and finish reason `length` are both model outcomes. Fixed-extractor
model, revision, method, and source-response hash must match the contract.

Unresolved or length-capped responses count as incorrect model outcomes.
Inference failures, extractor failures, coverage drift, hash drift, and
provenance drift fail the condition and must be rerun. No output is manually
repaired, discarded for poor format, or replaced using a reference answer.

## Statistical Analysis

- Benchmark effects retain the production macro aggregation.
- All intervals use 10,000 paired, task-stratified bootstrap replicates.
- Fidelity and abstraction effects use paired label swaps under the production
  benchmark-macro weights.
- The equally weighted causal triplets use exact McNemar tests.
- Cross-model interactions use 10,000 paired label-swap replicates.
- One Holm correction covers 21 prespecified tests: seven endpoints for each
  of two models and seven corresponding cross-model interactions.
- An effect is confirmatory only when its paired 95% interval excludes zero
  and its global Holm-adjusted p-value is below 0.05.
- Task and operation breakdowns are exploratory. Subgroup comparisons use Holm
  correction within each condition-versus-baseline family.

## Track 3

The spatial Track 3 evaluation is a separate benchmark replication, not part of
the 21-test controlled family. It evaluates main, no-image, and no-image-plus
conditions under direct and CoT prompts. The paper-aligned fixed judge scores
MCQ commitments and short VQA answers. Every dataset is reported separately
before macro aggregation.

## Reporting Rules

1. Report null and adverse effects alongside improvements.
2. Do not interpret item bootstraps as model-training replications.
3. Do not call item oracles deployable systems.
4. Do not generalize a two-model interaction to all VLM architectures.
5. Do not combine Track 3 with VPCI until its full evidence package passes the
   six-condition coverage and judge-provenance checks.
6. Clearly separate confirmatory primary endpoints from exploratory subgroup
   observations.
