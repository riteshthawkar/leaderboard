# Prespecified Track-3 Analysis Plan

Frozen before either paper-aligned Track-3 run was judged or scored.

## Evaluated models

1. `OpenGVLab/InternVL3_5-8B` at revision
   `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79`
2. `Qwen/Qwen3.6-27B` at revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`

Both runs use the paper-aligned v5 data, prompts, six conditions, unquantized
BF16 weights, greedy decoding, one recorded pass-at-1 seed, and the fixed
Qwen3-30B-A3B paper judge. SpatialBench and SAT-Real are scored as correct only
when every circular option rotation in an evaluation group is correct.

## Statistical unit and score

The inferential unit is the evaluation group, not an individual circular
rotation. Every contrast is paired on dataset, evaluation group, and answer
type. Accuracy is first calculated within each of the 13 datasets and then
macro-averaged with equal dataset weight, matching the leaderboard score.

Confidence intervals use paired within-dataset bootstrap resampling. Two-sided
p-values use paired label-swap randomization under the same equal-dataset
macro weighting. The random seed is `20260723`.

## Primary tests

Four tests are run for each model:

1. Main CoT minus main non-CoT accuracy.
2. Main non-CoT minus No-Image non-CoT accuracy, measuring image dependence.
3. No-Image++ non-CoT minus No-Image non-CoT on shared MCQ groups, measuring
   recovery when explicit abstention is available.
4. The CoT-by-image interaction:
   `(main CoT - main non-CoT) - (No-Image CoT - No-Image non-CoT)`.

Three cross-model tests compare Qwen3.6-27B against InternVL3.5-8B:

1. Main non-CoT accuracy difference.
2. Difference in the main CoT effect.
3. Difference in non-CoT image dependence.

These 11 two-sided tests form one confirmatory family and receive a single
Holm correction at alpha `0.05`.

## Descriptive analyses

Per-dataset condition accuracy, 2D/3D summaries, MCQ/VQA splits, judge-method
counts, completion stop reasons, and secondary contrasts are descriptive.
They may motivate future work but are not additional confirmatory claims.

## Quality gates

Analysis stops unless every expected judged row is present, unique, non-empty,
and resolved. The source run configurations and judged evidence files are
hashed. Private ground truth and reasoning traces remain outside public
submission packages.

## Interpretation limits

The model cohort contains only two checkpoints and one deterministic decoding
seed. Item-level resampling does not estimate model-family or decoding-seed
variance. Cross-model conclusions are therefore controlled case comparisons,
not general scaling laws.
