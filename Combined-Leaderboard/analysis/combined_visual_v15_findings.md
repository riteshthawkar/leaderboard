# Combined Visual Evaluation: Canonical v15 Findings

## Status

The 14-model cohort is now internally consistent and reproducible under the
production `qwen3-8b-gold-blind-evidence-extractor-v4` contract.

The canonical bundle contains 74,186 responses: 4,500 Do You See Me items and
799 Mind's Eye items for each model variant. The verifier found no raw-response
hash mismatch. The answer extractor received neither images nor reference
answers, and every non-committed output remains an explicit incorrect result.

This makes the bundle suitable for the production leaderboard. Publication
claims still need to distinguish visual task performance from answer-commitment
reliability and decoding-budget effects.

## Scoring

| Score | Definition |
| --- | --- |
| Perception | Mean task accuracy within 2D and 3D, followed by an equal mean of the two dimensions |
| Cognition | Unweighted mean of the eight Mind's Eye task accuracies |
| VPCI | Arithmetic mean of the perception and cognition scores |

The evaluated suite is a pinned expanded item set, not the exact item
distribution from either source paper. Paper human values are therefore
contextual references rather than matched-sample controls.

## Ranking

| Rank | Model | Perception | Cognition | VPCI |
| ---: | --- | ---: | ---: | ---: |
| 1 | Qwen3.6-27B | 36.87 | 51.98 | 44.43 |
| 2 | GLM-4.6V-Flash | 38.83 | 39.18 | 39.00 |
| 3 | Qwen3.5-9B, thinking disabled | 34.30 | 39.33 | 36.82 |
| 4 | DeepSeek-VL2 | 36.01 | 34.45 | 35.23 |
| 5 | Qwen3-VL-8B-Instruct | 35.71 | 32.84 | 34.27 |
| 6 | Qwen2.5-VL-7B-Instruct | 36.12 | 30.57 | 33.34 |
| 7 | MiniCPM-V 4.6 | 25.28 | 35.20 | 30.24 |
| 8 | Qwen3.5-9B, thinking enabled | 30.28 | 29.48 | 29.88 |
| 9 | InternVL3.5-8B, thinking | 25.09 | 31.33 | 28.21 |
| 10 | Gemma 3 27B IT | 24.86 | 26.30 | 25.58 |
| 11 | Phi-4 Multimodal Instruct | 22.34 | 26.53 | 24.43 |
| 12 | Kimi-VL-A3B-Instruct | 27.46 | 20.67 | 24.06 |
| 13 | Gemma 3 12B IT | 22.79 | 21.80 | 22.30 |
| 14 | Llama 3.2 11B Vision Instruct | 17.49 | 15.90 | 16.70 |

## Main Findings

### 1. Perception and cognition share competence but remain distinct

Across the 14 variants, perception and cognition correlate at Pearson
`r = 0.751`, with a model-bootstrap 95% interval of `[0.498, 0.906]`, and
Spearman `rho = 0.736`. The association remains positive when one variant is
selected per model family: Pearson ranges from `0.725` to `0.802` across the
ten family-balanced cohorts.

The first principal component explains `33.90%` of task-profile variance, while
the second explains `23.14%`. This supports shared visual competence plus a
substantial specialization axis. It does not support treating perception and
cognition as interchangeable.

### 2. The top ranking is stable

Qwen3.6-27B is rank one in every one of 10,000 shared stratified bootstrap
replicates. Its 5.42-point VPCI lead over GLM-4.6V-Flash has a 95% interval of
`[3.37, 7.47]` points.

Across 5,000 matched split halves, median VPCI rank correlation is `0.982`, the
top model agrees in every replicate, and median top-three Jaccard overlap is
`1.0`. Exact lower ranks should still be treated as tiers rather than precise
scientific orderings.

### 3. Aggregation choice does not drive the broad conclusion

Arithmetic versus geometric and harmonic VPCI each have rank correlation
`rho = 0.996`; both alternatives preserve the top six and only swap ranks
seven and eight. A bottleneck score based on the weaker layer has
`rho = 0.982`.

The Pareto frontier contains Qwen3.6-27B and GLM-4.6V-Flash. Qwen3.6 leads while
perception receives up to `86.7%` of the combined weight. GLM leads only when
perception receives at least `86.8%`.

### 4. Complexity produces a persistent perception hard tail

Mean perception accuracy falls from `37.51%` on easy items to `25.51%` on
medium items and `17.88%` on hard items. The easy-to-hard gap is `19.63`
points. It remains `17.23` points after removing form discrimination.

Six low-information strata contain 772 items, or `17.2%` of the perception
benchmark. Removing them changes VPCI rank correlation only to `0.996` and
does not change the top model. They must still be disclosed as a measurement
floor.

### 5. Several cognition tasks remain at or below chance

Correcting for four-option versus six-option tasks changes the cognition
ordering by at most one rank (`rho = 0.996`) and keeps Qwen3.6-27B first.

The cohort mean remains below chance after normalization on dynamic reasoning
(`-10.19` points), mental rotation (`-6.95`), and spatial visualization
(`-3.14`). Conceptual slippage (`+42.86`), hierarchical reasoning (`+30.95`),
and mental composition (`+22.29`) are substantially stronger.

### 6. Model errors are complementary, especially for perception

The all-model oracle reaches `79.10%` perception and `92.12%` cognition,
roughly 40 points above the best single model on each benchmark. This is an
upper bound, not a deployable system.

In leakage-free five-fold task routing repeated 100 times, selecting a
perception model by task raises held-out macro accuracy from `38.83%` to
`49.84%` on average. The mean gain is `11.02` points and is positive in every
repeat. The same method does not improve cognition: its mean change is
`-0.51` points and is positive in only `26%` of repeats.

This supports task-specific architectural complementarity in perception. A
fresh independently generated item set is still needed before presenting
routing as a headline generalization result.

### 7. Shared blind spots remain large

All 14 models miss 1,039 of 4,500 perception items (`23.1%`) and 63 of 799
cognition items (`7.9%`). Only 44 perception items are solved by every model,
and no cognition item is solved by every model.

The result shows that aggregate improvement has not eliminated common failure
regions. It also explains why the model oracle remains far above the best
single checkpoint.

### 8. Native thinking is confounded by commitment and output budget

For the same Qwen3.5-9B revision, thinking enabled scores 4.03 perception
points and 9.86 cognition points below thinking disabled. The bootstrap
intervals are `[-5.68, -2.38]` and `[-13.23, -6.48]` points, respectively.

This is not clean evidence that thinking reduces visual capability. The
thinking-enabled run reaches the output limit on `31.4%` of perception items
and `55.2%` of cognition items. Under the strict evidence contract, 2,551
perception responses and 472 cognition responses are unresolved or unsupported.

The defensible conclusion is that unrestricted reasoning under a shared finite
output budget can reduce answer-commitment reliability. A final-answer reserve
and additional decoding seeds are required to isolate the effect of thinking
on capability.

### 9. Gemma scaling is a descriptive case, not a scaling law

Gemma 3 27B exceeds Gemma 3 12B by 2.07 perception points, with a 95% interval
of `[0.87, 3.28]`, and by 4.49 cognition points, with an interval of
`[1.11, 8.00]`. The cross-layer interaction interval includes zero.

This supports an improvement for this one family and setup. It does not
establish a general parameter-scaling law or a layer-selective scaling effect.

### 10. Specific cross-layer mechanisms are not supported

None of 56 perception-cognition task associations survives false-discovery
correction. The result remains null after controlling for competence estimated
from the other 13 tasks and after averaging variants within model family.

Specific claims such as one perception task enabling one cognition task should
not be made from this cohort.

## Protocol Interpretation

The v4 audit records:

| Outcome | Count | Share |
| --- | ---: | ---: |
| Evidence-supported commitment | 50,960 | 68.7% |
| Explicit invalid-format commitment | 15,353 | 20.7% |
| Unresolved, truncated, or unsupported | 7,873 | 10.6% |

These are not missing rows. Invalid and unresolved outcomes are scored as
wrong by design. The leaderboard therefore measures task accuracy under a
strict answer-commitment protocol, not an unobservable upper bound on latent
model knowledge.

The extractor used 25 deterministic terminal fallbacks after repeated schema
failures. Every fallback is recorded in the evidence files and reproduced by
the checked-out validator.

## Interpretation Limits

1. The model is the unit for cross-model inference, so correlation analyses use
   only 14 non-independent variants from nine families.
2. Each variant has one stochastic evaluation run. Item bootstraps do not
   estimate decoding-seed variance.
3. The expanded item set differs from the source papers' exact test samples.
4. Human values from the papers are not matched-item baselines.
5. Output-format and answer-budget failures are part of the measured score.
6. The spatial-reasoning track is not included in these combined results.

## Reproduction

```bash
PYTHONPATH="$PWD:$PWD/backend" GROUND_TRUTHS_DIR="$PWD/Ground_truths" \
  .venv/bin/python analysis/combined_visual_v14.py

PYTHONPATH="$PWD:$PWD/backend" GROUND_TRUTHS_DIR="$PWD/Ground_truths" \
  .venv/bin/python analysis/submission_readiness_v14.py

PYTHONPATH="$PWD:$PWD/backend" GROUND_TRUTHS_DIR="$PWD/Ground_truths" \
  .venv/bin/python analysis/research/offline_complementarity.py
```

The script filenames are retained for compatibility. All default inputs and
outputs point to the canonical v15 bundle.
