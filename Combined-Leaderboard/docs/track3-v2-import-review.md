# External Track 3 v2 review

Review date: 2026-09-15. No new scores were published by this review.

Input: `track3_submissions_v2.zip`, SHA-256
`f324880b4e621eb982b09115ccdfc57393f1af9e3a3997c9ca31ed69910ba37f`.
The original archive remains private and is not part of the repository.
Its embedded validator was read, not executed. No GPU or model call was used.

Models: GPT-5 (`gpt-5_2025-08-07`), GPT-4o (`gpt-4o_2024-11-20`),
Gemma-4-31B-it (`google/gemma-4-31B-it`). All declare `spatial-cot-eval-v2`
and Qwen3-30B-A3B-Instruct-2507 as the judge. That is submitter-declared
provenance, not proof of the judge/revision that ran.

## Checks

All three models contain six conditions, 13 datasets, unique prediction and
verdict keys, matching pair metadata, and no ground-truth fields in prediction
rows. All per-dataset scores, macro-average rows, and main-condition deltas
match aggregation of the **submitted** verdicts within 0.02 percentage points.
This checks arithmetic only, not independent semantic correctness.

They cannot be imported into the installed `paper-aligned-v5-2026-07` release:

| Condition (each prompt mode) | Supplied rows | Required live rows | Missing exact public IDs | Unknown IDs |
| --- | ---: | ---: | ---: | ---: |
| Main | 15,370 | 13,951 | 3,709 | 5,128 |
| No-image | 15,370 | 13,539 | 3,968 | 5,799 |
| No-image-plus | 14,672 | 12,923 | 3,665 | 5,414 |

Each model has 90,824 rows, versus 80,826 in the live release. Exact comparison
qualifies each supplied index with its dataset name; it never guesses a mapping
from numerical indices to release IDs. The 3DSRBench and MindCube ID schemes
differ. SAT-Real and SpatialBench also differ in condition coverage. Main and
no-image each have 327 overlapping RealWorldQA IDs with a different answer type.
This is a benchmark-contract mismatch, not merely ZIP naming.

Gemma additionally has 243 null model outputs: 130 main-CoT, 112 no-image-plus
direct, and one no-image-plus CoT. None receives positive submitted credit, but
missing output evidence still needs an explicit failure record under an approved
contract; the importer must not invent model responses.

## What is needed

Obtain exports for the installed public release, with exact public IDs, groups,
answer types and conditions, using its canonical artifact harness. Mapping IDs
alone does not resolve differing sample scopes/types. The experiment owner must
confirm whether existing outputs can faithfully support that export or whether
some conditions need rerunning. Do not assign the current manifest hash to a
different experiment or silently change reported scores.

Alternatively, obtain explicit approval for a separate, versioned leaderboard
view and freeze the v2 public contract. Do not directly rank it against InternVL's
existing release. Remove personal contact metadata before any public packaging.

Reproduce the review (exit 2 means blocked from canonical repackaging):

```sh
python3 scripts/review_track3_v2.py /private/track3_submissions_v2.zip
```
